#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from sql_reasoner.experiments import (
    apply_overrides,
    expand_dependencies,
    paired_execution,
    select_experiments,
    stable_hash,
    validate_profile,
)
from sql_reasoner.io import load_yaml, read_json, write_json


def display_command(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    body = shlex.join(command)
    return f"{prefix} {body}" if prefix else body


def run_with_log(
    command: list[str], *, env_updates: dict[str, str], log_path: Path, dry_run: bool
) -> None:
    rendered = display_command(command, env_updates)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(env_updates)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def load_matrix(path: str) -> dict[str, Any]:
    matrix = load_yaml(path)
    required = ["suite", "base_configs", "resource_profiles", "experiments", "evaluation"]
    missing = [key for key in required if key not in matrix]
    if missing:
        raise ValueError(f"实验矩阵缺少字段: {missing}")
    return matrix


def experiment_paths(matrix: dict[str, Any], name: str) -> dict[str, Path]:
    suite = matrix["suite"]
    checkpoint = Path(matrix["paths"]["checkpoint_root"]) / suite / name
    root = Path(matrix["paths"]["output_root"]) / suite / name
    return {
        "checkpoint": checkpoint,
        "root": root,
        "config": root / "resolved_config.yaml",
        "metadata": root / "metadata.json",
        "train_log": root / "train.log",
    }


def resolve_config(
    matrix: dict[str, Any], name: str, paths: dict[str, Path]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    spec = matrix["experiments"][name]
    stage = spec["stage"]
    if stage == "base":
        return None, None
    profile = matrix["resource_profiles"][spec["resource_profile"]]
    if profile["stage"] != stage:
        raise ValueError(f"{name}: profile stage={profile['stage']} 与 stage={stage} 不一致")
    base = load_yaml(matrix["base_configs"][stage])
    overrides = spec.get("overrides", {})
    resolved = apply_overrides(base, overrides)
    validate_profile(resolved, profile, overrides)
    resolved["output_dir"] = str(paths["checkpoint"])
    if stage == "rl":
        dependency = spec.get("depends_on")
        if not dependency:
            raise ValueError(f"{name}: RL 实验必须声明 depends_on SFT 实验")
        dependency_paths = experiment_paths(matrix, dependency)
        resolved["sft_adapter"] = str(dependency_paths["checkpoint"])
        resolved.setdefault("data", {})["train"] = matrix["paths"]["rl_data"]
    return resolved, profile


def materialize(
    matrix: dict[str, Any], name: str
) -> tuple[dict[str, Path], dict[str, Any] | None, dict[str, Any] | None, str]:
    paths = experiment_paths(matrix, name)
    resolved, profile = resolve_config(matrix, name, paths)
    fingerprint = stable_hash(
        {
            "experiment": matrix["experiments"][name],
            "profile": profile,
            "resolved": resolved,
        }
    )
    paths["root"].mkdir(parents=True, exist_ok=True)
    if resolved is not None:
        paths["config"].write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    write_json(
        paths["metadata"],
        {
            "name": name,
            "suite": matrix["suite"],
            "stage": matrix["experiments"][name]["stage"],
            "fingerprint": fingerprint,
            "resource_profile": matrix["experiments"][name].get("resource_profile"),
            "compare_to": matrix["experiments"][name].get("compare_to"),
        },
    )
    return paths, resolved, profile, fingerprint


def completed(marker: Path, fingerprint: str) -> bool:
    if not marker.is_file():
        return False
    try:
        return read_json(marker).get("fingerprint") == fingerprint
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def mark_completed(marker: Path, fingerprint: str) -> None:
    write_json(marker, {"fingerprint": fingerprint, "completed_at": time.time()})


def train_one(
    matrix: dict[str, Any],
    name: str,
    *,
    dry_run: bool,
    force: bool,
    resume: str,
) -> None:
    paths, resolved, profile, fingerprint = materialize(matrix, name)
    stage = matrix["experiments"][name]["stage"]
    if stage == "base":
        return
    marker = paths["root"] / "train.done.json"
    if completed(marker, fingerprint) and not force:
        print(f"[skip] {name}: 训练已完成且配置未变化")
        return
    if marker.is_file() and not force:
        raise RuntimeError(
            f"{name}: 已有训练结果，但当前配置指纹已变化。"
            "请使用新的实验名，或确认后传入 --force 从头重跑；"
            "禁止自动续接不兼容的 checkpoint。"
        )
    assert resolved is not None and profile is not None
    common_env = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    if stage == "sft":
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            str(profile["launch"]["num_processes"]),
            "scripts/train_sft.py",
            "--config",
            str(paths["config"]),
            "--resume",
            resume,
        ]
        env_updates = common_env
    else:
        data_path = Path(resolved["data"]["train"])
        adapter_path = Path(resolved["sft_adapter"])
        if not dry_run and not data_path.is_file():
            raise FileNotFoundError(
                f"RL 数据不存在: {data_path}。实验运行器应先执行统一奖励方差筛选。"
            )
        if not dry_run and not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"SFT adapter 不完整: {adapter_path}")
        command = ["bash", "scripts/run_grpo.sh"]
        env_updates = {
            **common_env,
            **{key: str(value) for key, value in profile.get("env", {}).items()},
            "CONFIG": str(paths["config"]),
            "RESUME": resume,
        }
    run_with_log(command, env_updates=env_updates, log_path=paths["train_log"], dry_run=dry_run)
    if not dry_run:
        mark_completed(marker, fingerprint)


def ensure_rl_data(
    matrix: dict[str, Any],
    dependency: str,
    *,
    dry_run: bool,
    force: bool,
) -> None:
    selection = matrix["rl_selection"]
    data_path = Path(matrix["paths"]["rl_data"])
    stats_path = Path(selection["stats_output"])
    dependency_checkpoint = experiment_paths(matrix, dependency)["checkpoint"]
    fingerprint = stable_hash(
        {
            "selection": selection,
            "adapter": checkpoint_signature(dependency_checkpoint),
        }
    )
    marker = data_path.with_suffix(data_path.suffix + ".done.json")
    if completed(marker, fingerprint) and data_path.is_file():
        print(f"[skip] RL 数据已由相同 SFT checkpoint 和筛选配置生成: {data_path}")
        return
    if marker.exists() and not force:
        raise RuntimeError(
            f"RL 数据筛选来源或配置已变化: {data_path}。确认后使用 --force 重新生成。"
        )
    if not dry_run and not (dependency_checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(f"用于奖励方差筛选的 SFT adapter 不存在: {dependency_checkpoint}")
    command = [
        sys.executable,
        "scripts/select_grpo_data.py",
        "--input",
        selection["input"],
        "--output",
        str(data_path),
        "--stats-output",
        str(stats_path),
        "--adapter",
        str(dependency_checkpoint),
        "--candidates",
        str(selection["candidates"]),
        "--max-new-tokens",
        str(selection["max_new_tokens"]),
        "--limit",
        str(selection["limit"]),
        "--max-scan",
        str(selection["max_scan"]),
        "--strategy",
        selection["strategy"],
        "--seed",
        str(selection["seed"]),
    ]
    log_path = Path(matrix["paths"]["output_root"]) / matrix["suite"] / "_rl_data" / "selection.log"
    run_with_log(
        command,
        env_updates={
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
        },
        log_path=log_path,
        dry_run=dry_run,
    )
    if not dry_run:
        mark_completed(marker, fingerprint)


def checkpoint_signature(checkpoint: Path | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    adapter = checkpoint / "adapter_model.safetensors"
    if not adapter.is_file():
        return {"missing": str(adapter)}
    stat = adapter.stat()
    return {"path": str(adapter), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def evaluate_one(
    matrix: dict[str, Any],
    name: str,
    *,
    split: str,
    dry_run: bool,
    force: bool,
    limit: int | None,
) -> None:
    paths, _, _, fingerprint = materialize(matrix, name)
    stage = matrix["experiments"][name]["stage"]
    checkpoint = None if stage == "base" else paths["checkpoint"]
    eval_fingerprint = stable_hash(
        {
            "config": fingerprint,
            "split": split,
            "dataset": matrix["evaluation"][split],
            "limit": limit,
            "checkpoint": checkpoint_signature(checkpoint),
        }
    )
    output = paths["root"] / "eval" / split
    marker = output / "eval.done.json"
    if completed(marker, eval_fingerprint) and not force:
        print(f"[skip] {name}: {split} 评测已完成且配置未变化")
        return
    if (
        checkpoint is not None
        and not dry_run
        and not (checkpoint / "adapter_config.json").is_file()
    ):
        raise FileNotFoundError(f"评测 adapter 不存在: {checkpoint}")
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--dataset",
        matrix["evaluation"][split],
        "--output-dir",
        str(output),
        "--model",
        matrix["model_id"],
    ]
    if checkpoint is not None:
        command.extend(["--adapter", str(checkpoint)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    run_with_log(
        command,
        env_updates={
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "CUDA_VISIBLE_DEVICES": "0",
        },
        log_path=output / "evaluate.log",
        dry_run=dry_run,
    )
    if not dry_run:
        mark_completed(marker, eval_fingerprint)


def summarize(matrix: dict[str, Any], names: list[str], split: str) -> None:
    records = []
    predictions: dict[str, Path] = {}
    for name in names:
        paths = experiment_paths(matrix, name)
        eval_dir = paths["root"] / "eval" / split
        summary_path = eval_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json(summary_path)
        record = {
            "experiment": name,
            "stage": matrix["experiments"][name]["stage"],
            "compare_to": matrix["experiments"][name].get("compare_to"),
            **summary,
        }
        train_results = paths["checkpoint"] / "train_results.json"
        if train_results.is_file():
            metrics = read_json(train_results)
            record["train_runtime"] = metrics.get("train_runtime")
            record["train_loss"] = metrics.get("train_loss")
        vram_files = list(paths["checkpoint"].glob("vram*.json"))
        if vram_files:
            vram = [read_json(path) for path in vram_files]
            record["peak_reserved_bytes"] = max(
                value.get("peak_reserved_bytes", 0) for value in vram
            )
        records.append(record)
        predictions[name] = eval_dir / "predictions.jsonl"

    for record in records:
        baseline = record.get("compare_to")
        if baseline in predictions:
            record["mcnemar_vs_compare_to"] = paired_execution(
                predictions[baseline], predictions[record["experiment"]]
            )

    root = Path(matrix["paths"]["output_root"]) / matrix["suite"]
    write_json(root / f"comparison_{split}.json", records)
    lines = [
        f"# {matrix['suite']} — {split}",
        "",
        "| experiment | stage | execution | valid SQL | format | schema F1 | peak VRAM GiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        peak = row.get("peak_reserved_bytes")
        peak_text = f"{peak / 2**30:.2f}" if peak else "-"
        lines.append(
            f"| {row['experiment']} | {row['stage']} | "
            f"{row.get('execution_accuracy', 0):.2%} | {row.get('valid_sql_rate', 0):.2%} | "
            f"{row.get('format_rate', 0):.2%} | {row.get('schema_linking_f1', 0):.2%} | "
            f"{peak_text} |"
        )
    (root / f"comparison_{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总了 {len(records)} 个实验: {root / f'comparison_{split}.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-locked training and ablation runner")
    parser.add_argument("--matrix", default="configs/experiments.yaml")
    parser.add_argument(
        "--action",
        choices=["list", "materialize", "train", "eval", "all", "summarize"],
        default="list",
    )
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--eval-split", choices=["tuning", "final"], default="tuning")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", choices=["auto", "none"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.action != "list" and not args.group and not args.experiment:
        parser.error("执行操作必须显式提供至少一个 --group 或 --experiment，防止误跑全部实验")
    matrix = load_matrix(args.matrix)
    experiments = matrix["experiments"]
    selected = select_experiments(experiments, args.experiment, args.group)
    explicitly_selected = set(selected)
    if args.action in {"train", "all"}:
        selected = expand_dependencies(experiments, selected)

    if args.action == "list":
        for name in selected:
            spec = experiments[name]
            print(
                f"{name:20} stage={spec['stage']:4} "
                f"groups={','.join(spec.get('groups', [])) or '-'} "
                f"depends_on={spec.get('depends_on', '-')}"
            )
        return
    if args.action == "summarize":
        summarize(matrix, selected, args.eval_split)
        return

    prepared_rl_data_dependencies: set[str] = set()
    for name in selected:
        force_this_experiment = args.force and name in explicitly_selected
        if args.action == "materialize":
            paths, _, _, _ = materialize(matrix, name)
            print(f"{name}: {paths['metadata']}")
        if args.action in {"train", "all"}:
            spec = experiments[name]
            dependency = spec.get("depends_on")
            if spec["stage"] == "rl" and dependency not in prepared_rl_data_dependencies:
                ensure_rl_data(
                    matrix,
                    dependency,
                    dry_run=args.dry_run,
                    force=force_this_experiment,
                )
                prepared_rl_data_dependencies.add(dependency)
            train_one(
                matrix,
                name,
                dry_run=args.dry_run,
                force=force_this_experiment,
                resume="none" if force_this_experiment else args.resume,
            )
        if args.action in {"eval", "all"}:
            evaluate_one(
                matrix,
                name,
                split=args.eval_split,
                dry_run=args.dry_run,
                force=force_this_experiment,
                limit=args.limit,
            )
    if args.action in {"eval", "all"} and not args.dry_run:
        summarize(matrix, selected, args.eval_split)


if __name__ == "__main__":
    main()
