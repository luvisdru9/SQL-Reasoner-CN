#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from sql_reasoner.campaign import (
    hard_execution_accuracy,
    quality_key,
    select_best,
    summarize_repeats,
)
from sql_reasoner.experiments import stable_hash
from sql_reasoner.io import load_yaml, read_json, write_json


SFT_CANDIDATES = ["sft_default", "sft_lr_1e4", "sft_lr_3e4"]
RL_ALGORITHM_CANDIDATES = [
    "rl_grpo",
    "rl_dapo_loss",
    "rl_dapo_clip",
    "rl_dapo_components",
]
RL_TUNING_VARIANTS = ["rl_lr_5e6", "rl_lr_2e5", "rl_temp_06", "rl_temp_10"]
RL_REWARD_VARIANTS = ["rl_reward_execution_only", "rl_reward_no_plan"]
RL_CONFIRMATION_VARIANTS = ["rl_seed_7", "rl_seed_123"]


def render(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{prefix} {shlex.join(command)}" if prefix else shlex.join(command)


def run_command(
    command: list[str], *, env_updates: dict[str, str], log_path: Path, dry_run: bool
) -> None:
    rendered = render(command, env_updates)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    environment.update(env_updates)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
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
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def experiment_root(matrix: dict[str, Any], name: str) -> Path:
    return Path(matrix["paths"]["output_root"]) / matrix["suite"] / name


def checkpoint_path(matrix: dict[str, Any], name: str) -> Path:
    return Path(matrix["paths"]["checkpoint_root"]) / matrix["suite"] / name


def summary_path(matrix: dict[str, Any], name: str, split: str = "tuning") -> Path:
    return experiment_root(matrix, name) / "eval" / split / "summary.json"


def prediction_path(matrix: dict[str, Any], name: str, split: str = "tuning") -> Path:
    return experiment_root(matrix, name) / "eval" / split / "predictions.jsonl"


def file_signature(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def marker_matches(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        return read_json(path).get("fingerprint") == fingerprint
    except (OSError, AttributeError, ValueError):
        return False


def collect_summaries(
    matrix: dict[str, Any], names: list[str], split: str = "tuning"
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for name in names:
        path = summary_path(matrix, name, split)
        if path.is_file():
            summaries[name] = read_json(path)
    return summaries


def selection_record(
    phase: str, candidates: list[str], summaries: dict[str, dict[str, Any]], winner: str
) -> dict[str, Any]:
    return {
        "phase": phase,
        "winner": winner,
        "ranking_rule": (
            "execution accuracy, then count-weighted hard/extra-hard execution, valid SQL, "
            "schema F1, format rate, AST exact match"
        ),
        "candidates": {
            name: {
                "quality_key": quality_key(summaries[name]),
                "execution_accuracy": summaries[name].get("execution_accuracy"),
                "hard_execution_accuracy": hard_execution_accuracy(summaries[name]),
                "valid_sql_rate": summaries[name].get("valid_sql_rate"),
                "schema_linking_f1": summaries[name].get("schema_linking_f1"),
            }
            for name in candidates
        },
    }


def save_matrix(matrix: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(matrix, allow_unicode=True, sort_keys=False), encoding="utf-8")


def configure_rl_parent(matrix: dict[str, Any], best_sft: str) -> None:
    safe_name = best_sft.replace("/", "_")
    data_root = Path("data/processed/cspider")
    matrix["paths"]["rl_data"] = str(data_root / f"grpo_train_{matrix['suite']}_{safe_name}.jsonl")
    matrix["rl_selection"]["stats_output"] = str(
        data_root / f"grpo_train_{matrix['suite']}_{safe_name}.stats.json"
    )
    for spec in matrix["experiments"].values():
        if spec["stage"] != "rl":
            continue
        spec["depends_on"] = best_sft
        if spec.get("compare_to") == "sft_default":
            spec["compare_to"] = best_sft


def inherit_overrides(
    matrix: dict[str, Any], source: str, targets: list[str], *, compare_to: str
) -> None:
    inherited = copy.deepcopy(matrix["experiments"][source].get("overrides", {}))
    dependency = matrix["experiments"][source]["depends_on"]
    for target in targets:
        own = copy.deepcopy(matrix["experiments"][target].get("overrides", {}))
        matrix["experiments"][target]["overrides"] = {**inherited, **own}
        matrix["experiments"][target]["depends_on"] = dependency
        matrix["experiments"][target]["compare_to"] = compare_to


def runner_command(
    matrix_path: Path,
    action: str,
    experiments: list[str],
    *,
    split: str = "tuning",
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_experiments.py",
        "--matrix",
        str(matrix_path),
        "--action",
        action,
        "--eval-split",
        split,
    ]
    for name in experiments:
        command.extend(["--experiment", name])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if force:
        command.append("--force")
    if dry_run:
        command.append("--dry-run")
    return command


def run_experiment_phase(
    matrix_path: Path,
    experiments: list[str],
    *,
    log_path: Path,
    limit: int | None,
    force: bool,
    dry_run: bool,
) -> None:
    run_command(
        runner_command(
            matrix_path,
            "all",
            experiments,
            limit=limit,
            force=force,
            dry_run=dry_run,
        ),
        env_updates={"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")},
        log_path=log_path,
        dry_run=False,
    )


def choose(
    matrix: dict[str, Any],
    candidates: list[str],
    *,
    phase: str,
    campaign_root: Path,
    dry_run: bool,
    dry_run_default: str,
) -> str:
    if dry_run:
        print(f"[dry-run selection] {phase}: {dry_run_default}")
        return dry_run_default
    summaries = collect_summaries(matrix, candidates)
    winner = select_best(candidates, summaries)
    write_json(
        campaign_root / f"selection_{phase}.json",
        selection_record(phase, candidates, summaries, winner),
    )
    print(f"[{phase} winner] {winner}: {quality_key(summaries[winner])}")
    return winner


def run_final_dataset(
    matrix: dict[str, Any],
    dataset_name: str,
    dataset_path: str,
    finalists: dict[str, str],
    *,
    final_root: Path,
    log_path: Path,
    force: bool,
    dry_run: bool,
) -> None:
    evaluations_changed = False
    for role, experiment in finalists.items():
        output = final_root / dataset_name / role
        adapter = None if role == "base" else checkpoint_path(matrix, experiment)
        fingerprint = stable_hash(
            {
                "dataset": file_signature(Path(dataset_path)),
                "model": matrix["model_id"],
                "role": role,
                "experiment": experiment,
                "adapter": (
                    None
                    if adapter is None
                    else file_signature(adapter / "adapter_model.safetensors")
                ),
            }
        )
        marker = output / "eval.done.json"
        if marker_matches(marker, fingerprint) and not force:
            print(f"[skip] {dataset_name}/{role}: 已评测")
            continue
        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--dataset",
            dataset_path,
            "--output-dir",
            str(output),
            "--model",
            matrix["model_id"],
        ]
        if adapter is not None:
            command.extend(["--adapter", str(adapter)])
        run_command(
            command,
            env_updates={
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                "CUDA_VISIBLE_DEVICES": "0",
            },
            log_path=log_path,
            dry_run=dry_run,
        )
        if not dry_run:
            write_json(marker, {"fingerprint": fingerprint})
            evaluations_changed = True
    comparison = final_root / dataset_name / "comparison.json"
    if not dry_run and (force or evaluations_changed or not comparison.is_file()):
        run_command(
            [
                sys.executable,
                "scripts/compare_results.py",
                "--base",
                str(final_root / dataset_name / "base" / "predictions.jsonl"),
                "--sft",
                str(final_root / dataset_name / "sft" / "predictions.jsonl"),
                "--grpo",
                str(final_root / dataset_name / "rl" / "predictions.jsonl"),
                "--output",
                str(comparison),
            ],
            env_updates={},
            log_path=log_path,
            dry_run=False,
        )


def maybe_run_ceval(
    matrix: dict[str, Any],
    finalists: dict[str, str],
    *,
    final_root: Path,
    log_path: Path,
    force: bool,
    dry_run: bool,
) -> None:
    for role, experiment in finalists.items():
        output = final_root / "ceval" / f"{role}.json"
        if output.is_file() and not force:
            print(f"[skip] C-Eval/{role}: 已评测")
            continue
        command = [sys.executable, "scripts/evaluate_ceval.py", "--output", str(output)]
        if role != "base":
            command.extend(["--adapter", str(checkpoint_path(matrix, experiment))])
        run_command(
            command,
            env_updates={"CUDA_VISIBLE_DEVICES": "0"},
            log_path=log_path,
            dry_run=dry_run,
        )


def write_final_report(
    matrix: dict[str, Any],
    finalists: dict[str, str],
    deployment: str,
    campaign_root: Path,
) -> None:
    validation = collect_summaries(matrix, list(dict.fromkeys(finalists.values())))
    official = collect_summaries(matrix, list(dict.fromkeys(finalists.values())), "final")
    external = {
        dataset: {
            role: read_json(campaign_root / "final" / dataset / role / "summary.json")
            for role in finalists
        }
        for dataset in ("private", "robustness")
    }
    state = read_json(campaign_root / "state.json")
    repeat_path = campaign_root / "rl_seed_stability.json"
    repeat_summary = (
        read_json(repeat_path) if state.get("preset") == "full" and repeat_path.is_file() else None
    )
    base_validation = validation[finalists["base"]]
    sft_validation = validation[finalists["sft"]]
    rl_validation = validation[finalists["rl"]]
    targets = {
        "sft_execution_gain_at_least_5pp": (
            sft_validation["execution_accuracy"] >= base_validation["execution_accuracy"] + 0.05
        ),
        "rl_hard_gain_at_least_2pp": (
            hard_execution_accuracy(rl_validation) >= hard_execution_accuracy(sft_validation) + 0.02
        ),
        "rl_overall_drop_no_more_than_1pp": (
            rl_validation["execution_accuracy"] >= sft_validation["execution_accuracy"] - 0.01
        ),
    }
    report = {
        "selected": finalists,
        "deployment_default": deployment,
        "selection_split": "tuning validation",
        "validation": validation,
        "official_dev": official,
        **external,
        "target_checks_on_validation": targets,
        "rl_seed_stability_on_validation": repeat_summary,
        "note": (
            "Official dev/private/robustness are reported only after all choices were made "
            "on validation. Use SFT as deployment default when RL does not improve validation "
            "execution accuracy. Target checks are reported, never used to hide a negative result."
        ),
    }
    write_json(campaign_root / "final_report.json", report)
    lines = [
        "# Experiment campaign result",
        "",
        f"- Best SFT: `{finalists['sft']}`",
        f"- Best RL: `{finalists['rl']}`",
        f"- Deployment default: `{deployment}`",
    ]
    if repeat_summary is not None:
        lines.append(
            "- RL validation execution across seeds: "
            f"{repeat_summary['mean']:.2%} ± {repeat_summary['population_std']:.2%} "
            "(population std)"
        )
    lines.extend(
        [
            "",
            "| role | experiment | validation | official dev | private | robustness |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for role, experiment in finalists.items():
        lines.append(
            f"| {role} | {experiment} | "
            f"{validation.get(experiment, {}).get('execution_accuracy', 0):.2%} | "
            f"{official.get(experiment, {}).get('execution_accuracy', 0):.2%} | "
            f"{external['private'][role].get('execution_accuracy', 0):.2%} | "
            f"{external['robustness'][role].get('execution_accuracy', 0):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered target checks (validation)",
            "",
            *(f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in targets.items()),
        ]
    )
    (campaign_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command staged SQL-Reasoner experiment campaign"
    )
    parser.add_argument("--matrix", default="configs/experiments.yaml")
    parser.add_argument("--preset", choices=["core", "full"], default="full")
    parser.add_argument("--tuning-limit", type=int)
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--include-ceval", action="store_true")
    parser.add_argument("--skip-stability-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_matrix = load_yaml(args.matrix)
    output_root = Path(base_matrix["paths"]["output_root"]) / base_matrix["suite"]
    campaign_root = output_root / "campaign"
    campaign_root.mkdir(parents=True, exist_ok=True)
    log_path = campaign_root / "campaign.log"

    if not args.skip_stability_check:
        stability_path = Path("outputs/stability/summary.json")
        if not stability_path.is_file() or read_json(stability_path).get("status") != "pass":
            raise SystemExit("长稳定性测试尚未通过；先运行 make stability")

    required_data = [
        base_matrix["evaluation"]["tuning"],
        base_matrix["evaluation"]["final"],
        "data/benchmarks/retail_cn/test.jsonl",
        "data/benchmarks/retail_cn/robustness.jsonl",
    ]
    missing = [path for path in required_data if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"缺少实验数据: {missing}")

    run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        env_updates={},
        log_path=log_path,
        dry_run=args.dry_run,
    )
    run_command(
        [sys.executable, "scripts/capture_environment.py"],
        env_updates={},
        log_path=log_path,
        dry_run=args.dry_run,
    )

    matrix_sft = copy.deepcopy(base_matrix)
    matrix_sft_path = campaign_root / "matrix_sft.yaml"
    save_matrix(matrix_sft, matrix_sft_path)
    run_experiment_phase(
        matrix_sft_path,
        ["base", *SFT_CANDIDATES],
        log_path=log_path,
        limit=args.tuning_limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    best_sft = choose(
        matrix_sft,
        SFT_CANDIDATES,
        phase="sft",
        campaign_root=campaign_root,
        dry_run=args.dry_run,
        dry_run_default="sft_default",
    )

    matrix_rl = copy.deepcopy(matrix_sft)
    configure_rl_parent(matrix_rl, best_sft)
    matrix_rl_path = campaign_root / "matrix_rl_algorithm.yaml"
    save_matrix(matrix_rl, matrix_rl_path)
    run_experiment_phase(
        matrix_rl_path,
        RL_ALGORITHM_CANDIDATES,
        log_path=log_path,
        limit=args.tuning_limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    best_algorithm = choose(
        matrix_rl,
        RL_ALGORITHM_CANDIDATES,
        phase="rl_algorithm",
        campaign_root=campaign_root,
        dry_run=args.dry_run,
        dry_run_default="rl_dapo_components",
    )

    best_policy = best_algorithm
    if args.preset == "full":
        inherit_overrides(
            matrix_rl,
            best_algorithm,
            RL_TUNING_VARIANTS,
            compare_to=best_algorithm,
        )
        matrix_tuning_path = campaign_root / "matrix_rl_tuning.yaml"
        save_matrix(matrix_rl, matrix_tuning_path)
        tuning_candidates = [best_algorithm, *RL_TUNING_VARIANTS]
        run_experiment_phase(
            matrix_tuning_path,
            RL_TUNING_VARIANTS,
            log_path=log_path,
            limit=args.tuning_limit,
            force=args.force,
            dry_run=args.dry_run,
        )
        best_policy = choose(
            matrix_rl,
            tuning_candidates,
            phase="rl_tuning",
            campaign_root=campaign_root,
            dry_run=args.dry_run,
            dry_run_default=best_algorithm,
        )

        inherit_overrides(
            matrix_rl,
            best_policy,
            RL_REWARD_VARIANTS,
            compare_to=best_policy,
        )
        matrix_reward_path = campaign_root / "matrix_rl_reward.yaml"
        save_matrix(matrix_rl, matrix_reward_path)
        reward_candidates = [best_policy, *RL_REWARD_VARIANTS]
        run_experiment_phase(
            matrix_reward_path,
            RL_REWARD_VARIANTS,
            log_path=log_path,
            limit=args.tuning_limit,
            force=args.force,
            dry_run=args.dry_run,
        )
        best_policy = choose(
            matrix_rl,
            reward_candidates,
            phase="rl_reward",
            campaign_root=campaign_root,
            dry_run=args.dry_run,
            dry_run_default=best_policy,
        )

        inherit_overrides(
            matrix_rl,
            best_policy,
            RL_CONFIRMATION_VARIANTS,
            compare_to=best_policy,
        )
        matrix_confirmation_path = campaign_root / "matrix_rl_confirmation.yaml"
        save_matrix(matrix_rl, matrix_confirmation_path)
        run_experiment_phase(
            matrix_confirmation_path,
            RL_CONFIRMATION_VARIANTS,
            log_path=log_path,
            limit=args.tuning_limit,
            force=args.force,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            repeat_names = [best_policy, *RL_CONFIRMATION_VARIANTS]
            write_json(
                campaign_root / "rl_seed_stability.json",
                summarize_repeats(
                    repeat_names,
                    collect_summaries(matrix_rl, repeat_names),
                ),
            )

    final_matrix_path = campaign_root / "matrix_final.yaml"
    save_matrix(matrix_rl, final_matrix_path)
    finalists = {"base": "base", "sft": best_sft, "rl": best_policy}
    if not args.dry_run:
        validation = collect_summaries(matrix_rl, [best_sft, best_policy])
        sft_accuracy = validation[best_sft]["execution_accuracy"]
        rl_accuracy = validation[best_policy]["execution_accuracy"]
        deployment = best_policy if rl_accuracy > sft_accuracy else best_sft
    else:
        deployment = best_policy
    write_json(
        campaign_root / "state.json",
        {
            "preset": args.preset,
            "dry_run": args.dry_run,
            "best_sft": best_sft,
            "best_rl_algorithm": best_algorithm,
            "best_rl": best_policy,
            "deployment_default": deployment,
        },
    )

    if args.skip_final:
        print(f"Campaign tuning complete. State: {campaign_root / 'state.json'}")
        return

    run_command(
        runner_command(
            final_matrix_path,
            "eval",
            list(finalists.values()),
            split="final",
            force=args.force,
            dry_run=args.dry_run,
        ),
        env_updates={"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")},
        log_path=log_path,
        dry_run=False,
    )
    final_root = campaign_root / "final"
    for dataset_name, dataset_path in {
        "private": "data/benchmarks/retail_cn/test.jsonl",
        "robustness": "data/benchmarks/retail_cn/robustness.jsonl",
    }.items():
        run_final_dataset(
            matrix_rl,
            dataset_name,
            dataset_path,
            finalists,
            final_root=final_root,
            log_path=log_path,
            force=args.force,
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        run_command(
            [
                sys.executable,
                "scripts/compare_results.py",
                "--base",
                str(prediction_path(matrix_rl, "base", "final")),
                "--sft",
                str(prediction_path(matrix_rl, best_sft, "final")),
                "--grpo",
                str(prediction_path(matrix_rl, best_policy, "final")),
                "--output",
                str(final_root / "official_dev" / "comparison.json"),
            ],
            env_updates={},
            log_path=log_path,
            dry_run=False,
        )
        run_command(
            [
                sys.executable,
                "scripts/sample_errors.py",
                "--predictions",
                str(prediction_path(matrix_rl, best_policy, "final")),
                "--output",
                str(final_root / "error_analysis.md"),
                "--limit",
                "30",
            ],
            env_updates={},
            log_path=log_path,
            dry_run=False,
        )
    if args.include_ceval:
        maybe_run_ceval(
            matrix_rl,
            finalists,
            final_root=final_root,
            log_path=log_path,
            force=args.force,
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        write_final_report(matrix_rl, finalists, deployment, campaign_root)
        print(f"Campaign complete: {campaign_root / 'final_report.md'}")


if __name__ == "__main__":
    main()
