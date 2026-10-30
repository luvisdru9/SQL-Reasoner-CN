#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

from sql_reasoner.io import load_yaml, read_json, write_json


SFT_CANDIDATES = [(1, 8), (2, 4), (4, 2)]
RL_CANDIDATES = [(1, 4), (2, 2), (4, 1)]


def render(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{prefix} {shlex.join(command)}" if prefix else shlex.join(command)


def run(command: list[str], env_updates: dict[str, str], log_path: Path, dry_run: bool) -> int:
    rendered = render(command, env_updates)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment.update(env_updates)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
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
        return process.wait()


def read_result(stage: str, name: str, checkpoint: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": stage, "candidate": name, "checkpoint": str(checkpoint)}
    metrics_path = checkpoint / "train_results.json"
    if not metrics_path.is_file():
        result.update(status="failed", reason="train_results.json missing")
        return result
    metrics = read_json(metrics_path)
    vram_paths = sorted(checkpoint.glob("vram*.json"))
    if not vram_paths:
        result.update(status="failed", reason="VRAM metrics missing")
        return result
    vram = [read_json(path) for path in vram_paths]
    peak_reserved = max(row["peak_reserved_bytes"] for row in vram)
    total = min(row["total_bytes"] for row in vram)
    result.update(
        status="ok",
        train_runtime=metrics.get("train_runtime"),
        train_tokens_per_second=metrics.get("train_tokens_per_second"),
        train_steps_per_second=metrics.get("train_steps_per_second"),
        peak_reserved_bytes=peak_reserved,
        total_bytes=total,
        peak_reserved_fraction=peak_reserved / total,
    )
    return result


def completed_checkpoint(checkpoint: Path) -> Path:
    """Accept checkpoints produced before OUTPUT_DIR was passed explicitly to smoke runs."""
    if (checkpoint / "train_results.json").is_file():
        return checkpoint
    legacy_smoke = Path(f"{checkpoint}-smoke")
    if (legacy_smoke / "train_results.json").is_file():
        return legacy_smoke
    return checkpoint


def recommend(rows: list[dict[str, Any]], max_vram_fraction: float) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row.get("status") == "ok" and row["peak_reserved_fraction"] <= max_vram_fraction
    ]
    if not eligible:
        return None

    def score(row: dict[str, Any]) -> float:
        if row["stage"] == "sft":
            return float(row.get("train_tokens_per_second") or 0.0)
        return float(row.get("train_steps_per_second") or 0.0)

    fastest = max(score(row) for row in eligible)
    # Two- or ten-step smoke measurements are noisy. Treat candidates within
    # 5% of the fastest as equivalent and prefer the one with more VRAM
    # headroom for unusually long prompts/completions.
    near_fastest = [row for row in eligible if score(row) >= fastest * 0.95]
    return min(near_fastest, key=lambda row: row["peak_reserved_fraction"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate memory-sensitive microbatch settings before quality experiments"
    )
    parser.add_argument("--stage", choices=["sft", "rl", "all"], default="all")
    parser.add_argument("--sft-config", default="configs/sft.yaml")
    parser.add_argument("--rl-config", default="configs/grpo.yaml")
    parser.add_argument("--sft-adapter", default="checkpoints/sft-smoke")
    parser.add_argument("--rl-data", default="data/processed/cspider/grpo_smoke.jsonl")
    parser.add_argument("--sft-steps", type=int, default=10)
    parser.add_argument("--rl-steps", type=int, default=2)
    parser.add_argument("--max-vram-fraction", type=float, default=0.90)
    parser.add_argument("--root", default="outputs/resource_calibration")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.sft_steps <= 0 or args.rl_steps <= 0:
        parser.error("step 数必须为正数")
    if not 0 < args.max_vram_fraction < 1:
        parser.error("--max-vram-fraction 必须在 0 和 1 之间")

    root = Path(args.root)
    checkpoint_root = root / "checkpoints"
    config_root = root / "configs"
    common_env = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    results: list[dict[str, Any]] = []

    if args.stage in {"sft", "all"}:
        for batch, accumulation in SFT_CANDIDATES:
            name = f"sft_b{batch}_a{accumulation}"
            checkpoint = checkpoint_root / name
            if (checkpoint / "train_results.json").is_file() and not args.force:
                print(f"[skip] {name}: 已存在结果")
            else:
                command = [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_processes",
                    "2",
                    "scripts/train_sft.py",
                    "--config",
                    args.sft_config,
                    "--smoke",
                    "--smoke-steps",
                    str(args.sft_steps),
                    "--resume",
                    "none",
                    "--output-dir",
                    str(checkpoint),
                    "--per-device-batch-size",
                    str(batch),
                    "--gradient-accumulation-steps",
                    str(accumulation),
                ]
                return_code = run(command, common_env, root / "logs" / f"{name}.log", args.dry_run)
                if return_code:
                    print(f"[failed] {name}: exit={return_code}", flush=True)
            if not args.dry_run:
                results.append(read_result("sft", name, checkpoint))

    if args.stage in {"rl", "all"}:
        base_config = load_yaml(args.rl_config)
        for batch, accumulation in RL_CANDIDATES:
            name = f"rl_b{batch}_a{accumulation}"
            checkpoint = checkpoint_root / name
            result_checkpoint = completed_checkpoint(checkpoint)
            config = json.loads(json.dumps(base_config))
            config["output_dir"] = str(checkpoint)
            config["sft_adapter"] = args.sft_adapter
            config["data"]["train"] = args.rl_data
            config["training"]["per_device_batch_size"] = batch
            config["training"]["gradient_accumulation_steps"] = accumulation
            config["generation"]["generation_batch_size"] = 4
            config_root.mkdir(parents=True, exist_ok=True)
            config_path = config_root / f"{name}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            if (result_checkpoint / "train_results.json").is_file() and not args.force:
                print(f"[skip] {name}: 已存在结果 {result_checkpoint}")
            else:
                env = {
                    **common_env,
                    "CONFIG": str(config_path),
                    "SMOKE": "1",
                    "MAX_STEPS": str(args.rl_steps),
                    "OUTPUT_DIR": str(checkpoint),
                    "RESUME": "none",
                    "TRAIN_GPU": "0",
                    "ROLLOUT_GPU": "1",
                    "VLLM_GPU_MEMORY_UTILIZATION": "0.72",
                    "VLLM_ENFORCE_EAGER": "1",
                }
                return_code = run(
                    ["bash", "scripts/run_grpo.sh"],
                    env,
                    root / "logs" / f"{name}.log",
                    args.dry_run,
                )
                if return_code:
                    print(f"[failed] {name}: exit={return_code}", flush=True)
            if not args.dry_run:
                results.append(read_result("rl", name, completed_checkpoint(checkpoint)))

    if args.dry_run:
        return
    recommendations = {
        stage: recommend(
            [row for row in results if row["stage"] == stage], args.max_vram_fraction
        )
        for stage in ["sft", "rl"]
        if args.stage in {stage, "all"}
    }
    report = {
        "selection_rule": (
            "among successful candidates whose peak reserved VRAM is <= "
            f"{args.max_vram_fraction:.0%}, prefer lower VRAM when throughput is within 5% of "
            "the fastest; effective batch is held constant"
        ),
        "results": results,
        "recommendations": recommendations,
        "note": "Apply the chosen pair to configs/experiments.yaml only before quality experiments.",
    }
    write_json(root / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
