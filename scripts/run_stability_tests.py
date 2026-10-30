#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from sql_reasoner.io import read_json, write_json


GPU_QUERY = (
    "index,name,temperature.gpu,power.draw,power.limit,"
    "memory.used,memory.total,utilization.gpu"
)


def render(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{prefix} {shlex.join(command)}" if prefix else shlex.join(command)


def sample_gpus(stop: threading.Event, output: Path, interval: float) -> None:
    fields = [
        "timestamp",
        "index",
        "name",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={GPU_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    values = [value.strip() for value in line.split(",")]
                    if len(values) != 8:
                        continue
                    writer.writerow(
                        dict(
                            zip(
                                fields,
                                [f"{time.time():.3f}", *values],
                            )
                        )
                    )
                handle.flush()
            stop.wait(interval)


def run_stage(
    command: list[str],
    env_updates: dict[str, str],
    log_path: Path,
    monitor_path: Path,
    monitor_interval: float,
    dry_run: bool,
) -> int:
    rendered = render(command, env_updates)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return 0
    environment = os.environ.copy()
    environment.update(env_updates)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    monitor = threading.Thread(
        target=sample_gpus,
        args=(stop, monitor_path, monitor_interval),
        daemon=True,
    )
    monitor.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {rendered}\n")
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
            return process.wait()
    finally:
        stop.set()
        monitor.join(timeout=max(5.0, monitor_interval * 2))


def telemetry_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"sample_count": 0, "gpus": {}}
    grouped: dict[str, list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                numeric = {
                    key: float(row[key])
                    for key in [
                        "temperature_c",
                        "power_draw_w",
                        "power_limit_w",
                        "memory_used_mib",
                        "memory_total_mib",
                        "utilization_percent",
                    ]
                }
            except (KeyError, TypeError, ValueError):
                continue
            grouped.setdefault(row["index"], []).append(numeric)
    gpu_summaries = {}
    for index, rows in grouped.items():
        gpu_summaries[index] = {
            "samples": len(rows),
            "max_temperature_c": max(row["temperature_c"] for row in rows),
            "max_power_draw_w": max(row["power_draw_w"] for row in rows),
            "power_limit_w": max(row["power_limit_w"] for row in rows),
            "max_memory_used_mib": max(row["memory_used_mib"] for row in rows),
            "memory_total_mib": max(row["memory_total_mib"] for row in rows),
            "max_memory_fraction": max(
                row["memory_used_mib"] / row["memory_total_mib"] for row in rows
            ),
            "mean_utilization_percent": sum(row["utilization_percent"] for row in rows)
            / len(rows),
        }
    return {
        "sample_count": sum(len(rows) for rows in grouped.values()),
        "gpus": gpu_summaries,
    }


def validate_stage(
    stage: str,
    checkpoint: Path,
    return_code: int,
    telemetry: dict[str, Any],
    steps: int,
    max_temperature: float,
    max_vram_fraction: float,
) -> dict[str, Any]:
    reasons = []
    if return_code != 0:
        reasons.append(f"training exit code {return_code}")
    metrics_path = checkpoint / "train_results.json"
    state_path = checkpoint / "trainer_state.json"
    vram_paths = sorted(checkpoint.glob("vram*.json"))
    metrics = read_json(metrics_path) if metrics_path.is_file() else {}
    state = read_json(state_path) if state_path.is_file() else {}
    vram = [read_json(path) for path in vram_paths]
    if not metrics:
        reasons.append("train_results.json missing")
    if int(state.get("global_step", -1)) != steps:
        reasons.append(f"global_step={state.get('global_step')} expected={steps}")
    loss = metrics.get("train_loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        reasons.append(f"invalid train_loss={loss!r}")
    if not vram:
        reasons.append("VRAM metrics missing")
    else:
        peak_fraction = max(row["peak_reserved_bytes"] / row["total_bytes"] for row in vram)
        if peak_fraction > max_vram_fraction:
            reasons.append(
                f"trainer peak reserved VRAM {peak_fraction:.1%} > {max_vram_fraction:.1%}"
            )
    for index, gpu in telemetry.get("gpus", {}).items():
        if gpu["max_temperature_c"] > max_temperature:
            reasons.append(
                f"GPU {index} temperature {gpu['max_temperature_c']:.0f}C > {max_temperature:.0f}C"
            )
        if gpu["max_memory_fraction"] > max_vram_fraction:
            reasons.append(
                f"GPU {index} observed memory {gpu['max_memory_fraction']:.1%} "
                f"> {max_vram_fraction:.1%}"
            )
        if gpu["max_power_draw_w"] > gpu["power_limit_w"] + 5:
            reasons.append(
                f"GPU {index} power {gpu['max_power_draw_w']:.1f}W exceeds reported limit"
            )
    if telemetry.get("sample_count", 0) == 0:
        reasons.append("no nvidia-smi telemetry samples")
    return {
        "stage": stage,
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "checkpoint": str(checkpoint),
        "global_step": state.get("global_step"),
        "train_loss": loss,
        "train_runtime": metrics.get("train_runtime"),
        "train_tokens_per_second": metrics.get("train_tokens_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        "trainer_vram": vram,
        "telemetry": telemetry,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Long stability test",
        "",
        "| stage | status | steps | runtime | train loss | max temp | max memory |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["stages"]:
        gpus = list(row["telemetry"].get("gpus", {}).values())
        max_temp = max((gpu["max_temperature_c"] for gpu in gpus), default=0.0)
        max_memory = max((gpu["max_memory_fraction"] for gpu in gpus), default=0.0)
        lines.append(
            f"| {row['stage']} | {row['status']} | {row.get('global_step')} | "
            f"{row.get('train_runtime') or 0:.1f}s | {row.get('train_loss') or 0:.4f} | "
            f"{max_temp:.0f}C | {max_memory:.1%} |"
        )
        for reason in row["reasons"]:
            lines.append(f"\n- {row['stage']}: {reason}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run long SFT and GRPO stability tests")
    parser.add_argument("--stage", choices=["sft", "rl", "all"], default="all")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--monitor-interval", type=float, default=2.0)
    parser.add_argument("--max-temperature", type=float, default=83.0)
    parser.add_argument("--max-vram-fraction", type=float, default=0.90)
    parser.add_argument("--output-root", default="outputs/stability")
    parser.add_argument("--sft-output", default="checkpoints/sft-stability-b1")
    parser.add_argument("--rl-output", default="checkpoints/grpo-stability-b2")
    parser.add_argument("--sft-adapter", default="checkpoints/sft-smoke")
    parser.add_argument("--rl-data", default="data/processed/cspider/grpo_smoke.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.monitor_interval <= 0:
        parser.error("--steps and --monitor-interval must be positive")

    output_root = Path(args.output_root)
    common_env = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        ),
    }
    definitions = []
    if args.stage in {"sft", "all"}:
        definitions.append(
            (
                "sft",
                Path(args.sft_output),
                [
                    "accelerate",
                    "launch",
                    "--multi_gpu",
                    "--num_processes",
                    "2",
                    "scripts/train_sft.py",
                    "--config",
                    "configs/sft.yaml",
                    "--smoke",
                    "--smoke-steps",
                    str(args.steps),
                    "--resume",
                    "none",
                    "--output-dir",
                    args.sft_output,
                    "--per-device-batch-size",
                    "1",
                    "--gradient-accumulation-steps",
                    "8",
                ],
                common_env,
            )
        )
    if args.stage in {"rl", "all"}:
        definitions.append(
            (
                "rl",
                Path(args.rl_output),
                ["bash", "scripts/run_grpo.sh"],
                {
                    **common_env,
                    "CONFIG": "configs/grpo.yaml",
                    "SMOKE": "1",
                    "MAX_STEPS": str(args.steps),
                    "SFT_ADAPTER": args.sft_adapter,
                    "DATA_PATH": args.rl_data,
                    "OUTPUT_DIR": args.rl_output,
                    "RESUME": "none",
                    "TRAIN_GPU": "0",
                    "ROLLOUT_GPU": "1",
                    "VLLM_GPU_MEMORY_UTILIZATION": "0.72",
                },
            )
        )

    reports = []
    for stage, checkpoint, command, environment in definitions:
        result_path = output_root / stage / "stability_result.json"
        if result_path.is_file() and not args.force:
            existing = read_json(result_path)
            if existing.get("global_step") == args.steps and existing.get("status") == "pass":
                print(f"[skip] {stage}: 已完成 {args.steps} steps")
                reports.append(existing)
                continue
        print(f"\n===== {stage.upper()} stability: {args.steps} steps =====", flush=True)
        return_code = run_stage(
            command,
            environment,
            output_root / stage / "train.log",
            output_root / stage / "gpu_telemetry.csv",
            args.monitor_interval,
            args.dry_run,
        )
        if args.dry_run:
            continue
        telemetry = telemetry_summary(output_root / stage / "gpu_telemetry.csv")
        result = validate_stage(
            stage,
            checkpoint,
            return_code,
            telemetry,
            args.steps,
            args.max_temperature,
            args.max_vram_fraction,
        )
        write_json(result_path, result)
        reports.append(result)
        print(f"[{result['status']}] {stage}: {result['reasons'] or 'all checks passed'}")

    if args.dry_run:
        return
    report = {
        "steps": args.steps,
        "max_temperature": args.max_temperature,
        "max_vram_fraction": args.max_vram_fraction,
        "status": "pass" if reports and all(row["status"] == "pass" for row in reports) else "fail",
        "stages": reports,
    }
    write_json(output_root / "summary.json", report)
    write_markdown(output_root / "summary.md", report)
    print(f"\nStability summary: {output_root / 'summary.md'} ({report['status']})")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
