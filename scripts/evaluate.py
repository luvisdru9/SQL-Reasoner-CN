#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from sql_reasoner.experiments import stable_hash
from sql_reasoner.io import read_json, read_jsonl, write_json, write_jsonl
from sql_reasoner.metrics import evaluate_predictions
from sql_reasoner.modeling import ModelRunner


def _file_signature(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _generation_fingerprint(args: argparse.Namespace) -> str:
    adapter_file = (
        None
        if not args.adapter
        else _file_signature(Path(args.adapter) / "adapter_model.safetensors")
    )
    return stable_hash(
        {
            "dataset": _file_signature(Path(args.dataset)),
            "model": args.model,
            "adapter": adapter_file,
            "limit": args.limit,
            "max_new_tokens": 384,
            "temperature": 0.0,
        }
    )


def _read_generation_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    cached: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # An interrupted append can leave only the final line incomplete.
            if index == len(lines) - 1:
                break
            raise
        cached[str(row["id"])] = row
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or score Text-to-SQL predictions")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--predictions", help="Score an existing JSONL containing id,prediction")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--official-evaluator", help="Optional path to CSpider evaluation.py")
    parser.add_argument("--official-python", default="python2")
    parser.add_argument("--official-gold")
    parser.add_argument("--official-db-dir")
    parser.add_argument("--official-tables")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.dataset)
    runner = None
    if args.limit:
        rows = rows[: args.limit]
    if args.predictions:
        predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
        missing = [row["id"] for row in rows if row["id"] not in predictions]
        if missing:
            raise ValueError(f"预测缺少 {len(missing)} 条，例如 {missing[:3]}")
        records = [{**row, **predictions[row["id"]]} for row in rows]
    else:
        cache_path = output / "generation_cache.jsonl"
        cache_meta_path = output / "generation_cache.meta.json"
        generation_fingerprint = _generation_fingerprint(args)
        cache_matches = False
        if cache_meta_path.is_file():
            try:
                cache_matches = (
                    read_json(cache_meta_path).get("fingerprint") == generation_fingerprint
                )
            except (OSError, ValueError, AttributeError):
                cache_matches = False
        if cache_matches:
            cached = _read_generation_cache(cache_path)
            cache_mode = "a"
            print(f"恢复 {len(cached)}/{len(rows)} 条已生成预测", flush=True)
        else:
            cached = {}
            cache_mode = "w"
            write_json(
                cache_meta_path,
                {"fingerprint": generation_fingerprint, "complete": False},
            )

        missing_ids = {str(row["id"]) for row in rows} - set(cached)
        runner = ModelRunner(args.model, adapter_path=args.adapter) if missing_ids else None
        records = []
        with cache_path.open(cache_mode, encoding="utf-8") as cache_handle:
            for index, row in enumerate(rows, start=1):
                sample_id = str(row["id"])
                if sample_id in cached:
                    cached_row = cached[sample_id]
                    record = {
                        **row,
                        "prediction": cached_row["prediction"],
                        "latency_ms": cached_row["latency_ms"],
                        "generated_tokens": cached_row["generated_tokens"],
                        "tokens_per_second": cached_row["tokens_per_second"],
                    }
                    print(f"[{index}/{len(rows)}] {sample_id} cached", flush=True)
                else:
                    assert runner is not None
                    generation = runner.generate(row["question"], row["schema"])
                    record = {
                        **row,
                        "prediction": generation.text,
                        "latency_ms": generation.elapsed_ms,
                        "generated_tokens": generation.generated_tokens,
                        "tokens_per_second": generation.generated_tokens
                        / max(generation.elapsed_ms / 1000, 1e-9),
                    }
                    cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                    print(
                        f"[{index}/{len(rows)}] {sample_id} {generation.elapsed_ms:.0f}ms",
                        flush=True,
                    )
                records.append(record)

        write_json(
            cache_meta_path,
            {"fingerprint": generation_fingerprint, "complete": True},
        )

    summary, detailed = evaluate_predictions(records)
    latencies = [row["latency_ms"] for row in detailed if "latency_ms" in row]
    throughputs = [row["tokens_per_second"] for row in detailed if "tokens_per_second" in row]
    if latencies:
        summary["latency_p50_ms"] = statistics.median(latencies)
        summary["tokens_per_second_mean"] = statistics.fmean(throughputs)
        if runner is not None and runner.torch.cuda.is_available():
            summary["inference_peak_vram_bytes"] = max(
                runner.torch.cuda.max_memory_allocated(index)
                for index in range(runner.torch.cuda.device_count())
            )
    summary["model_id"] = args.model
    summary["adapter_path"] = args.adapter
    if args.adapter and Path(args.adapter).exists():
        summary["adapter_size_bytes"] = sum(
            path.stat().st_size for path in Path(args.adapter).rglob("*") if path.is_file()
        )
    write_json(output / "summary.json", summary)
    write_jsonl(output / "predictions.jsonl", detailed)

    if args.official_evaluator:
        required = [args.official_gold, args.official_db_dir, args.official_tables]
        if not all(required):
            raise ValueError(
                "使用官方 evaluator 时还需 --official-gold/--official-db-dir/--official-tables"
            )
        pred_path = output / "official_predictions.sql"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as handle:
            for row in detailed:
                try:
                    from sql_reasoner.formats import parse_model_output

                    sql = parse_model_output(row["prediction"]).sql
                except ValueError:
                    sql = "SELECT 0"
                handle.write(f"{sql}\n")
        command = [
            args.official_python,
            args.official_evaluator,
            "--gold",
            args.official_gold,
            "--pred",
            str(pred_path),
            "--etype",
            "match",
            "--db",
            args.official_db_dir,
            "--table",
            args.official_tables,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        (output / "official_eval.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"官方 evaluator 失败，见 {output / 'official_eval.txt'}")
    print(summary)


if __name__ == "__main__":
    main()
