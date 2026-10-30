#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path

from sql_reasoner.constants import REWARD_WEIGHTS
from sql_reasoner.io import read_jsonl, write_json, write_jsonl
from sql_reasoner.modeling import ModelRunner
from sql_reasoner.rewards import component_scores


def total_score(scores: dict[str, float]) -> float:
    return sum(REWARD_WEIGHTS[name] * value for name, value in scores.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time reward-variance filtering for GRPO")
    parser.add_argument("--input", default="data/processed/cspider/train.jsonl")
    parser.add_argument("--output", default="data/processed/cspider/grpo_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", default="checkpoints/sft")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=2000,
        help="Maximum prompts to sample before filling from the closest fallback examples",
    )
    parser.add_argument(
        "--strategy",
        choices=["variance", "random"],
        default="variance",
        help="Use reward-variance filtering or a random difficulty-matched control",
    )
    parser.add_argument("--stats-output", help="Defaults to <output>.stats.json")
    parser.add_argument(
        "--min-variance-positive",
        type=int,
        default=0,
        help="Fail after writing outputs if fewer selected prompts have non-constant rewards",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.candidates <= 0 or args.limit <= 0 or args.max_scan <= 0:
        parser.error("--candidates/--limit/--max-scan must be positive")
    if args.max_new_tokens <= 0 or args.log_every <= 0:
        parser.error("--max-new-tokens/--log-every must be positive")
    rows = [
        row
        for row in read_jsonl(args.input)
        if row["difficulty"] in {"medium", "hard", "extra-hard"}
    ]
    random.Random(args.seed).shuffle(rows)
    if args.strategy == "random":
        selected = rows[: args.limit]
        write_jsonl(args.output, selected)
        stats_path = args.stats_output or str(Path(args.output).with_suffix(".stats.json"))
        write_json(
            stats_path,
            {
                "strategy": args.strategy,
                "selected": len(selected),
                "seed": args.seed,
                "difficulty": {
                    difficulty: sum(row["difficulty"] == difficulty for row in selected)
                    for difficulty in ["medium", "hard", "extra-hard"]
                },
            },
        )
        print(f"selected={len(selected)} strategy=random", flush=True)
        return

    runner = ModelRunner(args.model, adapter_path=args.adapter)
    selected = []
    fallback = []
    scanned = 0
    all_candidate_scores = []
    print(
        f"开始奖励方差筛选: candidates={args.candidates} limit={args.limit} "
        f"max_scan={min(args.max_scan, len(rows))} max_new_tokens={args.max_new_tokens}",
        flush=True,
    )
    for row_index, row in enumerate(rows[: args.max_scan]):
        scanned += 1
        candidate_scores = []
        for candidate_index in range(args.candidates):
            generation = runner.generate(
                row["question"],
                row["schema"],
                temperature=0.8,
                top_p=0.95,
                seed=args.seed + row_index * args.candidates + candidate_index,
                max_new_tokens=args.max_new_tokens,
            )
            candidate_scores.append(
                total_score(component_scores(generation.text, row["db_path"], row["gold_sql"]))
            )
        all_candidate_scores.extend(candidate_scores)
        enriched = {**row, "selection_candidate_scores": candidate_scores}
        if min(candidate_scores) < max(candidate_scores):
            selected.append(enriched)
        else:
            fallback.append(enriched)
        if scanned == 1 or scanned % args.log_every == 0:
            print(
                f"[scan {scanned}/{min(args.max_scan, len(rows))}] "
                f"variance_positive={len(selected)} fallback={len(fallback)}",
                flush=True,
            )
        if len(selected) >= args.limit:
            break
    if len(selected) < args.limit:
        fallback.sort(
            key=lambda row: abs(sum(row["selection_candidate_scores"]) / args.candidates - 0.7)
        )
        selected.extend(fallback[: args.limit - len(selected)])
    write_jsonl(args.output, selected)
    variance_positive = sum(
        min(row["selection_candidate_scores"]) < max(row["selection_candidate_scores"])
        for row in selected
    )
    stats_path = args.stats_output or str(Path(args.output).with_suffix(".stats.json"))
    write_json(
        stats_path,
        {
            "strategy": args.strategy,
            "seed": args.seed,
            "candidates": args.candidates,
            "scanned": scanned,
            "selected": len(selected),
            "variance_positive": variance_positive,
            "variance_positive_rate": variance_positive / len(selected) if selected else 0.0,
            "candidate_reward_mean": statistics.fmean(all_candidate_scores)
            if all_candidate_scores
            else 0.0,
            "candidate_reward_min": min(all_candidate_scores) if all_candidate_scores else 0.0,
            "candidate_reward_max": max(all_candidate_scores) if all_candidate_scores else 0.0,
        },
    )
    print(
        f"selected={len(selected)} variance_positive={variance_positive} scanned={scanned}",
        flush=True,
    )
    if variance_positive < args.min_variance_positive:
        raise SystemExit(
            f"奖励方差样本不足: {variance_positive} < {args.min_variance_positive}; "
            f"详见 {stats_path}"
        )


if __name__ == "__main__":
    main()
