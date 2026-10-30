#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sql_reasoner.io import read_jsonl, write_json
from sql_reasoner.metrics import mcnemar_exact


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare aligned checkpoint predictions")
    parser.add_argument("--base", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--grpo", required=True)
    parser.add_argument("--output", default="outputs/comparison.json")
    args = parser.parse_args()
    named = {
        "base": {row["id"]: row for row in read_jsonl(args.base)},
        "sft": {row["id"]: row for row in read_jsonl(args.sft)},
        "grpo": {row["id"]: row for row in read_jsonl(args.grpo)},
    }
    ids = list(named["base"])
    if any(set(rows) != set(ids) for rows in named.values()):
        raise ValueError("三个预测文件的样本 id 必须完全一致")

    def correct(checkpoint: str) -> list[bool]:
        return [bool(named[checkpoint][sample_id]["scores"]["execution"]) for sample_id in ids]

    result = {
        "count": len(ids),
        "execution_accuracy": {
            name: sum(correct(name)) / len(ids) if ids else 0.0 for name in named
        },
        "mcnemar": {
            "base_vs_sft": mcnemar_exact(correct("base"), correct("sft")),
            "sft_vs_grpo": mcnemar_exact(correct("sft"), correct("grpo")),
        },
    }
    write_json(args.output, result)
    markdown = [
        "| checkpoint | Execution Accuracy |",
        "|---|---:|",
        *[f"| {name} | {value:.2%} |" for name, value in result["execution_accuracy"].items()],
        "",
        "```json",
        json.dumps(result["mcnemar"], ensure_ascii=False, indent=2),
        "```",
    ]
    Path(args.output).with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
