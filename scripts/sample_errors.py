#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from sql_reasoner.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified manual error-analysis worksheet"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="outputs/error_analysis.md")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in read_jsonl(args.predictions):
        if row.get("error_type"):
            groups[row["error_type"]].append(row)
    selected = []
    while len(selected) < args.limit and any(groups.values()):
        for error_type in sorted(groups):
            if groups[error_type] and len(selected) < args.limit:
                selected.append(groups[error_type].pop(0))
    lines = [
        "# Manual Error Analysis",
        "",
        "逐条填写根因与修复方向；自动分类只是抽样辅助，不作为最终人工结论。",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        lines.extend(
            [
                f"## {index}. {row['id']} — {row['error_type']}",
                "",
                f"- 问题：{row['question']}",
                f"- Gold SQL：`{row['gold_sql']}`",
                "- 模型输出：",
                "",
                "```text",
                row.get("prediction", ""),
                "```",
                "",
                "- 人工根因：待填写",
                "- 是否属于 schema / Join / aggregation / nested / value：待填写",
                "- 可行修复：待填写",
                "",
            ]
        )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    print(f"sampled={len(selected)} output={target}")


if __name__ == "__main__":
    main()
