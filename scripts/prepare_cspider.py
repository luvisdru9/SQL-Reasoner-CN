#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from sql_reasoner.formats import format_model_output
from sql_reasoner.io import read_json, write_json, write_jsonl
from sql_reasoner.planning import estimate_difficulty, plan_from_sql
from sql_reasoner.prompts import messages_for_sft
from sql_reasoner.sandbox import execute_read_only
from sql_reasoner.schema import introspect_schema, serialize_schema


def database_path(root: Path, db_id: str) -> Path:
    candidates = [root / db_id / f"{db_id}.sqlite", root / f"{db_id}.sqlite"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到数据库 {db_id}: {candidates}")


def convert(rows, db_root: Path, split_by_db: dict[str, str], source_split: str):
    converted = []
    rejected = []
    schema_cache = {}
    for index, row in enumerate(rows):
        db_id = row["db_id"]
        sql = row.get("query") or row.get("sql")
        question = row.get("question") or row.get("question_toks")
        if isinstance(question, list):
            question = "".join(question)
        try:
            db_path = database_path(db_root, db_id)
            if db_id not in schema_cache:
                schema_cache[db_id] = serialize_schema(introspect_schema(db_path))
            execute_read_only(db_path, sql)
            plan = plan_from_sql(sql)
            difficulty = row.get("difficulty") or estimate_difficulty(sql)
            output = format_model_output(plan, sql)
            split = source_split if source_split == "official_dev" else split_by_db[db_id]
            converted.append(
                {
                    "id": f"{source_split}-{index:06d}",
                    "db_id": db_id,
                    "question": question,
                    "schema": schema_cache[db_id],
                    "difficulty": difficulty,
                    "gold_plan": plan,
                    "gold_sql": sql,
                    "db_path": str(db_path),
                    "split": split,
                    **messages_for_sft(question, schema_cache[db_id], output),
                }
            )
        except Exception as exc:
            rejected.append(
                {"index": index, "db_id": db_id, "reason": f"{type(exc).__name__}: {exc}"}
            )
    return converted, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CSpider for SFT/GRPO")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--dev-json", required=True)
    parser.add_argument("--database-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/cspider")
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = read_json(args.train_json)
    dev = read_json(args.dev_json)
    db_ids = sorted({row["db_id"] for row in train})
    rng = random.Random(args.seed)
    rng.shuffle(db_ids)
    val_count = max(1, round(len(db_ids) * args.validation_ratio))
    val_ids = set(db_ids[:val_count])
    split_by_db = {db_id: ("validation" if db_id in val_ids else "train") for db_id in db_ids}
    db_root = Path(args.database_dir)
    converted_train, rejected_train = convert(train, db_root, split_by_db, "train_source")
    converted_dev, rejected_dev = convert(dev, db_root, {}, "official_dev")
    output = Path(args.output_dir)
    by_split = {
        "train": [row for row in converted_train if row["split"] == "train"],
        "validation": [row for row in converted_train if row["split"] == "validation"],
        "official_dev": converted_dev,
    }
    for split, rows in by_split.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    rejected = rejected_train + rejected_dev
    write_jsonl(output / "rejected.jsonl", rejected)
    write_json(
        output / "stats.json",
        {
            "seed": args.seed,
            "validation_db_ids": sorted(val_ids),
            "counts": {key: len(value) for key, value in by_split.items()},
            "difficulty": dict(Counter(row["difficulty"] for row in converted_train)),
            "rejected": len(rejected),
        },
    )
    print({key: len(value) for key, value in by_split.items()}, "rejected", len(rejected))


if __name__ == "__main__":
    main()
