from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sql_reasoner.io import read_jsonl


class PrepareCSpiderTest(unittest.TestCase):
    def test_database_level_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_root = root / "database"
            for db_id in ("train_a", "train_b", "dev_c"):
                db_dir = db_root / db_id
                db_dir.mkdir(parents=True)
                with sqlite3.connect(db_dir / f"{db_id}.sqlite") as conn:
                    conn.executescript(
                        "CREATE TABLE items(id INTEGER); INSERT INTO items VALUES (1);"
                    )
            train = [
                {"db_id": "train_a", "question": "数量？", "query": "SELECT COUNT(*) FROM items"},
                {"db_id": "train_b", "question": "编号？", "query": "SELECT id FROM items"},
            ]
            dev = [{"db_id": "dev_c", "question": "数量？", "query": "SELECT COUNT(*) FROM items"}]
            (root / "train.json").write_text(
                json.dumps(train, ensure_ascii=False), encoding="utf-8"
            )
            (root / "dev.json").write_text(json.dumps(dev, ensure_ascii=False), encoding="utf-8")
            output = root / "processed"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_cspider.py",
                    "--train-json",
                    str(root / "train.json"),
                    "--dev-json",
                    str(root / "dev.json"),
                    "--database-dir",
                    str(db_root),
                    "--output-dir",
                    str(output),
                    "--validation-ratio",
                    "0.5",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            train_rows = read_jsonl(output / "train.jsonl")
            val_rows = read_jsonl(output / "validation.jsonl")
            dev_rows = read_jsonl(output / "official_dev.jsonl")
            self.assertTrue(train_rows)
            self.assertTrue(val_rows)
            self.assertFalse(
                {row["db_id"] for row in train_rows} & {row["db_id"] for row in val_rows}
            )
            self.assertEqual({row["split"] for row in dev_rows}, {"official_dev"})
            self.assertEqual(read_jsonl(output / "rejected.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
