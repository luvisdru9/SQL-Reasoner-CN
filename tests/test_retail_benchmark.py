from __future__ import annotations

import unittest
from pathlib import Path

from sql_reasoner.formats import parse_model_output
from sql_reasoner.io import read_jsonl
from sql_reasoner.sandbox import execution_matches


class RetailBenchmarkTest(unittest.TestCase):
    def test_committed_benchmark_is_complete_and_executable(self) -> None:
        root = Path("data/benchmarks/retail_cn")
        test_rows = read_jsonl(root / "test.jsonl")
        robust_rows = read_jsonl(root / "robustness.jsonl")
        self.assertEqual(len(test_rows), 50)
        self.assertEqual(len(robust_rows), 30)
        self.assertEqual(len({row["id"] for row in test_rows + robust_rows}), 80)
        for row in test_rows + robust_rows:
            completion = row["completion"][0]["content"]
            sql = parse_model_output(completion).sql
            self.assertTrue(execution_matches(row["db_path"], sql, row["gold_sql"]), row["id"])


if __name__ == "__main__":
    unittest.main()
