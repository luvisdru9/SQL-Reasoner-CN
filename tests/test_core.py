from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from sql_reasoner.formats import format_model_output, parse_model_output
from sql_reasoner.metrics import (
    bootstrap_ci,
    evaluate_predictions,
    mcnemar_exact,
    structural_exact_match,
)
from sql_reasoner.planning import parse_sql, plan_from_sql
from sql_reasoner.rewards import component_scores
from sql_reasoner.sandbox import (
    UnsafeSQLError,
    execute_read_only,
    execution_matches,
)


class CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "test.sqlite"
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
                CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL,
                  FOREIGN KEY(user_id) REFERENCES users(id));
                INSERT INTO users VALUES (1, '甲', '上海'), (2, '乙', '北京');
                INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 20.0), (3, 2, 15.0);
                """
            )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_format_round_trip(self) -> None:
        plan = {"tables": ["users"], "joins": [], "filters": [], "aggregation": ["COUNT(*)"]}
        text = format_model_output(plan, "SELECT COUNT(*) FROM users")
        parsed = parse_model_output(text)
        self.assertEqual(parsed.plan, plan)
        self.assertEqual(parsed.sql, "SELECT COUNT(*) FROM users")
        with self.assertRaises(ValueError):
            parse_model_output("SELECT * FROM users")

    def test_read_only_sandbox(self) -> None:
        result = execute_read_only(self.db_path, "SELECT name FROM users ORDER BY id")
        self.assertEqual(result.rows, (("甲",), ("乙",)))
        with self.assertRaises(UnsafeSQLError):
            execute_read_only(self.db_path, "DELETE FROM users")
        with self.assertRaises((UnsafeSQLError, ValueError)):
            execute_read_only(self.db_path, "SELECT * FROM users; DROP TABLE users")
        with self.assertRaises(UnsafeSQLError):
            execute_read_only(self.db_path, "PRAGMA user_version")
        self.assertEqual(
            execute_read_only(self.db_path, "SELECT COUNT(*) FROM users").rows[0][0], 2
        )

    def test_execution_semantics(self) -> None:
        self.assertTrue(
            execution_matches(
                self.db_path, "SELECT name FROM users", "SELECT name FROM users ORDER BY id"
            )
        )
        self.assertFalse(
            execution_matches(
                self.db_path,
                "SELECT name FROM users ORDER BY id DESC",
                "SELECT name FROM users ORDER BY id",
            )
        )
        self.assertTrue(execution_matches(self.db_path, "SELECT 10", "SELECT 10.0"))

    def test_planning_and_rewards(self) -> None:
        sql = "SELECT u.city, SUM(o.amount) FROM users u JOIN orders o ON u.id=o.user_id GROUP BY u.city"
        plan = plan_from_sql(sql)
        text = format_model_output(plan, sql)
        scores = component_scores(text, str(self.db_path), sql)
        self.assertEqual(
            scores,
            {
                "format": 1.0,
                "safe_sql": 1.0,
                "schema": 1.0,
                "plan_consistency": 1.0,
                "execution": 1.0,
            },
        )

    def test_metrics(self) -> None:
        self.assertTrue(structural_exact_match("select id from users", "SELECT id FROM users"))
        self.assertEqual(mcnemar_exact([True, True], [True, True])["p_value"], 1.0)
        low, high = bootstrap_ci([0.0, 1.0, 1.0], samples=100, seed=1)
        self.assertLessEqual(low, high)

    def test_malformed_model_sql_is_scored_as_invalid(self) -> None:
        malformed_sql = "SELECT city FROM users GROUP BY city HAVING count(*) >= ORDER BY city"
        text = format_model_output(
            {"tables": ["users"], "joins": [], "filters": [], "aggregation": []},
            malformed_sql,
        )
        with self.assertRaises(ValueError):
            parse_sql(malformed_sql)

        scores = component_scores(text, str(self.db_path), "SELECT city FROM users")
        self.assertEqual(scores["format"], 1.0)
        self.assertEqual(scores["safe_sql"], 0.0)
        self.assertEqual(scores["schema"], 0.0)
        self.assertEqual(scores["plan_consistency"], 0.0)
        self.assertEqual(scores["execution"], 0.0)

        summary, detailed = evaluate_predictions(
            [
                {
                    "id": "malformed",
                    "prediction": text,
                    "db_path": str(self.db_path),
                    "gold_sql": "SELECT city FROM users",
                    "difficulty": "easy",
                }
            ]
        )
        self.assertEqual(summary["valid_sql_rate"], 0.0)
        self.assertEqual(detailed[0]["error_type"], "syntax_or_unsafe")


if __name__ == "__main__":
    unittest.main()
