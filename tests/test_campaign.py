from __future__ import annotations

import unittest

from sql_reasoner.campaign import hard_execution_accuracy, select_best, summarize_repeats


class CampaignSelectionTests(unittest.TestCase):
    def test_hard_accuracy_is_count_weighted(self) -> None:
        summary = {
            "by_difficulty": {
                "hard": {"count": 3, "execution_accuracy": 2 / 3},
                "extra-hard": {"count": 1, "execution_accuracy": 0.0},
            }
        }
        self.assertEqual(hard_execution_accuracy(summary), 0.5)

    def test_execution_accuracy_is_primary(self) -> None:
        summaries = {
            "a": {"execution_accuracy": 0.6, "valid_sql_rate": 1.0},
            "b": {"execution_accuracy": 0.61, "valid_sql_rate": 0.0},
        }
        self.assertEqual(select_best(["a", "b"], summaries), "b")

    def test_hard_accuracy_breaks_execution_tie(self) -> None:
        summaries = {
            "a": {
                "execution_accuracy": 0.6,
                "by_difficulty": {"hard": {"count": 2, "execution_accuracy": 0.5}},
            },
            "b": {
                "execution_accuracy": 0.6,
                "by_difficulty": {"hard": {"count": 2, "execution_accuracy": 1.0}},
            },
        }
        self.assertEqual(select_best(["a", "b"], summaries), "b")

    def test_repeat_summary_reports_variation_without_selecting_a_seed(self) -> None:
        result = summarize_repeats(
            ["seed42", "seed7", "seed123"],
            {
                "seed42": {"execution_accuracy": 0.60},
                "seed7": {"execution_accuracy": 0.63},
                "seed123": {"execution_accuracy": 0.57},
            },
        )
        self.assertAlmostEqual(result["mean"], 0.60)
        self.assertAlmostEqual(result["population_std"], (0.0006**0.5))
        self.assertEqual(result["experiments"], ["seed42", "seed7", "seed123"])


if __name__ == "__main__":
    unittest.main()
