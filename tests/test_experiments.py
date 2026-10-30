from __future__ import annotations

import unittest

from sql_reasoner.experiments import (
    apply_overrides,
    expand_dependencies,
    get_dotted,
    select_experiments,
    validate_profile,
)


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = {
            "model_id": "model",
            "training": {"batch": 1, "learning_rate": 1e-5},
        }
        self.profile = {
            "locked": {"model_id": "model", "training.batch": 1},
            "tunable": ["training.learning_rate"],
        }

    def test_tunable_override_preserves_resource_lock(self) -> None:
        resolved = apply_overrides(self.base, {"training.learning_rate": 2e-5})
        validate_profile(resolved, self.profile, {"training.learning_rate": 2e-5})
        self.assertEqual(get_dotted(resolved, "training.batch"), 1)
        self.assertEqual(get_dotted(resolved, "training.learning_rate"), 2e-5)

    def test_locked_override_is_rejected(self) -> None:
        resolved = apply_overrides(self.base, {"training.batch": 2})
        with self.assertRaisesRegex(ValueError, "未列入 tunable"):
            validate_profile(resolved, self.profile, {"training.batch": 2})

    def test_profile_drift_is_rejected(self) -> None:
        drifted = apply_overrides(self.base, {})
        drifted["training"]["batch"] = 2
        with self.assertRaisesRegex(ValueError, "资源配置锁校验失败"):
            validate_profile(drifted, self.profile, {})

    def test_group_selection_and_dependency_order(self) -> None:
        experiments = {
            "base": {"groups": ["core"]},
            "sft": {"groups": ["core"]},
            "rl": {"groups": ["rl"], "depends_on": "sft"},
        }
        selected = select_experiments(experiments, [], ["rl"])
        self.assertEqual(selected, ["rl"])
        self.assertEqual(expand_dependencies(experiments, selected), ["sft", "rl"])


if __name__ == "__main__":
    unittest.main()
