from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LockedConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs" / "prequential_teacher.lock.json").read_text()
        )

    def test_exactly_37_ordered_stages_are_locked(self) -> None:
        stages = self.config["stages"]
        self.assertEqual(len(stages), 37)
        self.assertEqual([stage["id"] for stage in stages], list(range(1, 38)))
        for stage in stages:
            for key in (
                "source",
                "transform",
                "gate",
                "weight",
                "expected_metrics",
                "source_artifact",
                "output_artifact",
                "report_artifact",
            ):
                self.assertIn(key, stage)

    def test_holdout_is_later_and_still_reserved(self) -> None:
        development = self.config["periods"]["development"]
        holdout = self.config["periods"]["final_holdout"]
        self.assertLess(development["end_date"], holdout["start_date"])
        self.assertEqual(holdout["status"], "reserved_not_evaluated")
        forbidden = set(self.config["holdout_protocol"]["forbidden_columns"])
        safe = set(self.config["holdout_protocol"]["safe_input_columns"])
        self.assertFalse(forbidden & safe)

    def test_environment_and_causality_are_explicit(self) -> None:
        self.assertRegex(self.config["environment"]["python"], r"^3\.13\.6$")
        self.assertEqual(
            self.config["causality"]["training_comparison"],
            "outcome_available_at < block_start",
        )


if __name__ == "__main__":
    unittest.main()
