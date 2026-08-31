from __future__ import annotations

import unittest

import numpy as np

from scripts.prequential_causality import (
    completion_safe_training_mask,
    outcome_available_at,
    synthetic_prequential_predictions,
)


class CompletionSafeCausalityTest(unittest.TestCase):
    def test_future_label_mutation_cannot_change_earlier_predictions(self) -> None:
        hour = 3_600_000
        times = np.arange(12, dtype=np.int64) * hour
        play = np.array(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            dtype=np.int64,
        )
        labels = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.int8)
        cutoff = 8 * hour

        before = synthetic_prequential_predictions(times, play, labels, block_hours=2)
        mutated = labels.copy()
        mutated[outcome_available_at(times, play) >= cutoff] ^= 1
        after = synthetic_prequential_predictions(times, play, mutated, block_hours=2)

        np.testing.assert_array_equal(before[times < cutoff], after[times < cutoff])

    def test_same_boundary_feedback_is_invisible(self) -> None:
        hour = 3_600_000
        times = np.array([hour, 2 * hour, 3 * hour], dtype=np.int64)
        play = np.array([hour, 0, 1], dtype=np.int64)
        mask = completion_safe_training_mask(times, play, 2 * hour)
        np.testing.assert_array_equal(mask, np.array([False, False, False]))

    def test_negative_play_time_is_clamped_to_zero(self) -> None:
        times = np.array([100, 200], dtype=np.int64)
        play = np.array([-50, 25], dtype=np.int64)
        np.testing.assert_array_equal(outcome_available_at(times, play), [100, 225])


if __name__ == "__main__":
    unittest.main()
