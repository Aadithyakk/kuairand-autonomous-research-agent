#!/usr/bin/env python3
"""Shared completion-safe primitives for prequential training.

The strict inequality is intentional: feedback completing exactly at a block
boundary is unavailable to the model that scores that boundary.
"""
from __future__ import annotations

import numpy as np


def outcome_available_at(
    event_time_ms: np.ndarray, play_time_ms: np.ndarray
) -> np.ndarray:
    """Return the first timestamp at which each outcome may be consumed."""
    event_time = np.asarray(event_time_ms, dtype=np.int64)
    play_time = np.asarray(play_time_ms, dtype=np.int64)
    if event_time.shape != play_time.shape:
        raise ValueError("event_time_ms and play_time_ms must have the same shape")
    return event_time + np.maximum(play_time, 0)


def completion_safe_training_mask(
    event_time_ms: np.ndarray,
    play_time_ms: np.ndarray,
    block_start_ms: int,
    history_ms: int | None = None,
) -> np.ndarray:
    """Select only outcomes strictly observable before ``block_start_ms``."""
    event_time = np.asarray(event_time_ms, dtype=np.int64)
    mask = outcome_available_at(event_time, play_time_ms) < int(block_start_ms)
    if history_ms is not None:
        if history_ms <= 0:
            raise ValueError("history_ms must be positive when supplied")
        mask &= event_time >= int(block_start_ms) - int(history_ms)
    return mask


def synthetic_prequential_predictions(
    event_time_ms: np.ndarray,
    play_time_ms: np.ndarray,
    labels: np.ndarray,
    block_hours: int,
) -> np.ndarray:
    """Small deterministic oracle used only by the causality regression test.

    It predicts the Laplace-smoothed mean of labels that are available before
    each block. It is deliberately simple: the test targets the information
    boundary, not model quality.
    """
    times = np.asarray(event_time_ms, dtype=np.int64)
    y = np.asarray(labels, dtype=np.float64)
    if times.shape != y.shape:
        raise ValueError("event_time_ms and labels must have the same shape")
    block_ms = int(block_hours) * 3_600_000
    if block_ms <= 0:
        raise ValueError("block_hours must be positive")
    units = times // block_ms
    predictions = np.empty(len(times), dtype=np.float64)
    for unit in np.unique(units):
        block_start = int(unit) * block_ms
        train = completion_safe_training_mask(times, play_time_ms, block_start)
        predictions[units == unit] = (float(y[train].sum()) + 1.0) / (
            int(train.sum()) + 2.0
        )
    return predictions
