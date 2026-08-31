from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .champion import within_user_rank
from .neighbors import _metadata, _view_value


SUPPORTED_PAPER_SIGNALS = (
    "user_item_affinity",
    "user_author_affinity",
    "user_music_affinity",
    "user_tag_affinity",
    "user_video_type_affinity",
    "item_prior",
    "author_prior",
    "music_prior",
    "tag_prior",
    "video_type_prior",
    "repeat_penalty",
    "slate_author_frequency",
    "slate_music_frequency",
    "duration_match",
)


def _program(parameters: dict[str, Any]) -> tuple[list[str], np.ndarray, float, float]:
    raw_signals = parameters.get("paper_signals", [])
    signals = [str(signal) for signal in raw_signals]
    if not 1 <= len(signals) <= 6:
        raise ValueError("paper_signals must contain between one and six signals")
    if len(set(signals)) != len(signals) or any(
        signal not in SUPPORTED_PAPER_SIGNALS for signal in signals
    ):
        raise ValueError("paper_signals must contain distinct supported signal names")
    raw_weights = parameters.get("paper_signal_weights", [])
    if len(raw_weights) != len(signals):
        raise ValueError("paper_signal_weights must provide one value per signal")
    weights = np.asarray(raw_weights, dtype=np.float32)
    if not np.all(np.isfinite(weights)) or not np.any(np.abs(weights) > 0):
        raise ValueError("paper_signal_weights must be finite and include a non-zero value")
    if np.any(np.abs(weights) > 1):
        raise ValueError("paper_signal_weights must be in [-1, 1]")
    weights /= float(np.abs(weights).sum())
    smoothing = max(0.1, min(float(parameters.get("paper_smoothing", 8.0)), 100.0))
    entity_smoothing = max(
        0.1, min(float(parameters.get("paper_item_smoothing", 20.0)), 200.0)
    )
    return signals, weights, smoothing, entity_smoothing


def build_paper_signal_scores(
    train_rows: list[tuple],
    valid_rows: list[tuple],
    data_dir: Path,
    parameters: dict[str, Any],
    *,
    video_metadata: dict[str, dict[str, str]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Interpret a reviewed signal program without reading validation outcomes."""
    signals, weights, smoothing, entity_smoothing = _program(parameters)
    metadata = video_metadata if video_metadata is not None else _metadata(data_dir)
    train_labels = np.asarray([float(row[6]) for row in train_rows], dtype=np.float64)
    global_prior = float(train_labels.mean()) if len(train_labels) else 0.0
    valid_users = [str(row[1]) for row in valid_rows]

    needed_views: set[str] = set()
    for signal in signals:
        for view in ("item", "author", "music", "tag", "video_type"):
            if signal in {f"user_{view}_affinity", f"{view}_prior"}:
                needed_views.add(view)

    view_statistics: dict[str, dict[str, Any]] = {}
    for view in needed_views:
        entity_count: defaultdict[str, int] = defaultdict(int)
        entity_positive: defaultdict[str, float] = defaultdict(float)
        user_entity_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        user_entity_positive: defaultdict[tuple[str, str], float] = defaultdict(float)
        for row, label in zip(train_rows, train_labels):
            user = str(row[1])
            entity = _view_value(row, view, metadata)
            entity_count[entity] += 1
            entity_positive[entity] += float(label)
            user_entity_count[(user, entity)] += 1
            user_entity_positive[(user, entity)] += float(label)
        view_statistics[view] = {
            "entity_count": entity_count,
            "entity_positive": entity_positive,
            "user_entity_count": user_entity_count,
            "user_entity_positive": user_entity_positive,
        }

    repeat_counts = Counter((str(row[1]), str(row[2])) for row in train_rows)
    slate_author_counts = Counter(
        (str(row[1]), _view_value(row, "author", metadata)) for row in valid_rows
    )
    slate_music_counts = Counter(
        (str(row[1]), _view_value(row, "music", metadata)) for row in valid_rows
    )
    user_duration_sum: defaultdict[str, float] = defaultdict(float)
    user_duration_count: defaultdict[str, int] = defaultdict(int)
    for row in train_rows:
        user = str(row[1])
        user_duration_sum[user] += float(np.log1p(max(float(row[5]), 0.0)))
        user_duration_count[user] += 1
    global_duration = float(np.mean([
        np.log1p(max(float(row[5]), 0.0)) for row in train_rows
    ])) if train_rows else 0.0

    raw_signal_values: dict[str, np.ndarray] = {}
    signal_support: dict[str, float] = {}
    for signal in signals:
        values = np.zeros(len(valid_rows), dtype=np.float32)
        supported = np.zeros(len(valid_rows), dtype=bool)
        if signal in {"repeat_penalty", "slate_author_frequency", "slate_music_frequency", "duration_match"}:
            for index, row in enumerate(valid_rows):
                user = str(row[1])
                if signal == "repeat_penalty":
                    count = repeat_counts[(user, str(row[2]))]
                    values[index] = -float(np.log1p(count))
                    supported[index] = count > 0
                elif signal == "slate_author_frequency":
                    count = slate_author_counts[(user, _view_value(row, "author", metadata))]
                    values[index] = -float(np.log1p(max(0, count - 1)))
                    supported[index] = count > 1
                elif signal == "slate_music_frequency":
                    count = slate_music_counts[(user, _view_value(row, "music", metadata))]
                    values[index] = -float(np.log1p(max(0, count - 1)))
                    supported[index] = count > 1
                else:
                    mean_duration = (
                        user_duration_sum[user] / user_duration_count[user]
                        if user_duration_count[user] else global_duration
                    )
                    values[index] = -abs(
                        float(np.log1p(max(float(row[5]), 0.0))) - mean_duration
                    )
                    supported[index] = user_duration_count[user] > 0
        else:
            is_user_signal = signal.startswith("user_")
            view = signal.removeprefix("user_").removesuffix("_affinity") if is_user_signal else signal.removesuffix("_prior")
            stats = view_statistics[view]
            for index, row in enumerate(valid_rows):
                user = str(row[1])
                entity = _view_value(row, view, metadata)
                entity_count = stats["entity_count"][entity]
                entity_prior = (
                    stats["entity_positive"][entity] + entity_smoothing * global_prior
                ) / (entity_count + entity_smoothing)
                if is_user_signal:
                    key = (user, entity)
                    count = stats["user_entity_count"][key]
                    values[index] = (
                        stats["user_entity_positive"][key] + smoothing * entity_prior
                    ) / (count + smoothing)
                    supported[index] = count > 0
                else:
                    values[index] = entity_prior
                    supported[index] = entity_count > 0
        raw_signal_values[signal] = values
        signal_support[signal] = float(np.mean(supported)) if len(supported) else 0.0

    ranked = np.stack([
        within_user_rank(valid_users, raw_signal_values[signal]) for signal in signals
    ])
    scores = np.asarray(weights @ ranked, dtype=np.float32)
    diagnostics = {
        "kind": "reviewed_paper_signal_program",
        "executor_slug": str(parameters.get("paper_executor_slug", "")),
        "signals": signals,
        "signal_weights": [float(value) for value in weights],
        "signal_support": signal_support,
        "smoothing": smoothing,
        "entity_smoothing": entity_smoothing,
        "training_rows": len(train_rows),
        "prediction_rows": len(valid_rows),
        "validation_outcomes_accessed": False,
        "hidden_test_accessed": False,
    }
    return scores, diagnostics
