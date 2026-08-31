#!/usr/bin/env python3
"""Leak-free UMRE-lite screen over frozen out-of-time score streams.

The three source models were fit through 2022-04-13.  Their 2022-04-14
predictions fit a small global monotonic fusion layer; 2022-04-15..21 is held
out for the only metric comparison performed by this script.  No row or label
after 2022-04-21 is loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


SOURCE_CACHE = ROOT / "runtime" / "stacked-reranker" / "out-of-time-components-cutoff-20220413.npz"
DEFAULT_OUTPUT = ROOT / "results" / "parallel-methods" / "umre-lite-screen.json"
DEFAULT_SCORES = ROOT / "runtime" / "parallel-umre-lite" / "screen-scores.npz"


def as_metrics(result: dict) -> dict[str, float | int]:
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def load_standard_rows_through_april21(data_dir: Path) -> list[tuple]:
    """Load the single April 8-21 standard-log file while preserving row order."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]
    rows = []
    # Deliberately do not open the later standard log: this experiment is a
    # train-only screen and must not load April 22-28 labels.
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append((
                int(row["date"]), row["user_id"], row["video_id"],
                video_to_author.get(row["video_id"], "UNK"), row["tab"],
                float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0,
                int(row["hourmin"]) // 100, int(row["time_ms"]),
                float(row["play_time_ms"]),
            ))
    if len(rows) != 1_141_112:
        raise RuntimeError(f"Unexpected April 8-21 row count: {len(rows)}")
    return rows


def fractional_user_rank(users: Sequence[object], scores: np.ndarray) -> np.ndarray:
    """Map every source to [0, 1] ranks independently inside each user slate."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(users) != len(values):
        raise ValueError(f"User/score alignment failed: {len(users)} != {len(values)}")
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.empty(len(values), dtype=np.float64)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.argsort(values[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = np.arange(len(indices), dtype=np.float64)
        output[indices] = (ranks + 0.5) / len(indices)
    return output


def pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted increasing isotonic projection (pool-adjacent-violators)."""
    block_values: list[float] = []
    block_weights: list[float] = []
    block_lengths: list[int] = []
    for raw_value, raw_weight in zip(values, weights, strict=True):
        block_values.append(float(raw_value))
        block_weights.append(float(raw_weight))
        block_lengths.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total = block_weights[-2] + block_weights[-1]
            merged = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / total
            merged_length = block_lengths[-2] + block_lengths[-1]
            block_values[-2:] = [merged]
            block_weights[-2:] = [total]
            block_lengths[-2:] = [merged_length]
    return np.concatenate([
        np.full(length, value, dtype=np.float64)
        for value, length in zip(block_values, block_lengths, strict=True)
    ])


def fit_monotonic_knots(
    rank: np.ndarray, labels: np.ndarray, *, knots: int = 8, smoothing: float = 2048.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit an eight-knot, Bayesian-shrunk monotonic response curve."""
    centers = (np.arange(knots, dtype=np.float64) + 0.5) / knots
    bins = np.minimum((np.asarray(rank) * knots).astype(np.int64), knots - 1)
    counts = np.bincount(bins, minlength=knots).astype(np.float64)
    positives = np.bincount(bins, weights=labels, minlength=knots).astype(np.float64)
    prior = float(np.mean(labels))
    raw = (positives + smoothing * prior) / (counts + smoothing)
    monotonic = pava(raw, counts + smoothing)
    return centers, monotonic, {
        "counts": counts.astype(int).tolist(),
        "raw_rates": raw.tolist(),
        "monotonic_rates": monotonic.tolist(),
        "global_rate": prior,
        "smoothing_rows_per_knot": smoothing,
    }


def apply_monotonic_knots(rank: np.ndarray, centers: np.ndarray, rates: np.ndarray) -> np.ndarray:
    return np.interp(rank, centers, rates, left=float(rates[0]), right=float(rates[-1]))


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights summing to one."""
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    condition = sorted_values - cumulative / np.arange(1, len(values) + 1) > 0
    rho = int(np.flatnonzero(condition)[-1])
    theta = cumulative[rho] / (rho + 1)
    return np.maximum(values - theta, 0.0)


def fit_regularized_fusion(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    weight_penalty: float = 0.25,
    slope_penalty: float = 0.02,
    steps: int = 1200,
    learning_rate: float = 0.05,
) -> tuple[np.ndarray, float, float, dict]:
    """Fit nonnegative simplex weights with shrinkage toward equal fusion."""
    width = features.shape[1]
    target = np.full(width, 1.0 / width, dtype=np.float64)
    weights = target.copy()
    prevalence = float(np.clip(labels.mean(), 1e-6, 1.0 - 1e-6))
    intercept = math.log(prevalence / (1.0 - prevalence))
    slope = 1.0
    for step in range(steps):
        combined = features @ weights
        logits = np.clip(intercept + slope * combined, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = probabilities - labels
        decay = 1.0 / math.sqrt(1.0 + step / 200.0)
        rate = learning_rate * decay
        gradient_weights = (
            slope * (features.T @ residual) / len(labels)
            + 2.0 * weight_penalty * (weights - target)
        )
        gradient_slope = float(np.mean(residual * combined)) + 2.0 * slope_penalty * (slope - 1.0)
        gradient_intercept = float(np.mean(residual))
        weights = project_simplex(weights - rate * gradient_weights)
        slope = max(0.0, slope - rate * gradient_slope)
        intercept -= rate * gradient_intercept
    logits = np.clip(intercept + slope * (features @ weights), -30.0, 30.0)
    loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    return weights, slope, intercept, {
        "loss": loss,
        "steps": steps,
        "weight_penalty_to_equal": weight_penalty,
        "slope_penalty_to_one": slope_penalty,
        "learning_rate": learning_rate,
    }


def standardize_fit_apply(fit: np.ndarray, screen: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (fit - mean) / scale, (screen - mean) / scale, {
        "fit_mean": mean.tolist(), "fit_scale": scale.tolist(),
    }


def deltas(candidate: dict, control: dict) -> dict[str, float]:
    return {
        "primary": float(candidate["primary"] - control["primary"]),
        "gauc": float(candidate["gauc"] - control["gauc"]),
        "ndcg5": float(candidate["ndcg5"] - control["ndcg5"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()

    # Preserve the original CSV order used by the aligned cached predictions.
    all_rows = load_standard_rows_through_april21(
        ROOT / "external" / "KuaiRand-Pure" / "data"
    )
    meta_rows = [row for row in all_rows if int(row[0]) > 20220413]
    fit_mask = np.asarray([int(row[0]) == 20220414 for row in meta_rows])
    screen_mask = np.asarray([int(row[0]) >= 20220415 for row in meta_rows])
    fit_rows = [row for row, keep in zip(meta_rows, fit_mask, strict=True) if keep]
    screen_rows = [row for row, keep in zip(meta_rows, screen_mask, strict=True) if keep]
    if not fit_rows or not screen_rows:
        raise RuntimeError("Expected non-empty April 14 fit and April 15-21 screen rows")

    with np.load(SOURCE_CACHE, allow_pickle=False) as archive:
        required = {"point0", "point2", "pairwise", "deep"}
        if not required.issubset(archive.files):
            raise KeyError(f"Missing aligned score streams: {sorted(required - set(archive.files))}")
        components = {
            "pointwise_fm": 0.5 * (
                np.asarray(archive["point0"], dtype=np.float64)
                + np.asarray(archive["point2"], dtype=np.float64)
            ),
            "pairwise_fm": np.asarray(archive["pairwise"], dtype=np.float64),
            "deepfm": np.asarray(archive["deep"], dtype=np.float64),
        }
    if any(len(values) != len(meta_rows) for values in components.values()):
        lengths = {name: len(values) for name, values in components.items()}
        raise RuntimeError(f"Artifact/date alignment failed: rows={len(meta_rows)}, streams={lengths}")

    fit_users = [str(row[1]) for row in fit_rows]
    fit_y = np.asarray([row[6] for row in fit_rows], dtype=np.float64)
    screen_users = [str(row[1]) for row in screen_rows]
    screen_y = np.asarray([row[6] for row in screen_rows], dtype=np.float64)

    fit_rank_columns, screen_rank_columns = [], []
    standalone = {}
    for name, values in components.items():
        fit_rank = fractional_user_rank(fit_users, values[fit_mask])
        screen_rank = fractional_user_rank(screen_users, values[screen_mask])
        fit_rank_columns.append(fit_rank)
        screen_rank_columns.append(screen_rank)
        standalone[name] = as_metrics(
            runner.evaluate_module.evaluate(screen_users, screen_y, screen_rank)
        )
    fit_ranks = np.column_stack(fit_rank_columns)
    screen_ranks = np.column_stack(screen_rank_columns)

    # Exact matching control: same rows, sources, regularizer and optimizer; only
    # the per-source monotonic response transformation is omitted.
    control_fit, control_screen, control_standardization = standardize_fit_apply(
        fit_ranks, screen_ranks
    )
    control_weights, control_slope, control_intercept, control_fit_info = fit_regularized_fusion(
        control_fit, fit_y
    )
    control_scores = control_intercept + control_slope * (control_screen @ control_weights)
    control_metrics = as_metrics(
        runner.evaluate_module.evaluate(screen_users, screen_y, control_scores)
    )

    transformed_fit_columns, transformed_screen_columns = [], []
    curves = {}
    for column, name in enumerate(components):
        centers, rates, curve = fit_monotonic_knots(fit_ranks[:, column], fit_y)
        transformed_fit_columns.append(
            apply_monotonic_knots(fit_ranks[:, column], centers, rates)
        )
        transformed_screen_columns.append(
            apply_monotonic_knots(screen_ranks[:, column], centers, rates)
        )
        curves[name] = {"centers": centers.tolist(), **curve}
    transformed_fit = np.column_stack(transformed_fit_columns)
    transformed_screen = np.column_stack(transformed_screen_columns)
    umre_fit, umre_screen, umre_standardization = standardize_fit_apply(
        transformed_fit, transformed_screen
    )
    umre_weights, umre_slope, umre_intercept, umre_fit_info = fit_regularized_fusion(
        umre_fit, fit_y
    )
    umre_scores = umre_intercept + umre_slope * (umre_screen @ umre_weights)
    umre_metrics = as_metrics(
        runner.evaluate_module.evaluate(screen_users, screen_y, umre_scores)
    )
    gain = deltas(umre_metrics, control_metrics)
    passed = (
        gain["primary"] >= 1e-5
        and gain["gauc"] >= -1e-12
        and gain["ndcg5"] >= -1e-12
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        screen_dates=np.asarray([row[0] for row in screen_rows], dtype=np.int32),
        control_scores=control_scores.astype(np.float32),
        umre_scores=umre_scores.astype(np.float32),
        screen_labels=screen_y.astype(np.float32),
    )
    result = {
        "experiment": "global UMRE-lite monotonic fusion",
        "status": "passed_train_only_screen" if passed else "rejected_at_train_only_screen",
        "protocol": {
            "source_model_fit": "2022-04-08..2022-04-13",
            "fusion_fit": "2022-04-14",
            "screen": "2022-04-15..2022-04-21",
            "confirmation_labels_read": False,
            "hidden_test_accessed": False,
            "alignment": (
                "The cache contains predictions for the source script's rows dated after "
                "cutoff 20220413. Exact length equality against April 14-21 rows was required."
            ),
            "source_note": (
                "The initially preferred ordered-CatBoost/watch-ratio-DeepFM/YetiRank trio had "
                "no verified aligned April 14-21 cache. The smallest honest alternative uses "
                "pointwise FM, pairwise FM, and DeepFM streams from one aligned cache."
            ),
        },
        "data": {
            "fusion_fit_rows": len(fit_rows),
            "screen_rows": len(screen_rows),
            "screen_users": len(set(screen_users)),
            "source_cache": str(SOURCE_CACHE.relative_to(ROOT)),
            "source_streams": list(components),
        },
        "standalone_screen_metrics": standalone,
        "control": {
            "description": "regularized nonnegative linear rank consensus on the same three streams",
            "metrics": control_metrics,
            "weights": dict(zip(components, control_weights.tolist(), strict=True)),
            "slope": control_slope,
            "intercept": control_intercept,
            "standardization": control_standardization,
            "fit": control_fit_info,
        },
        "umre_lite": {
            "description": "eight-knot monotonic transforms plus the matching regularized nonnegative fusion",
            "metrics": umre_metrics,
            "weights": dict(zip(components, umre_weights.tolist(), strict=True)),
            "slope": umre_slope,
            "intercept": umre_intercept,
            "standardization": umre_standardization,
            "fit": umre_fit_info,
            "curves": curves,
        },
        "delta_vs_matching_control": gain,
        "screen_gate": {
            "required": "primary >= +0.00001 and both component metrics nonnegative",
            "passed": passed,
        },
        "recommendation": (
            "Run a separately authorized April 22-28 confirmation with refit score streams."
            if passed else
            "Do not confirm or add a personalized gate; the global monotonic layer did not clear the locked screen."
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(args.scores_output.resolve().relative_to(ROOT)),
            "report": str(args.output.resolve().relative_to(ROOT)),
        },
        "resource_usage": tracker.finish(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
