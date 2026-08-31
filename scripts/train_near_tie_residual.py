#!/usr/bin/env python3
"""Train-only screen of a bounded near-tie hard-negative residual.

All score streams were frozen at 2022-04-13.  The residual is fit on the
morning/afternoon of 2022-04-14, its magnitude is selected on the evening of
2022-04-14, and 2022-04-15..21 is a locked screen.  This script deliberately
does not open the April 22+ standard log.
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


SOURCE_CACHE = (
    ROOT / "runtime" / "stacked-reranker"
    / "out-of-time-components-cutoff-20220413.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "parallel-methods" / "near-tie-screen.json"
DEFAULT_SCORES = ROOT / "runtime" / "parallel-near-tie" / "screen-scores.npz"


def load_rows_through_april21(data_dir: Path) -> list[tuple]:
    """Read only the April 8-21 standard log, preserving its cache row order."""
    rows: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            rows.append((
                int(row["date"]), str(row["user_id"]), str(row["video_id"]),
                str(row["tab"]), float(row["duration_ms"]),
                1 if row["long_view"] != "0" else 0,
                int(row["hourmin"]) // 100, int(row["time_ms"]),
            ))
    if len(rows) != 1_141_112:
        raise RuntimeError(f"Unexpected April 8-21 row count: {len(rows)}")
    return rows


def fractional_user_rank(users: Sequence[object], values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(users) != len(scores):
        raise ValueError(f"User/score alignment failed: {len(users)} != {len(scores)}")
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float64)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[order] = np.arange(len(indices), dtype=np.float64)
        output[indices] = (ranks + 0.5) / len(indices)
    return output


def as_metrics(result: dict) -> dict[str, float | int]:
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def metric_delta(candidate: dict, control: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in ("primary", "gauc", "ndcg5")
    }


def rank_matrix(
    users: list[str], streams: dict[str, np.ndarray], mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    columns = [fractional_user_rank(users, values[mask]) for values in streams.values()]
    matrix = np.column_stack(columns)
    return matrix, matrix[:, list(streams).index("base")]


def build_dense_features(
    rows: list[tuple], ranks: np.ndarray, base_rank: np.ndarray,
    stream_names: list[str], tab_names: list[str],
) -> np.ndarray:
    """Outcome-free disagreement and context features for a tiny residual."""
    duration = np.log1p(np.asarray([row[4] for row in rows], dtype=np.float64))
    hour = np.asarray([row[6] for row in rows], dtype=np.float64)
    tab = np.asarray([row[3] for row in rows], dtype=object)
    non_base = [index for index, name in enumerate(stream_names) if name != "base"]
    disagreement = ranks[:, non_base] - base_rank[:, None]
    spread = np.std(ranks[:, non_base], axis=1, keepdims=True)
    dense = [
        disagreement,
        spread,
        duration[:, None],
        np.sin(2.0 * math.pi * hour / 24.0)[:, None],
        np.cos(2.0 * math.pi * hour / 24.0)[:, None],
    ]
    dense.extend((tab == name).astype(np.float64)[:, None] for name in tab_names)
    return np.column_stack(dense)


def standardize(
    fit: np.ndarray, *others: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (
        (fit - mean) / scale,
        [(values - mean) / scale for values in others],
        {"mean": mean.tolist(), "scale": scale.tolist()},
    )


def mine_near_ties(
    users: list[str], labels: np.ndarray, base_rank: np.ndarray,
    *, max_rank_gap: float = 0.20, negatives_per_positive: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Mine close positive/negative pairs and upweight pairs touching top ten."""
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    pair_weights: list[float] = []
    top_ten_pairs = 0
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        positive = indices[labels[indices] > 0.5]
        negative = indices[labels[indices] <= 0.5]
        if not len(positive) or not len(negative):
            continue
        descending = indices[np.argsort(-base_rank[indices], kind="stable")]
        top_ten = set(descending[:10].tolist())
        for pos in positive:
            gaps = np.abs(base_rank[negative] - base_rank[pos])
            order = np.argsort(gaps, kind="stable")[:negatives_per_positive]
            for local in order:
                neg = int(negative[local])
                gap = float(gaps[local])
                if gap > max_rank_gap:
                    continue
                touches_top = int(pos) in top_ten or neg in top_ten
                top_ten_pairs += int(touches_top)
                # Close pairs carry the most information; top-ten pairs receive
                # an additional deterministic factor of two.
                pair_weights.append((2.0 if touches_top else 1.0) * math.exp(-gap / 0.10))
                positive_indices.append(int(pos))
                negative_indices.append(neg)
    if not positive_indices:
        raise RuntimeError("No near-tie positive/negative pairs were mined")
    weights = np.asarray(pair_weights, dtype=np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return (
        np.asarray(positive_indices, dtype=np.int64),
        np.asarray(negative_indices, dtype=np.int64),
        weights,
        {
            "pairs": len(positive_indices),
            "top_ten_pairs": top_ten_pairs,
            "top_ten_fraction": top_ten_pairs / len(positive_indices),
            "max_fractional_rank_gap": max_rank_gap,
            "negatives_per_positive": negatives_per_positive,
        },
    )


def train_head(
    features: np.ndarray, base_rank: np.ndarray, labels: np.ndarray,
    positive: np.ndarray, negative: np.ndarray, pair_weights: np.ndarray,
    *, seed: int = 2718,
) -> tuple[np.ndarray, dict]:
    """Fit a small tanh residual with BCE anchor and near-tie pair loss."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - research runtime has torch
        raise RuntimeError("Near-tie residual requires installed PyTorch") from error

    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    x = torch.as_tensor(features, dtype=torch.float32)
    base = torch.as_tensor(base_rank, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    pos = torch.as_tensor(positive, dtype=torch.long)
    neg = torch.as_tensor(negative, dtype=torch.long)
    weights = torch.as_tensor(pair_weights, dtype=torch.float32)

    coefficient = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float32))
    residual_bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    calibration_slope_raw = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    prevalence = float(np.clip(labels.mean(), 1e-5, 1.0 - 1e-5))
    calibration_intercept = torch.nn.Parameter(
        torch.tensor(math.log(prevalence / (1.0 - prevalence)), dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        [coefficient, residual_bias, calibration_slope_raw, calibration_intercept],
        lr=0.025,
    )
    history = []
    # The training correction is deliberately bounded to 0.10 rank units.
    # A separate, earlier validation slice selects an equal or smaller bound.
    training_bound = 0.10
    for step in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        unit_correction = torch.tanh(x @ coefficient + residual_bias)
        final_score = base + training_bound * unit_correction
        slope = functional.softplus(calibration_slope_raw) + 1e-4
        logits = calibration_intercept + slope * final_score
        bce = functional.binary_cross_entropy_with_logits(logits, y)
        pair_difference = final_score[pos] - final_score[neg]
        pair_loss = torch.mean(weights * functional.softplus(-pair_difference))
        regularization = 0.02 * torch.mean(coefficient.square())
        loss = bce + 0.20 * pair_loss + regularization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [coefficient, residual_bias, calibration_slope_raw, calibration_intercept], 5.0
        )
        optimizer.step()
        if step == 1 or step % 50 == 0:
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "bce": float(bce.detach()),
                "pair_loss": float(pair_loss.detach()),
                "coefficient_l2": float(torch.linalg.vector_norm(coefficient).detach()),
            })
    return coefficient.detach().numpy(), {
        "bias": float(residual_bias.detach()),
        "calibration_slope": float(functional.softplus(calibration_slope_raw).detach()),
        "calibration_intercept": float(calibration_intercept.detach()),
        "training_bound": training_bound,
        "bce_anchor_weight": 1.0,
        "pair_loss_weight": 0.20,
        "coefficient_l2_weight": 0.02,
        "steps": 300,
        "history": history,
    }


def apply_head(features: np.ndarray, coefficients: np.ndarray, bias: float) -> np.ndarray:
    return np.tanh(features @ coefficients + bias)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()

    all_rows = load_rows_through_april21(ROOT / "external" / "KuaiRand-Pure" / "data")
    meta_rows = [row for row in all_rows if row[0] > 20220413]
    fit_mask = np.asarray([
        row[0] == 20220414 and row[6] < 18 for row in meta_rows
    ], dtype=bool)
    select_mask = np.asarray([
        row[0] == 20220414 and row[6] >= 18 for row in meta_rows
    ], dtype=bool)
    screen_mask = np.asarray([row[0] >= 20220415 for row in meta_rows], dtype=bool)
    fit_rows = [row for row, keep in zip(meta_rows, fit_mask, strict=True) if keep]
    select_rows = [row for row, keep in zip(meta_rows, select_mask, strict=True) if keep]
    screen_rows = [row for row, keep in zip(meta_rows, screen_mask, strict=True) if keep]
    if min(len(fit_rows), len(select_rows), len(screen_rows)) == 0:
        raise RuntimeError("Expected non-empty fit, magnitude-selection, and screen slices")

    with np.load(SOURCE_CACHE, allow_pickle=False) as archive:
        names = ["point0", "point2", "pairwise", "deep", "temporal", "base"]
        if not set(names).issubset(archive.files):
            raise KeyError(f"Aligned cache lacks one of {names}")
        streams = {name: np.asarray(archive[name], dtype=np.float64) for name in names}
    if any(len(values) != len(meta_rows) for values in streams.values()):
        raise RuntimeError(
            f"Frozen-score alignment failed: rows={len(meta_rows)}, "
            f"streams={{name: len(value) for name, value in streams.items()}}"
        )
    if not all(np.isfinite(values).all() for values in streams.values()):
        raise RuntimeError("Frozen cache contains non-finite values")

    fit_users = [row[1] for row in fit_rows]
    select_users = [row[1] for row in select_rows]
    screen_users = [row[1] for row in screen_rows]
    fit_y = np.asarray([row[5] for row in fit_rows], dtype=np.float32)
    select_y = np.asarray([row[5] for row in select_rows], dtype=np.float32)
    screen_y = np.asarray([row[5] for row in screen_rows], dtype=np.float32)
    fit_ranks, fit_base = rank_matrix(fit_users, streams, fit_mask)
    select_ranks, select_base = rank_matrix(select_users, streams, select_mask)
    screen_ranks, screen_base = rank_matrix(screen_users, streams, screen_mask)

    tabs = sorted({row[3] for row in fit_rows})
    raw_fit = build_dense_features(fit_rows, fit_ranks, fit_base, names, tabs)
    raw_select = build_dense_features(select_rows, select_ranks, select_base, names, tabs)
    raw_screen = build_dense_features(screen_rows, screen_ranks, screen_base, names, tabs)
    fit_x, (select_x, screen_x), feature_scaling = standardize(
        raw_fit, raw_select, raw_screen
    )

    positive, negative, pair_weights, mining = mine_near_ties(
        fit_users, fit_y, fit_base
    )
    coefficients, training = train_head(
        fit_x, fit_base, fit_y, positive, negative, pair_weights
    )
    select_unit = apply_head(select_x, coefficients, float(training["bias"]))
    screen_unit = apply_head(screen_x, coefficients, float(training["bias"]))

    select_control = as_metrics(
        runner.evaluate_module.evaluate(select_users, select_y, select_base)
    )
    magnitudes = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
    selection_trials = []
    for magnitude in magnitudes:
        scores = select_base + magnitude * select_unit
        metrics = as_metrics(
            runner.evaluate_module.evaluate(select_users, select_y, scores)
        )
        delta = metric_delta(metrics, select_control)
        eligible = delta["gauc"] >= -1e-12 and delta["ndcg5"] >= -1e-12
        selection_trials.append({
            "magnitude": magnitude,
            "metrics": metrics,
            "delta_vs_zero": delta,
            "eligible": eligible,
        })
    eligible_trials = [trial for trial in selection_trials if trial["eligible"]]
    selected = max(
        eligible_trials,
        key=lambda trial: (trial["metrics"]["primary"], -trial["magnitude"]),
    )
    selected_magnitude = float(selected["magnitude"])

    screen_control_scores = screen_base.copy()
    screen_candidate_scores = screen_base + selected_magnitude * screen_unit
    screen_control = as_metrics(
        runner.evaluate_module.evaluate(screen_users, screen_y, screen_control_scores)
    )
    screen_candidate = as_metrics(
        runner.evaluate_module.evaluate(screen_users, screen_y, screen_candidate_scores)
    )
    screen_delta = metric_delta(screen_candidate, screen_control)
    passed = all(screen_delta[key] > 0.0 for key in ("primary", "gauc", "ndcg5"))

    if not (
        np.isfinite(screen_control_scores).all()
        and np.isfinite(screen_candidate_scores).all()
        and np.isfinite(screen_unit).all()
    ):
        raise RuntimeError("Non-finite output score detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        dates=np.asarray([row[0] for row in screen_rows], dtype=np.int32),
        users=np.asarray(screen_users, dtype="U32"),
        labels=screen_y,
        frozen_base_raw=streams["base"][screen_mask].astype(np.float32),
        frozen_base_rank=screen_base.astype(np.float32),
        residual_unit=screen_unit.astype(np.float32),
        candidate_scores=screen_candidate_scores.astype(np.float32),
    )
    report = {
        "experiment": "bounded near-tie hard-negative residual",
        "status": "passed_train_only_screen" if passed else "rejected_at_train_only_screen",
        "protocol": {
            "frozen_source_model_fit": "2022-04-08..2022-04-13",
            "residual_fit": "2022-04-14 hours 00..17",
            "magnitude_selection": "2022-04-14 hours 18..23",
            "locked_screen": "2022-04-15..2022-04-21",
            "confirmation_labels_read": False,
            "hidden_test_accessed": False,
            "alignment": (
                "All streams came from one cutoff-20220413 cache and exact length equality "
                "with rows dated after April 13 was required."
            ),
        },
        "data": {
            "fit_rows": len(fit_rows),
            "selection_rows": len(select_rows),
            "screen_rows": len(screen_rows),
            "screen_users": len(set(screen_users)),
            "source_cache": str(SOURCE_CACHE.relative_to(ROOT)),
            "source_streams": names,
        },
        "model": {
            "description": (
                "Linear disagreement/context head passed through tanh; pointwise BCE anchors "
                "the frozen base while weighted same-user near-tie pairs sharpen close ordering."
            ),
            "feature_count": int(fit_x.shape[1]),
            "tab_levels": tabs,
            "feature_scaling": feature_scaling,
            "coefficient_l2": float(np.linalg.norm(coefficients)),
            "training": training,
            "pair_mining": mining,
        },
        "magnitude_selection": {
            "zero_residual_control": select_control,
            "trials": selection_trials,
            "selected_magnitude": selected_magnitude,
            "selection_rule": (
                "Highest primary among magnitudes with nonnegative GAUC and nDCG deltas; "
                "ties prefer the smaller magnitude."
            ),
        },
        "locked_screen": {
            "exact_zero_residual_control": screen_control,
            "candidate": screen_candidate,
            "delta": screen_delta,
            "gate": "primary, GAUC, and nDCG@5 must each improve strictly",
            "passed": passed,
        },
        "recommendation": (
            "Recommend a separately authorized April 22-28 confirmation."
            if passed else
            "Do not confirm; retain the frozen base because the locked all-metric gate failed."
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(args.scores_output.resolve().relative_to(ROOT)),
            "report": str(args.output.resolve().relative_to(ROOT)),
        },
        "resource_usage": tracker.finish(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
