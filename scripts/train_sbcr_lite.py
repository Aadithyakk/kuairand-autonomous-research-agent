#!/usr/bin/env python3
"""Small, leak-free SBCR-inspired calibrated-ranking ablation.

The teacher was frozen through 2022-04-11.  Candidate ranking weights are
selected with April 12-13 training and April 14 validation, then the selected
candidate and exact pointwise/calibration control are refit through April 14
and screened on April 15-21.  The April 22+ file is never opened unless the
strict screen gate passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


TEACHER_CACHE = (
    ROOT / "runtime" / "stacked-reranker"
    / "out-of-time-base-cutoff-20220411.npz"
)
DEFAULT_OUTPUT = ROOT / "results" / "calibrated-ranking" / "sbcr-screen.json"
DEFAULT_SCORES = (
    ROOT / "runtime" / "parallel-calibrated-ranking" / "sbcr"
    / "screen-scores.npz"
)
# Predeclared before reading any selection or screen metric.  Zero is an
# explicit candidate and all nonzero weights are intentionally tiny.
BOOST_WEIGHT_GRID = (0.0, 0.0025, 0.005, 0.01)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows_through_april21(data_dir: Path) -> list[tuple]:
    """Read only the first standard-log file and preserve its original order."""
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


def group_teacher_context(
    users: Sequence[object], scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create label-free item and per-user teacher-distribution context.

    Returns an order-equivalent standardized teacher score, per-item ranking
    features, and query-only calibration context.  No current or peer outcome
    appears in these features.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(users) != len(values):
        raise ValueError(f"Teacher alignment failed: {len(users)} != {len(values)}")
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    base = np.empty(len(values), dtype=np.float64)
    item_context = np.empty((len(values), 11), dtype=np.float64)
    query_context = np.empty((len(values), 7), dtype=np.float64)
    group_sizes = []
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        group = values[indices]
        mean = float(group.mean())
        standard_deviation = max(float(group.std()), 1e-6)
        z = (group - mean) / standard_deviation
        order = np.argsort(group, kind="stable")
        fractional = np.empty(len(group), dtype=np.float64)
        fractional[order] = (np.arange(len(group), dtype=np.float64) + 0.5) / len(group)
        rank_z = (fractional - fractional.mean()) / max(float(fractional.std()), 1e-6)
        quantiles = np.quantile(z, (0.10, 0.25, 0.50, 0.75, 0.90))
        log_count = math.log1p(len(group))
        base[indices] = rank_z
        item_context[indices] = np.column_stack([
            z,
            fractional,
            np.full(len(group), quantiles[0]),
            np.full(len(group), quantiles[1]),
            np.full(len(group), quantiles[2]),
            np.full(len(group), quantiles[3]),
            np.full(len(group), quantiles[4]),
            np.full(len(group), math.log(standard_deviation)),
            np.full(len(group), log_count),
            quantiles[4] - z,
            z - quantiles[0],
        ])
        query_context[indices] = np.column_stack([
            np.full(len(group), mean),
            np.full(len(group), math.log(standard_deviation)),
            np.full(len(group), quantiles[0]),
            np.full(len(group), quantiles[1]),
            np.full(len(group), quantiles[3]),
            np.full(len(group), quantiles[4]),
            np.full(len(group), log_count),
        ])
        group_sizes.append(len(group))
    return base, item_context, query_context, {
        "users": len(groups),
        "mean_rows_per_user": float(np.mean(group_sizes)),
        "max_rows_per_user": int(max(group_sizes)),
        "outcome_features": 0,
    }


def build_student_features(
    rows: list[tuple], teacher_context: np.ndarray, tab_names: list[str],
) -> np.ndarray:
    duration = np.log1p(np.asarray([row[4] for row in rows], dtype=np.float64))
    hour = np.asarray([row[6] for row in rows], dtype=np.float64)
    tabs = np.asarray([row[3] for row in rows], dtype=object)
    columns = [
        teacher_context,
        duration[:, None],
        np.sin(2.0 * math.pi * hour / 24.0)[:, None],
        np.cos(2.0 * math.pi * hour / 24.0)[:, None],
    ]
    columns.extend((tabs == name).astype(np.float64)[:, None] for name in tab_names)
    return np.column_stack(columns)


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    return mean, np.where(scale > 1e-7, scale, 1.0)


def apply_standardizer(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray,
) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def binary_logloss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 20,
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    indices = np.minimum((p * bins).astype(np.int64), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            error += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))
    return error


def train_student(
    features: np.ndarray,
    calibration_context: np.ndarray,
    base_scores: np.ndarray,
    labels: np.ndarray,
    *,
    boost_weight: float,
    seed: int,
    epochs: int = 8,
    batch_size: int = 8192,
) -> tuple[dict[str, np.ndarray | float], dict]:
    """Train shuffled pointwise and detached-calibration towers.

    The self-boost term is sample-separable: positives are compared with the
    frozen teacher's upper per-user anchor and negatives with its lower anchor.
    Therefore mini-batches remain fully shuffled, as in SBCR.
    """
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("SBCR-lite requires installed PyTorch") from error

    torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    random = np.random.default_rng(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    context = torch.as_tensor(calibration_context, dtype=torch.float32)
    base = torch.as_tensor(base_scores, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)

    width = x.shape[1]
    context_width = context.shape[1]
    ranking_first = torch.nn.Linear(width, 32)
    ranking_second = torch.nn.Linear(32, 1)
    calibration_first = torch.nn.Linear(context_width, 12)
    calibration_second = torch.nn.Linear(12, 2)
    point_slope_raw = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    prevalence = float(np.clip(labels.mean(), 1e-5, 1.0 - 1e-5))
    point_intercept = torch.nn.Parameter(
        torch.tensor(math.log(prevalence / (1.0 - prevalence)), dtype=torch.float32)
    )
    parameters = [
        *ranking_first.parameters(), *ranking_second.parameters(),
        *calibration_first.parameters(), *calibration_second.parameters(),
        point_slope_raw, point_intercept,
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.003)
    history = []
    correction_bound = 0.20
    # Columns 3 and 5 of the unstandardized item teacher context correspond to
    # its 25th and 75th percentiles.  Their standardized versions are only
    # neural inputs; fixed order-equivalent rank anchors below provide the
    # sample-separable self-boost comparisons.
    lower_anchor, upper_anchor = -0.675, 0.675
    for epoch in range(1, epochs + 1):
        order = random.permutation(len(labels))
        sums = {"loss": 0.0, "point_bce": 0.0, "boost": 0.0, "calibration_bce": 0.0}
        rows_seen = 0
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            xb, cb, bb, yb = x[indices], context[indices], base[indices], y[indices]
            optimizer.zero_grad(set_to_none=True)
            residual = correction_bound * torch.tanh(
                ranking_second(torch.relu(ranking_first(xb))).squeeze(1)
            )
            ranking_score = bb + residual
            point_slope = functional.softplus(point_slope_raw) + 1e-4
            point_logits = point_intercept + point_slope * ranking_score
            point_bce = functional.binary_cross_entropy_with_logits(point_logits, yb)

            # Per-row self boost against context anchors from the old teacher.
            boost = torch.mean(
                yb * functional.softplus(upper_anchor - ranking_score)
                + (1.0 - yb) * functional.softplus(ranking_score - lower_anchor)
            )

            calibration_parameters = calibration_second(
                torch.relu(calibration_first(cb))
            )
            calibration_slope = functional.softplus(calibration_parameters[:, 0]) + 1e-4
            calibration_logits = (
                calibration_parameters[:, 1]
                + calibration_slope * ranking_score.detach()
            )
            calibration_bce = functional.binary_cross_entropy_with_logits(
                calibration_logits, yb
            )
            regularization = 1e-4 * sum(
                torch.mean(parameter.square())
                for parameter in ranking_first.parameters()
            )
            loss = point_bce + boost_weight * boost + calibration_bce + regularization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            count = len(indices)
            rows_seen += count
            for key, value in (
                ("loss", loss), ("point_bce", point_bce), ("boost", boost),
                ("calibration_bce", calibration_bce),
            ):
                sums[key] += float(value.detach()) * count
        history.append({
            "epoch": epoch,
            **{key: value / rows_seen for key, value in sums.items()},
        })
    state = {
        "ranking_first_weight": ranking_first.weight.detach().numpy(),
        "ranking_first_bias": ranking_first.bias.detach().numpy(),
        "ranking_second_weight": ranking_second.weight.detach().numpy(),
        "ranking_second_bias": ranking_second.bias.detach().numpy(),
        "calibration_first_weight": calibration_first.weight.detach().numpy(),
        "calibration_first_bias": calibration_first.bias.detach().numpy(),
        "calibration_second_weight": calibration_second.weight.detach().numpy(),
        "calibration_second_bias": calibration_second.bias.detach().numpy(),
        "point_slope_raw": float(point_slope_raw.detach()),
        "point_intercept": float(point_intercept.detach()),
    }
    return state, {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.003,
        "correction_bound": correction_bound,
        "boost_weight": boost_weight,
        "point_bce_weight": 1.0,
        "detached_calibration_bce_weight": 1.0,
        "calibration_gradient_stopped_at_ranking_score": True,
        "sample_level_shuffle": True,
        "history": history,
    }


def linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return values @ weight.T + bias


def predict_student(
    state: dict[str, np.ndarray | float],
    features: np.ndarray,
    calibration_context: np.ndarray,
    base_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(
        linear(
            features,
            np.asarray(state["ranking_first_weight"]),
            np.asarray(state["ranking_first_bias"]),
        ),
        0.0,
    )
    residual = 0.20 * np.tanh(
        linear(
            hidden,
            np.asarray(state["ranking_second_weight"]),
            np.asarray(state["ranking_second_bias"]),
        ).reshape(-1)
    )
    ranking = np.asarray(base_scores, dtype=np.float64) + residual
    calibration_hidden = np.maximum(
        linear(
            calibration_context,
            np.asarray(state["calibration_first_weight"]),
            np.asarray(state["calibration_first_bias"]),
        ),
        0.0,
    )
    calibration_parameters = linear(
        calibration_hidden,
        np.asarray(state["calibration_second_weight"]),
        np.asarray(state["calibration_second_bias"]),
    )
    slope = np.logaddexp(0.0, calibration_parameters[:, 0]) + 1e-4
    logits = calibration_parameters[:, 1] + slope * ranking
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return ranking, probabilities


def evaluate_model(
    state: dict[str, np.ndarray | float],
    features: np.ndarray,
    calibration_context: np.ndarray,
    base_scores: np.ndarray,
    users: list[str],
    labels: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    ranking, probabilities = predict_student(
        state, features, calibration_context, base_scores
    )
    metrics = as_metrics(runner.evaluate_module.evaluate(users, labels, ranking))
    metrics["calibration_logloss"] = binary_logloss(labels, probabilities)
    metrics["calibration_ece20"] = expected_calibration_error(labels, probabilities)
    return metrics, ranking, probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()
    incidents: list[str] = [
        "The pre-existing frozen-teacher NPZ has no companion provenance manifest; its "
        "training cutoff is encoded in the filename. Exact date-filtered row-count/order "
        "alignment and finite values were verified before use."
    ]

    rows = load_rows_through_april21(ROOT / "external" / "KuaiRand-Pure" / "data")
    meta_rows = [row for row in rows if row[0] > 20220411]
    selection_train_mask = np.asarray(
        [20220412 <= row[0] <= 20220413 for row in meta_rows], dtype=bool
    )
    selection_valid_mask = np.asarray([row[0] == 20220414 for row in meta_rows], dtype=bool)
    refit_mask = np.asarray(
        [20220412 <= row[0] <= 20220414 for row in meta_rows], dtype=bool
    )
    screen_mask = np.asarray([20220415 <= row[0] <= 20220421 for row in meta_rows], dtype=bool)

    with np.load(TEACHER_CACHE, allow_pickle=False) as archive:
        if "scores" not in archive.files:
            raise KeyError("Frozen teacher archive has no scores array")
        teacher = np.asarray(archive["scores"], dtype=np.float64).reshape(-1)
    if len(teacher) != len(meta_rows):
        raise RuntimeError(
            f"Teacher/date alignment failed: {len(teacher)} != {len(meta_rows)}"
        )
    if not np.isfinite(teacher).all():
        raise RuntimeError("Frozen teacher contains non-finite scores")

    masks = {
        "selection_train": selection_train_mask,
        "selection_valid": selection_valid_mask,
        "refit": refit_mask,
        "screen": screen_mask,
    }
    split_rows = {
        name: [row for row, keep in zip(meta_rows, mask, strict=True) if keep]
        for name, mask in masks.items()
    }
    tab_names = sorted({row[3] for row in split_rows["selection_train"]})
    prepared = {}
    context_metadata = {}
    for name, mask in masks.items():
        current_rows = split_rows[name]
        users = [row[1] for row in current_rows]
        labels = np.asarray([row[5] for row in current_rows], dtype=np.float32)
        base, item_context, query_context, metadata = group_teacher_context(
            users, teacher[mask]
        )
        raw_features = build_student_features(current_rows, item_context, tab_names)
        prepared[name] = {
            "rows": current_rows,
            "users": users,
            "labels": labels,
            "base": base,
            "raw_features": raw_features,
            "raw_calibration_context": query_context,
        }
        context_metadata[name] = metadata

    selection_feature_mean, selection_feature_scale = fit_standardizer(
        prepared["selection_train"]["raw_features"]
    )
    selection_context_mean, selection_context_scale = fit_standardizer(
        prepared["selection_train"]["raw_calibration_context"]
    )
    for name in ("selection_train", "selection_valid"):
        prepared[name]["features"] = apply_standardizer(
            prepared[name]["raw_features"], selection_feature_mean, selection_feature_scale
        )
        prepared[name]["calibration_context"] = apply_standardizer(
            prepared[name]["raw_calibration_context"],
            selection_context_mean,
            selection_context_scale,
        )

    selection_trials = []
    selection_states = {}
    for weight in BOOST_WEIGHT_GRID:
        state, training = train_student(
            prepared["selection_train"]["features"],
            prepared["selection_train"]["calibration_context"],
            prepared["selection_train"]["base"],
            prepared["selection_train"]["labels"],
            boost_weight=weight,
            seed=4101,
        )
        metrics, _, _ = evaluate_model(
            state,
            prepared["selection_valid"]["features"],
            prepared["selection_valid"]["calibration_context"],
            prepared["selection_valid"]["base"],
            prepared["selection_valid"]["users"],
            prepared["selection_valid"]["labels"],
        )
        selection_states[weight] = state
        selection_trials.append({
            "boost_weight": weight,
            "metrics": metrics,
            "training": training,
        })
    selection_control = selection_trials[0]["metrics"]
    for trial in selection_trials:
        trial["delta_vs_weight_zero"] = metric_delta(trial["metrics"], selection_control)
        trial["eligible"] = (
            trial["delta_vs_weight_zero"]["gauc"] >= -1e-12
            and trial["delta_vs_weight_zero"]["ndcg5"] >= -1e-12
        )
    eligible_trials = [trial for trial in selection_trials if trial["eligible"]]
    selected_trial = max(
        eligible_trials,
        key=lambda trial: (trial["metrics"]["primary"], -trial["boost_weight"]),
    )
    selected_weight = float(selected_trial["boost_weight"])

    # Refit both arms on exactly the same April 12-14 rows, transformations,
    # seed, shuffling algorithm, epochs, and architecture.
    refit_feature_mean, refit_feature_scale = fit_standardizer(
        prepared["refit"]["raw_features"]
    )
    refit_context_mean, refit_context_scale = fit_standardizer(
        prepared["refit"]["raw_calibration_context"]
    )
    for name in ("refit", "screen"):
        prepared[name]["features"] = apply_standardizer(
            prepared[name]["raw_features"], refit_feature_mean, refit_feature_scale
        )
        prepared[name]["calibration_context"] = apply_standardizer(
            prepared[name]["raw_calibration_context"],
            refit_context_mean,
            refit_context_scale,
        )
    control_state, control_training = train_student(
        prepared["refit"]["features"],
        prepared["refit"]["calibration_context"],
        prepared["refit"]["base"],
        prepared["refit"]["labels"],
        boost_weight=0.0,
        seed=9127,
    )
    candidate_state, candidate_training = train_student(
        prepared["refit"]["features"],
        prepared["refit"]["calibration_context"],
        prepared["refit"]["base"],
        prepared["refit"]["labels"],
        boost_weight=selected_weight,
        seed=9127,
    )
    control_metrics, control_scores, control_probabilities = evaluate_model(
        control_state,
        prepared["screen"]["features"],
        prepared["screen"]["calibration_context"],
        prepared["screen"]["base"],
        prepared["screen"]["users"],
        prepared["screen"]["labels"],
    )
    candidate_metrics, candidate_scores, candidate_probabilities = evaluate_model(
        candidate_state,
        prepared["screen"]["features"],
        prepared["screen"]["calibration_context"],
        prepared["screen"]["base"],
        prepared["screen"]["users"],
        prepared["screen"]["labels"],
    )
    teacher_metrics = as_metrics(
        runner.evaluate_module.evaluate(
            prepared["screen"]["users"],
            prepared["screen"]["labels"],
            prepared["screen"]["base"],
        )
    )
    delta = metric_delta(candidate_metrics, control_metrics)
    passed = (
        delta["primary"] >= 0.0001
        and delta["gauc"] > 0.0
        and delta["ndcg5"] > 0.0
    )
    if selected_weight == 0.0:
        incidents.append(
            "The pre-screen selector preferred the exact zero-weight arm; the locked screen "
            "was still executed to preserve the predeclared protocol."
        )

    arrays_to_check = (
        prepared["screen"]["base"], control_scores, candidate_scores,
        control_probabilities, candidate_probabilities,
    )
    if not all(np.isfinite(values).all() for values in arrays_to_check):
        raise RuntimeError("Non-finite screen output detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        dates=np.asarray([row[0] for row in prepared["screen"]["rows"]], dtype=np.int32),
        users=np.asarray(prepared["screen"]["users"], dtype="U32"),
        labels=prepared["screen"]["labels"],
        teacher_raw=teacher[screen_mask].astype(np.float32),
        teacher_rank=prepared["screen"]["base"].astype(np.float32),
        control_scores=control_scores.astype(np.float32),
        candidate_scores=candidate_scores.astype(np.float32),
        control_calibrated_probabilities=control_probabilities.astype(np.float32),
        candidate_calibrated_probabilities=candidate_probabilities.astype(np.float32),
    )
    report = {
        "experiment": "SBCR-lite shuffled self-boosted calibrated ranking",
        "paper": {
            "title": "A Self-boosted Framework for Calibrated Ranking",
            "authors": "Shunyu Zhang et al.",
            "url": "https://arxiv.org/abs/2406.08010",
            "scope_note": (
                "Small CPU ablation, not a production reproduction: teacher score-distribution "
                "context, sample-separable boost loss, and stopped-gradient calibration are retained."
            ),
        },
        "status": "passed_train_only_screen" if passed else "rejected_at_train_only_screen",
        "protocol": {
            "teacher_fit_through": "2022-04-11",
            "weight_selection_train": "2022-04-12..2022-04-13",
            "weight_selection_validation": "2022-04-14",
            "matched_refit": "2022-04-12..2022-04-14",
            "locked_screen": "2022-04-15..2022-04-21",
            "april_22_plus_accessed": False,
            "hidden_test_accessed": False,
            "weight_grid_predeclared": list(BOOST_WEIGHT_GRID),
            "zero_preference": "ties on primary choose the smaller boost weight",
        },
        "data": {
            "teacher_cache": str(TEACHER_CACHE.relative_to(ROOT)),
            "teacher_sha256": file_sha256(TEACHER_CACHE),
            "teacher_rows": len(teacher),
            "selection_train_rows": len(split_rows["selection_train"]),
            "selection_validation_rows": len(split_rows["selection_valid"]),
            "refit_rows": len(split_rows["refit"]),
            "screen_rows": len(split_rows["screen"]),
            "screen_users": len(set(prepared["screen"]["users"])),
            "context_metadata": context_metadata,
            "outcome_derived_context_features": 0,
        },
        "selection": {
            "trials": selection_trials,
            "selected_boost_weight": selected_weight,
            "rule": (
                "Best primary among arms with nonnegative GAUC and nDCG@5 deltas versus "
                "weight zero; ties prefer the smaller weight."
            ),
        },
        "locked_screen": {
            "frozen_teacher_reference": teacher_metrics,
            "matched_control": control_metrics,
            "control_delta_vs_frozen_teacher": metric_delta(
                control_metrics, teacher_metrics
            ),
            "candidate": candidate_metrics,
            "delta": delta,
            "gate": "primary >= +0.0001 and GAUC > 0 and nDCG@5 > 0",
            "passed": passed,
            "control_training": control_training,
            "candidate_training": candidate_training,
        },
        "incidents": incidents,
        "confirmation": {
            "opened": False,
            "reason": (
                "Train-only gate passed; confirmation requires the separately aligned frozen-"
                "champion adapter."
                if passed else
                "Train-only gate failed, so April 22-28 remained sealed."
            ),
        },
        "recommendation": (
            "Proceed to one fixed April 22-28 confirmation without retuning."
            if passed else
            "Reject SBCR-lite and do not open April 22-28 for this branch."
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(args.scores_output.resolve().relative_to(ROOT)),
            "report": str(args.output.resolve().relative_to(ROOT)),
        },
        "resource_usage": tracker.finish(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_boost_weight": selected_weight,
        "frozen_teacher_reference": teacher_metrics,
        "control": control_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "incidents": incidents,
        "resource_usage": report["resource_usage"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
