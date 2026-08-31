#!/usr/bin/env python3
"""Daily causal pairwise logistic reranker on champion-near user pairs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
CHAMPION = Path(os.environ.get(
    "KUAI_PAIRWISE_CHAMPION",
    str(HERE / "causal_streaming_user_duration_expanded_scores.npz"),
))
OUTPUT = Path(os.environ.get(
    "KUAI_PAIRWISE_OUTPUT",
    str(HERE / "prequential_daily_pairwise_logistic_scores.npz"),
))
REPORT = Path(os.environ.get(
    "KUAI_PAIRWISE_REPORT",
    str(HERE / "prequential_daily_pairwise_logistic_results.json"),
))
CACHES = (
    HERE / "prequential_standard_feedback_features.npz",
    HERE / "causal_streaming_random_features.npz",
    HERE / "causal_random_watch_features.npz",
    HERE / "causal_random_user_state_features.npz",
    HERE / "causal_decayed_random_features.npz",
    HERE / "causal_random_action_state_features.npz",
    HERE / "causal_random_transition_features.npz",
)
ALPHAS = tuple(float(value) for value in os.environ.get(
    "KUAI_PAIRWISE_ALPHAS", "0.001,0.0003"
).split(",") if value.strip())
NEGATIVES_PER_POSITIVE = int(os.environ.get("KUAI_PAIRWISE_NEGATIVES", "2"))
PAIR_WEIGHT_MODE = os.environ.get("KUAI_PAIRWISE_WEIGHT_MODE", "none")
HISTORY_DAYS = int(os.environ.get("KUAI_PAIRWISE_HISTORY_DAYS", "0"))
BLOCK_HOURS = int(os.environ.get("KUAI_PAIRWISE_BLOCK_HOURS", "0"))

sys.path.insert(0, str(SCRIPT_DIR))
from joint_terminal_gate_search import factorize, rank_ordinal  # noqa: E402
from prequential_causality import completion_safe_training_mask  # noqa: E402


def load_matrix(codes: np.ndarray, counts: np.ndarray):
    columns = []
    names = []
    for cache in CACHES:
        with np.load(cache) as archive:
            for name in archive.files:
                columns.append(np.nan_to_num(np.asarray(archive[name], dtype=np.float32)))
                names.append(f"{cache.stem}:{name}")
    if os.environ.get("KUAI_PAIRWISE_IMMEDIATE") == "1":
        cache = HERE / "prequential_immediate_feedback_features.npz"
        with np.load(cache) as archive:
            for name in archive.files:
                columns.append(np.nan_to_num(np.asarray(archive[name], dtype=np.float32)))
                names.append(f"{cache.stem}:{name}")
    if os.environ.get("KUAI_PAIRWISE_STATIC_MODELS") == "1":
        from prequential_daily_static_stack_logistic import RUNTIME, STATIC_MODELS

        for model_name in STATIC_MODELS:
            with np.load(RUNTIME / model_name) as archive:
                values = np.asarray(archive["scores"], dtype=np.float64)
            columns.append(rank_ordinal(values, codes, counts).astype(np.float32))
            names.append(f"static_rank:{model_name}")
    return np.column_stack(columns).astype(np.float32), names


def build_pairs(
    indices: np.ndarray,
    users: np.ndarray,
    labels: np.ndarray,
    champion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"index": indices, "user": users[indices]})
    left = []
    right = []
    pair_weights = []
    for group in frame.groupby("user", sort=False)["index"]:
        group_indices = group[1].to_numpy(dtype=np.int64)
        positives = group_indices[labels[group_indices] > 0]
        negatives = group_indices[labels[group_indices] <= 0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        user_pair_count = len(positives) * min(NEGATIVES_PER_POSITIVE, len(negatives))
        for positive in positives:
            distance = np.abs(champion[negatives] - champion[positive])
            selected_negatives = negatives[
                np.argsort(distance, kind="stable")[:NEGATIVES_PER_POSITIVE]
            ]
            left.extend([positive] * len(selected_negatives))
            right.extend(selected_negatives.tolist())
            for negative in selected_negatives:
                weight = 1.0
                if "user" in PAIR_WEIGHT_MODE:
                    exponent = 1.0 if "full" in PAIR_WEIGHT_MODE else 0.5
                    weight /= max(user_pair_count, 1) ** exponent
                if "lambda" in PAIR_WEIGHT_MODE:
                    group_size = max(len(group_indices), 1)
                    positive_position = group_size - champion[positive]
                    negative_position = group_size - champion[negative]
                    positive_discount = 1.0 / np.log2(max(positive_position, 1.0) + 1.0)
                    negative_discount = 1.0 / np.log2(max(negative_position, 1.0) + 1.0)
                    weight *= 0.05 + abs(positive_discount - negative_discount)
                pair_weights.append(weight)
    weights = np.asarray(pair_weights, dtype=np.float64)
    if len(weights):
        weights /= weights.mean()
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        weights,
    )


def main() -> None:
    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "time_ms", "play_time_ms", "long_view"],
        dtype={"user_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    users = rows["user_id"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype=np.int64)
    times = rows["time_ms"].to_numpy(dtype=np.int64)
    play_times = rows["play_time_ms"].to_numpy(dtype=np.int64)
    labels = rows["long_view"].to_numpy(dtype=np.int8)
    with np.load(CHAMPION) as archive:
        champion = np.asarray(archive["selected"], dtype=np.float64)
    codes, _, counts = factorize(users)
    champion_rank = rank_ordinal(champion, codes, counts).astype(np.float32)
    matrix, feature_names = load_matrix(codes, counts)
    matrix = np.column_stack([champion_rank, matrix]).astype(np.float32)

    outputs = {}
    fit_log = []
    if BLOCK_HOURS > 0:
        block_ms = BLOCK_HOURS * 60 * 60 * 1000
        units = times // block_ms
    else:
        block_ms = 0
        units = dates
    unique_units = np.unique(units)
    for alpha in ALPHAS:
        score = champion_rank.astype(np.float64).copy()
        for current_unit in unique_units[1:]:
            if BLOCK_HOURS > 0:
                block_start = int(current_unit) * block_ms
                train_mask = completion_safe_training_mask(
                    times, play_times, block_start
                )
                if HISTORY_DAYS > 0:
                    train_mask &= times >= block_start - HISTORY_DAYS * 86_400_000
            else:
                train_mask = dates < current_unit
                if HISTORY_DAYS > 0:
                    train_mask &= dates >= current_unit - HISTORY_DAYS
            train_indices = np.flatnonzero(train_mask)
            test_mask = units == current_unit
            left, right, pair_weights = build_pairs(
                train_indices, users, labels, champion_rank
            )
            if len(left) < 100:
                continue
            positive_difference = matrix[left] - matrix[right]
            pair_matrix = np.vstack([positive_difference, -positive_difference])
            pair_labels = np.r_[
                np.ones(len(positive_difference), dtype=np.int8),
                np.zeros(len(positive_difference), dtype=np.int8),
            ]
            model = make_pipeline(
                StandardScaler(with_mean=False),
                SGDClassifier(
                    loss="log_loss", penalty="l2", alpha=alpha,
                    fit_intercept=False, max_iter=100, tol=1e-4,
                    average=True, random_state=2026,
                ),
            )
            model.fit(
                pair_matrix, pair_labels,
                sgdclassifier__sample_weight=np.r_[pair_weights, pair_weights],
            )
            score[test_mask] = model.decision_function(matrix[test_mask])
            record = {
                "alpha": alpha,
                "test_unit": int(current_unit),
                "test_date": int(dates[np.flatnonzero(test_mask)[0]]),
                "train_rows": len(train_indices), "oriented_pairs": len(pair_labels),
                "test_rows": int(test_mask.sum()),
            }
            fit_log.append(record)
            print(json.dumps(record), flush=True)
        outputs[f"alpha_{alpha:g}"] = score

    np.savez_compressed(
        OUTPUT,
        champion=champion.astype(np.float32),
        **{name: value.astype(np.float32) for name, value in outputs.items()},
    )
    report = {
        "experiment": "daily causal pairwise logistic boundary reranker",
        "evaluation_mode": "online/prequential; not comparable to a static locked model",
        "feature_count": matrix.shape[1],
        "feature_names": ["champion_rank", *feature_names],
        "fit_log": fit_log,
        "pair_sampling": (
            f"up to {NEGATIVES_PER_POSITIVE} champion-nearest negatives per positive "
            "within user"
        ),
        "static_models_included": os.environ.get("KUAI_PAIRWISE_STATIC_MODELS") == "1",
        "immediate_feedback_included": os.environ.get("KUAI_PAIRWISE_IMMEDIATE") == "1",
        "pair_weight_mode": PAIR_WEIGHT_MODE,
        "history_days": HISTORY_DAYS or "all_prior_dates",
        "retrain_block_hours": BLOCK_HOURS or "daily",
        "uses_only_prior_dates_for_each_daily_model": True,
        "hidden_test_accessed": False,
        "artifacts": {"scores": str(OUTPUT), "report": str(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
