#!/usr/bin/env python3
"""Completion-safe block-retrained CatBoost pointwise and ranking models."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
CHAMPION = Path(os.environ.get(
    "KUAI_CATBOOST_CHAMPION",
    str(HERE / "causal_streaming_user_duration_expanded_scores.npz"),
))
OUTPUT = Path(os.environ.get(
    "KUAI_CATBOOST_OUTPUT", str(HERE / "prequential_block_catboost_scores.npz")
))
REPORT = Path(os.environ.get(
    "KUAI_CATBOOST_REPORT", str(HERE / "prequential_block_catboost_results.json")
))
BLOCK_HOURS = int(os.environ.get("KUAI_CATBOOST_BLOCK_HOURS", "24"))
ITERATIONS = int(os.environ.get("KUAI_CATBOOST_ITERATIONS", "100"))
DEPTH = int(os.environ.get("KUAI_CATBOOST_DEPTH", "6"))
MODEL_TYPES = tuple(
    value.strip() for value in os.environ.get(
        "KUAI_CATBOOST_MODELS", "classifier,ranker"
    ).split(",") if value.strip()
)

sys.path.insert(0, str(SCRIPT_DIR))
from joint_terminal_gate_search import factorize, rank_ordinal  # noqa: E402
from prequential_daily_pairwise_logistic import load_matrix  # noqa: E402


def balanced_weights(user_codes: np.ndarray) -> np.ndarray:
    counts = np.bincount(user_codes)
    weights = 1.0 / np.maximum(counts[user_codes], 1)
    return (weights / weights.mean()).astype(np.float32)


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
    availability = times + np.maximum(
        rows["play_time_ms"].to_numpy(dtype=np.int64), 0
    )
    labels = rows["long_view"].to_numpy(dtype=np.int8)
    codes, _, counts = factorize(users)
    with np.load(CHAMPION) as archive:
        champion = np.asarray(archive["selected"], dtype=np.float64)
    champion_rank = rank_ordinal(champion, codes, counts).astype(np.float32)
    matrix, feature_names = load_matrix(codes, counts)
    matrix = np.column_stack([champion_rank, matrix]).astype(np.float32)

    block_ms = BLOCK_HOURS * 3_600_000
    units = times // block_ms
    unique_units = np.unique(units)
    outputs = {name: champion_rank.astype(np.float64).copy() for name in MODEL_TYPES}
    fit_log: list[dict] = []
    common = dict(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=0.05,
        l2_leaf_reg=20.0,
        random_strength=1.0,
        random_seed=int(os.environ.get("KUAI_CATBOOST_SEED", "2026")),
        thread_count=int(os.environ.get("KUAI_CATBOOST_THREADS", "6")),
        verbose=False,
        allow_writing_files=False,
    )
    for current_unit in unique_units[1:]:
        block_start = int(current_unit) * block_ms
        train_index = np.flatnonzero(availability < block_start)
        test_mask = units == current_unit
        if len(train_index) < 1000 or test_mask.sum() == 0:
            continue

        if "classifier" in outputs:
            model = CatBoostClassifier(loss_function="Logloss", **common)
            model.fit(
                matrix[train_index], labels[train_index],
                sample_weight=balanced_weights(codes[train_index]),
            )
            outputs["classifier"][test_mask] = model.predict(
                matrix[test_mask], prediction_type="RawFormulaVal"
            )

        if "ranker" in outputs:
            order = train_index[np.argsort(codes[train_index], kind="stable")]
            model = CatBoostRanker(
                loss_function=os.environ.get(
                    "KUAI_CATBOOST_RANK_LOSS", "YetiRankPairwise"
                ),
                **common,
            )
            pool = Pool(matrix[order], labels[order], group_id=codes[order])
            model.fit(pool)
            outputs["ranker"][test_mask] = model.predict(matrix[test_mask])

        record = {
            "test_unit": int(current_unit),
            "test_date": int(dates[np.flatnonzero(test_mask)[0]]),
            "train_rows": int(len(train_index)),
            "test_rows": int(test_mask.sum()),
        }
        fit_log.append(record)
        print(json.dumps(record), flush=True)

    np.savez_compressed(
        OUTPUT,
        champion=champion.astype(np.float32),
        **{name: score.astype(np.float32) for name, score in outputs.items()},
    )
    report = {
        "experiment": "completion-safe block-retrained CatBoost",
        "evaluation_mode": "online/prequential; not a static locked model",
        "models": MODEL_TYPES,
        "block_hours": BLOCK_HOURS,
        "iterations": ITERATIONS,
        "depth": DEPTH,
        "feature_count": int(matrix.shape[1]),
        "feature_names": ["champion_rank", *feature_names],
        "fit_log": fit_log,
        "outcome_available_at": "time_ms + play_time_ms",
        "hidden_test_accessed": False,
        "artifacts": {"scores": str(OUTPUT), "report": str(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
