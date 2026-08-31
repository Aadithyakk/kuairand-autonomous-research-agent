#!/usr/bin/env python3
"""Build causal random-panel watch-progress features for standard-feed ranking."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_causal_streaming_random_features import (
    DATA,
    FAMILIES,
    LAGS_MS,
    row_key,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "causal_random_watch_features.npz"
MANIFEST = HERE / "causal_random_watch_feature_manifest.json"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    metadata = pd.read_csv(
        DATA / "video_features_basic_pure.csv",
        usecols=["video_id", "author_id", "music_id", "tag", "video_type"],
        dtype={
            "video_id": "string", "author_id": "string", "music_id": "string",
            "tag": "string", "video_type": "string",
        },
    ).drop_duplicates("video_id", keep="last")
    metadata["tag_primary"] = (
        metadata["tag"].fillna("UNK").astype(str).str.split(",").str[0]
    )
    metadata = metadata.drop(columns=["tag"])

    random_rows = pd.read_csv(
        DATA / "log_random_4_22_to_5_08_pure.csv",
        usecols=["user_id", "video_id", "date", "time_ms", "duration_ms", "play_time_ms"],
        dtype={"user_id": "string", "video_id": "string"},
    )
    random_rows = random_rows.loc[random_rows["date"] <= 20220428].merge(
        metadata, how="left", on="video_id", sort=False,
    )
    random_rows["watch_ratio"] = np.clip(
        random_rows["play_time_ms"].to_numpy(dtype=np.float64)
        / np.maximum(random_rows["duration_ms"].to_numpy(dtype=np.float64), 1.0),
        0.0, 2.0,
    )

    standard = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "video_id", "date", "time_ms", "duration_ms"],
        dtype={"user_id": "string", "video_id": "string"},
    )
    standard = standard.loc[standard["date"] <= 20220428].reset_index(drop=True)
    standard["row_index"] = np.arange(len(standard), dtype=np.int64)
    standard = standard.merge(metadata, how="left", on="video_id", sort=False)

    for frame in (random_rows, standard):
        for column in ("author_id", "music_id", "tag_primary", "video_type"):
            frame[column] = frame[column].fillna("UNK").astype(str)
        frame["duration_bucket"] = (
            frame["duration_ms"].astype(np.int64) // 10_000
        ).clip(0, 60).astype(str)
        frame["user_id"] = frame["user_id"].astype(str)
        frame["video_id"] = frame["video_id"].astype(str)

    train = pd.read_csv(
        DATA / "log_standard_4_08_to_4_21_pure.csv",
        usecols=["play_time_ms", "duration_ms"],
    )
    training_prior = float(np.clip(
        train["play_time_ms"].to_numpy(dtype=np.float64)
        / np.maximum(train["duration_ms"].to_numpy(dtype=np.float64), 1.0),
        0.0, 2.0,
    ).mean())
    return random_rows, standard, training_prior


def build_features(
    random_rows: pd.DataFrame,
    standard: pd.DataFrame,
    training_prior: float,
) -> dict[str, np.ndarray]:
    key_columns = sorted({column for keys, _ in FAMILIES.values() for column in keys})
    random_columns = ["time_ms", "watch_ratio", *key_columns]
    query_columns = ["time_ms", "row_index", *key_columns]
    random_values = list(
        random_rows.sort_values("time_ms", kind="stable")[random_columns]
        .itertuples(index=False, name=None)
    )
    query_values = list(
        standard.sort_values("time_ms", kind="stable")[query_columns]
        .itertuples(index=False, name=None)
    )
    random_positions = {name: random_columns.index(name) for name in key_columns}
    query_positions = {name: query_columns.index(name) for name in key_columns}
    features = {
        f"watch_{family}_{lag_name}": np.zeros(len(standard), dtype=np.float32)
        for lag_name in LAGS_MS for family in FAMILIES
    }

    for lag_name, lag_ms in LAGS_MS.items():
        stats = {family: {} for family in FAMILIES}
        global_sum = 0.0
        global_count = 0
        random_index = 0
        for query_number, query in enumerate(query_values):
            cutoff = int(query[0]) - lag_ms
            while random_index < len(random_values) and int(random_values[random_index][0]) <= cutoff:
                event = random_values[random_index]
                outcome = float(event[1])
                global_sum += outcome
                global_count += 1
                for family, (columns, _) in FAMILIES.items():
                    key = row_key(event, tuple(random_positions[column] for column in columns))
                    total, count = stats[family].get(key, (0.0, 0))
                    stats[family][key] = (total + outcome, count + 1)
                random_index += 1

            prior = (global_sum + 200.0 * training_prior) / (global_count + 200.0)
            output_index = int(query[1])
            for family, (columns, smoothing) in FAMILIES.items():
                key = row_key(query, tuple(query_positions[column] for column in columns))
                total, count = stats[family].get(key, (0.0, 0))
                features[f"watch_{family}_{lag_name}"][output_index] = (
                    (total + smoothing * prior) / (count + smoothing) - prior
                )
            if query_number and query_number % 30_000 == 0:
                print(json.dumps({
                    "lag": lag_name,
                    "query_rows": query_number,
                    "random_history_rows": random_index,
                }), flush=True)
    return features


def main() -> None:
    random_rows, standard, training_prior = load_inputs()
    features = build_features(random_rows, standard, training_prior)
    np.savez_compressed(CACHE, **features)
    manifest = {
        "experiment": "strictly causal random-panel watch-progress features",
        "outcome": "clip(play_time_ms / duration_ms, 0, 2)",
        "random_rows_available_through_20220428": len(random_rows),
        "validation_rows": len(standard),
        "feature_count": len(features),
        "families": list(FAMILIES),
        "lags_ms": LAGS_MS,
        "uses_only_random_events_before_query_minus_lag": True,
        "uses_standard_validation_outcomes": False,
        "hidden_test_accessed": False,
        "cache": str(CACHE),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
