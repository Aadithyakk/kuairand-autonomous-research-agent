#!/usr/bin/env python3
"""Build causal last-random-event transition features for standard candidates."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_causal_streaming_random_features import DATA


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "causal_random_transition_features.npz"
MANIFEST = HERE / "causal_random_transition_manifest.json"
LAGS_MS = {"5m": 5 * 60 * 1000, "1h": 60 * 60 * 1000}
KEYS = ("video_id", "author_id", "music_id", "tag_primary", "video_type", "duration_bucket")


def load_inputs():
    metadata = pd.read_csv(
        DATA / "video_features_basic_pure.csv",
        usecols=["video_id", "author_id", "music_id", "tag", "video_type"],
        dtype={column: "string" for column in ("video_id", "author_id", "music_id", "tag", "video_type")},
    ).drop_duplicates("video_id", keep="last")
    metadata["tag_primary"] = metadata["tag"].fillna("UNK").astype(str).str.split(",").str[0]
    metadata = metadata.drop(columns=["tag"])
    random_rows = pd.read_csv(
        DATA / "log_random_4_22_to_5_08_pure.csv",
        usecols=["user_id", "video_id", "date", "time_ms", "duration_ms", "play_time_ms", "long_view"],
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
        for column in KEYS[:-1]:
            frame[column] = frame[column].fillna("UNK").astype(str)
        frame["duration_bucket"] = (
            frame["duration_ms"].astype(np.int64) // 10_000
        ).clip(0, 60).astype(str)
        frame["user_id"] = frame["user_id"].astype(str)
    return random_rows, standard


def main() -> None:
    random_rows, standard = load_inputs()
    random_columns = ["time_ms", "user_id", "long_view", "watch_ratio", *KEYS]
    query_columns = ["time_ms", "row_index", "user_id", *KEYS]
    random_values = list(
        random_rows.sort_values("time_ms", kind="stable")[random_columns]
        .itertuples(index=False, name=None)
    )
    query_values = list(
        standard.sort_values("time_ms", kind="stable")[query_columns]
        .itertuples(index=False, name=None)
    )
    features = {
        f"transition_{key}_{stat}_{lag_name}": np.zeros(len(standard), dtype=np.float32)
        for lag_name in LAGS_MS for key in KEYS for stat in ("same", "long", "watch")
    }
    for lag_name, lag_ms in LAGS_MS.items():
        last_event = {}
        random_index = 0
        for query_number, query in enumerate(query_values):
            cutoff = int(query[0]) - lag_ms
            while random_index < len(random_values) and int(random_values[random_index][0]) <= cutoff:
                event = random_values[random_index]
                last_event[event[1]] = event
                random_index += 1
            event = last_event.get(query[2])
            if event is not None:
                output_index = int(query[1])
                for key_index, key in enumerate(KEYS):
                    same = float(event[4 + key_index] == query[3 + key_index])
                    features[f"transition_{key}_same_{lag_name}"][output_index] = same
                    features[f"transition_{key}_long_{lag_name}"][output_index] = same * float(event[2])
                    features[f"transition_{key}_watch_{lag_name}"][output_index] = same * float(event[3])
            if query_number and query_number % 30_000 == 0:
                print(json.dumps({
                    "lag": lag_name, "query_rows": query_number,
                    "random_history_rows": random_index,
                }), flush=True)
    np.savez_compressed(CACHE, **features)
    manifest = {
        "experiment": "causal last-random-event transition matches",
        "keys": list(KEYS), "lags_ms": LAGS_MS,
        "feature_count": len(features), "validation_rows": len(standard),
        "uses_only_random_events_before_query_minus_lag": True,
        "uses_standard_validation_outcomes": False,
        "hidden_test_accessed": False,
        "cache": str(CACHE),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
