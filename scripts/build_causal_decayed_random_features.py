#!/usr/bin/env python3
"""Build causal exponentially decayed random-panel preference features."""
from __future__ import annotations

import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_causal_streaming_random_features import DATA, row_key


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "causal_decayed_random_features.npz"
MANIFEST = HERE / "causal_decayed_random_feature_manifest.json"
LAG_MS = 5 * 60 * 1000
HALF_LIVES_MS = {
    "1h": 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "24h": 24 * 60 * 60 * 1000,
    "72h": 72 * 60 * 60 * 1000,
}
FAMILIES = {
    "type": (("video_type",), 20.0),
    "duration": (("duration_bucket",), 15.0),
    "user_author": (("user_id", "author_id"), 2.0),
    "user_music": (("user_id", "music_id"), 2.0),
    "user_tag": (("user_id", "tag_primary"), 3.0),
    "user_type": (("user_id", "video_type"), 3.0),
    "user_duration": (("user_id", "duration_bucket"), 3.0),
}


def load_inputs():
    metadata = pd.read_csv(
        DATA / "video_features_basic_pure.csv",
        usecols=["video_id", "author_id", "music_id", "tag", "video_type"],
        dtype={
            "video_id": "string", "author_id": "string", "music_id": "string",
            "tag": "string", "video_type": "string",
        },
    ).drop_duplicates("video_id", keep="last")
    metadata["tag_primary"] = metadata["tag"].fillna("UNK").astype(str).str.split(",").str[0]
    metadata = metadata.drop(columns=["tag"])

    random_rows = pd.read_csv(
        DATA / "log_random_4_22_to_5_08_pure.csv",
        usecols=[
            "user_id", "video_id", "date", "time_ms", "duration_ms",
            "play_time_ms", "long_view",
        ],
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
        usecols=["long_view", "play_time_ms", "duration_ms"],
    )
    priors = {
        "long": float(train["long_view"].mean()),
        "watch": float(np.clip(
            train["play_time_ms"].to_numpy(dtype=np.float64)
            / np.maximum(train["duration_ms"].to_numpy(dtype=np.float64), 1.0),
            0.0, 2.0,
        ).mean()),
    }
    return random_rows, standard, priors


def decay(state, timestamp: int, scale: float):
    long_sum, watch_sum, count, last_time = state
    factor = math.exp(-(timestamp - last_time) * scale) if timestamp > last_time else 1.0
    return long_sum * factor, watch_sum * factor, count * factor, timestamp


def build_features(random_rows, standard, priors):
    key_columns = sorted({column for keys, _ in FAMILIES.values() for column in keys})
    random_columns = ["time_ms", "long_view", "watch_ratio", *key_columns]
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
        f"decay_{outcome}_{family}_{half_life}": np.zeros(len(standard), dtype=np.float32)
        for half_life in HALF_LIVES_MS for family in FAMILIES for outcome in ("long", "watch")
    }

    for half_name, half_life_ms in HALF_LIVES_MS.items():
        scale = math.log(2.0) / half_life_ms
        stats = {family: {} for family in FAMILIES}
        first_time = int(random_values[0][0]) if random_values else 0
        global_state = (0.0, 0.0, 0.0, first_time)
        random_index = 0
        for query_number, query in enumerate(query_values):
            cutoff = int(query[0]) - LAG_MS
            while random_index < len(random_values) and int(random_values[random_index][0]) <= cutoff:
                event = random_values[random_index]
                timestamp = int(event[0])
                global_state = decay(global_state, timestamp, scale)
                global_state = (
                    global_state[0] + float(event[1]),
                    global_state[1] + float(event[2]),
                    global_state[2] + 1.0,
                    timestamp,
                )
                for family, (columns, _) in FAMILIES.items():
                    key = row_key(event, tuple(random_positions[column] for column in columns))
                    state = stats[family].get(key, (0.0, 0.0, 0.0, timestamp))
                    state = decay(state, timestamp, scale)
                    stats[family][key] = (
                        state[0] + float(event[1]), state[1] + float(event[2]),
                        state[2] + 1.0, timestamp,
                    )
                random_index += 1

            current_global = decay(global_state, cutoff, scale)
            prior_long = (current_global[0] + 200.0 * priors["long"]) / (current_global[2] + 200.0)
            prior_watch = (current_global[1] + 200.0 * priors["watch"]) / (current_global[2] + 200.0)
            output_index = int(query[1])
            for family, (columns, smoothing) in FAMILIES.items():
                key = row_key(query, tuple(query_positions[column] for column in columns))
                state = stats[family].get(key)
                if state is None:
                    continue
                current = decay(state, cutoff, scale)
                features[f"decay_long_{family}_{half_name}"][output_index] = (
                    (current[0] + smoothing * prior_long) / (current[2] + smoothing) - prior_long
                )
                features[f"decay_watch_{family}_{half_name}"][output_index] = (
                    (current[1] + smoothing * prior_watch) / (current[2] + smoothing) - prior_watch
                )
            if query_number and query_number % 30_000 == 0:
                print(json.dumps({
                    "half_life": half_name,
                    "query_rows": query_number,
                    "random_history_rows": random_index,
                }), flush=True)
    return features


def main() -> None:
    random_rows, standard, priors = load_inputs()
    features = build_features(random_rows, standard, priors)
    np.savez_compressed(CACHE, **features)
    manifest = {
        "experiment": "causal exponentially decayed random-panel preferences",
        "lag_ms": LAG_MS,
        "half_lives_ms": HALF_LIVES_MS,
        "outcomes": ["long_view", "clipped_watch_ratio"],
        "families": list(FAMILIES),
        "feature_count": len(features),
        "validation_rows": len(standard),
        "uses_only_random_events_before_query_minus_lag": True,
        "uses_standard_validation_outcomes": False,
        "hidden_test_accessed": False,
        "cache": str(CACHE),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
