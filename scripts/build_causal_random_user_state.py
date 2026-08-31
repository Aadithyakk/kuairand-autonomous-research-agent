#!/usr/bin/env python3
"""Build strictly causal random-panel user engagement-state features."""
from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_causal_streaming_random_features import DATA, LAGS_MS


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "causal_random_user_state_features.npz"
MANIFEST = HERE / "causal_random_user_state_manifest.json"


def load_inputs():
    random_rows = pd.read_csv(
        DATA / "log_random_4_22_to_5_08_pure.csv",
        usecols=[
            "user_id", "date", "time_ms", "duration_ms", "play_time_ms", "long_view",
        ],
        dtype={"user_id": "string"},
    )
    random_rows = random_rows.loc[random_rows["date"] <= 20220428].copy()
    random_rows["watch_ratio"] = np.clip(
        random_rows["play_time_ms"].to_numpy(dtype=np.float64)
        / np.maximum(random_rows["duration_ms"].to_numpy(dtype=np.float64), 1.0),
        0.0, 2.0,
    )
    standard = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "time_ms"],
        dtype={"user_id": "string"},
    )
    standard = standard.loc[standard["date"] <= 20220428].reset_index(drop=True)
    standard["row_index"] = np.arange(len(standard), dtype=np.int64)
    train = pd.read_csv(
        DATA / "log_standard_4_08_to_4_21_pure.csv",
        usecols=["long_view", "duration_ms", "play_time_ms"],
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


def build_features(random_rows, standard, priors):
    random_values = list(
        random_rows.sort_values("time_ms", kind="stable")
        [["time_ms", "user_id", "long_view", "watch_ratio"]]
        .itertuples(index=False, name=None)
    )
    query_values = list(
        standard.sort_values("time_ms", kind="stable")
        [["time_ms", "row_index", "user_id"]]
        .itertuples(index=False, name=None)
    )
    names = (
        "cum_long", "cum_watch", "last_long", "last_watch",
        "last3_long", "last3_watch", "last10_long", "last10_watch",
        "log_count", "recency_log_minutes",
    )
    features = {
        f"user_state_{name}_{lag_name}": np.zeros(len(standard), dtype=np.float32)
        for lag_name in LAGS_MS for name in names
    }
    for lag_name, lag_ms in LAGS_MS.items():
        totals = defaultdict(lambda: [0.0, 0.0, 0])
        recent = defaultdict(lambda: deque(maxlen=10))
        last_time = {}
        random_index = 0
        for query_number, query in enumerate(query_values):
            cutoff = int(query[0]) - lag_ms
            while random_index < len(random_values) and int(random_values[random_index][0]) <= cutoff:
                timestamp, user, long_value, watch_value = random_values[random_index]
                state = totals[user]
                state[0] += float(long_value)
                state[1] += float(watch_value)
                state[2] += 1
                recent[user].append((float(long_value), float(watch_value)))
                last_time[user] = int(timestamp)
                random_index += 1

            output_index = int(query[1])
            user = query[2]
            state = totals.get(user)
            if state is None:
                continue
            long_sum, watch_sum, count = state
            history = recent[user]
            last = history[-1]
            last3 = list(history)[-3:]
            last10 = list(history)
            values = {
                "cum_long": (long_sum + 8.0 * priors["long"]) / (count + 8.0) - priors["long"],
                "cum_watch": (watch_sum + 8.0 * priors["watch"]) / (count + 8.0) - priors["watch"],
                "last_long": last[0] - priors["long"],
                "last_watch": last[1] - priors["watch"],
                "last3_long": float(np.mean([item[0] for item in last3])) - priors["long"],
                "last3_watch": float(np.mean([item[1] for item in last3])) - priors["watch"],
                "last10_long": float(np.mean([item[0] for item in last10])) - priors["long"],
                "last10_watch": float(np.mean([item[1] for item in last10])) - priors["watch"],
                "log_count": float(np.log1p(count)),
                "recency_log_minutes": float(np.log1p(max(cutoff - last_time[user], 0) / 60_000.0)),
            }
            for name, value in values.items():
                features[f"user_state_{name}_{lag_name}"][output_index] = value
            if query_number and query_number % 30_000 == 0:
                print(json.dumps({
                    "lag": lag_name,
                    "query_rows": query_number,
                    "random_history_rows": random_index,
                }), flush=True)
    return features


def main() -> None:
    random_rows, standard, priors = load_inputs()
    features = build_features(random_rows, standard, priors)
    np.savez_compressed(CACHE, **features)
    manifest = {
        "experiment": "causal random-panel overall user engagement state",
        "lags_ms": LAGS_MS,
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
