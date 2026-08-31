#!/usr/bin/env python3
"""Build causal recent random-panel action-state features per user."""
from __future__ import annotations

from collections import defaultdict, deque
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
CACHE = HERE / "causal_random_action_state_features.npz"
MANIFEST = HERE / "causal_random_action_state_manifest.json"
LAGS_MS = {"5m": 5 * 60 * 1000, "1h": 60 * 60 * 1000}
ACTIONS = (
    "is_click", "is_like", "is_follow", "is_comment", "is_forward",
    "is_hate", "is_profile_enter",
)


def main() -> None:
    random_rows = pd.read_csv(
        DATA / "log_random_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "time_ms", *ACTIONS],
        dtype={"user_id": "string"},
    )
    random_rows = random_rows.loc[random_rows["date"] <= 20220428]
    standard = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "time_ms"],
        dtype={"user_id": "string"},
    )
    standard = standard.loc[standard["date"] <= 20220428].reset_index(drop=True)
    standard["row_index"] = np.arange(len(standard), dtype=np.int64)
    random_values = list(
        random_rows.sort_values("time_ms", kind="stable")
        [["time_ms", "user_id", *ACTIONS]].itertuples(index=False, name=None)
    )
    query_values = list(
        standard.sort_values("time_ms", kind="stable")
        [["time_ms", "row_index", "user_id"]].itertuples(index=False, name=None)
    )
    features = {
        f"action_{action}_{stat}_{lag_name}": np.zeros(len(standard), dtype=np.float32)
        for lag_name in LAGS_MS for action in ACTIONS for stat in ("last", "recent3", "recent10")
    }
    for lag_name, lag_ms in LAGS_MS.items():
        recent = defaultdict(lambda: deque(maxlen=10))
        random_index = 0
        for query_number, query in enumerate(query_values):
            cutoff = int(query[0]) - lag_ms
            while random_index < len(random_values) and int(random_values[random_index][0]) <= cutoff:
                event = random_values[random_index]
                recent[event[1]].append(tuple(float(value) for value in event[2:]))
                random_index += 1
            output_index = int(query[1])
            history = recent.get(query[2])
            if history:
                values = np.asarray(history, dtype=np.float64)
                for action_index, action in enumerate(ACTIONS):
                    features[f"action_{action}_last_{lag_name}"][output_index] = values[-1, action_index]
                    features[f"action_{action}_recent3_{lag_name}"][output_index] = values[-3:, action_index].mean()
                    features[f"action_{action}_recent10_{lag_name}"][output_index] = values[:, action_index].mean()
            if query_number and query_number % 30_000 == 0:
                print(json.dumps({
                    "lag": lag_name, "query_rows": query_number,
                    "random_history_rows": random_index,
                }), flush=True)
    np.savez_compressed(CACHE, **features)
    manifest = {
        "experiment": "causal recent random-panel user action state",
        "actions": list(ACTIONS),
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
