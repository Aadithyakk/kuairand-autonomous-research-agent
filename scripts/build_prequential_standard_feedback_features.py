#!/usr/bin/env python3
"""Build causal online-feedback features within the standard-feed stream."""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
CACHE = HERE / "prequential_standard_feedback_features.npz"
MANIFEST = HERE / "prequential_standard_feedback_manifest.json"
LAGS_MS = {
    "5m": 5 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "6h": 6 * 60 * 60 * 1000,
    "24h": 24 * 60 * 60 * 1000,
}
FAMILIES = {
    "user": (("user_id",), 20.0),
    "video": (("video_id",), 20.0),
    "author": (("author_id",), 40.0),
    "music": (("music_id",), 50.0),
    "tag": (("tag_primary",), 80.0),
    "type": (("video_type",), 200.0),
    "duration": (("duration_bucket",), 150.0),
    "author_duration": (("author_id", "duration_bucket"), 24.0),
    "user_video": (("user_id", "video_id"), 4.0),
    "user_author": (("user_id", "author_id"), 8.0),
    "user_music": (("user_id", "music_id"), 8.0),
    "user_tag": (("user_id", "tag_primary"), 12.0),
    "user_type": (("user_id", "video_type"), 12.0),
    "user_duration": (("user_id", "duration_bucket"), 12.0),
}
RECENT_K = (1, 3, 5, 10)


def load_rows() -> tuple[pd.DataFrame, float]:
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
    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=[
            "user_id", "video_id", "date", "time_ms", "duration_ms", "long_view",
        ],
        dtype={"user_id": "string", "video_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    rows["row_index"] = np.arange(len(rows), dtype=np.int64)
    rows = rows.merge(metadata, how="left", on="video_id", sort=False)
    for column in ("author_id", "music_id", "tag_primary", "video_type"):
        rows[column] = rows[column].fillna("UNK").astype(str)
    rows["duration_bucket"] = (
        rows["duration_ms"].astype(np.int64) // 10_000
    ).clip(0, 60).astype(str)
    rows["user_id"] = rows["user_id"].astype(str)
    rows["video_id"] = rows["video_id"].astype(str)
    prior = float(pd.read_csv(
        DATA / "log_standard_4_08_to_4_21_pure.csv", usecols=["long_view"],
    )["long_view"].mean())
    return rows, prior


def make_key(event: tuple, positions: tuple[int, ...]):
    if len(positions) == 1:
        return event[positions[0]]
    return tuple(event[position] for position in positions)


def build_features(rows: pd.DataFrame, training_prior: float) -> dict[str, np.ndarray]:
    key_columns = sorted({column for keys, _ in FAMILIES.values() for column in keys})
    columns = ["time_ms", "row_index", "long_view", *key_columns]
    ordered = list(rows.sort_values("time_ms", kind="stable")[columns].itertuples(
        index=False, name=None,
    ))
    positions = {name: columns.index(name) for name in key_columns}
    features = {
        **{
            f"online_{family}_{lag}": np.zeros(len(rows), dtype=np.float32)
            for lag in LAGS_MS for family in FAMILIES
        },
        **{
            f"online_user_recent{k}_{lag}": np.zeros(len(rows), dtype=np.float32)
            for lag in LAGS_MS for k in RECENT_K
        },
    }

    for lag_name, lag_ms in LAGS_MS.items():
        stats = {family: {} for family in FAMILIES}
        user_history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
        global_sum = 0.0
        global_count = 0
        history_index = 0
        for query_number, query in enumerate(ordered):
            cutoff = int(query[0]) - lag_ms
            while history_index < len(ordered) and int(ordered[history_index][0]) <= cutoff:
                event = ordered[history_index]
                outcome = float(event[2])
                global_sum += outcome
                global_count += 1
                for family, (family_columns, _) in FAMILIES.items():
                    family_positions = tuple(positions[column] for column in family_columns)
                    key = make_key(event, family_positions)
                    total, count = stats[family].get(key, (0.0, 0))
                    stats[family][key] = (total + outcome, count + 1)
                user_history[str(event[positions["user_id"]])].append(outcome)
                history_index += 1

            prior = (global_sum + 200.0 * training_prior) / (global_count + 200.0)
            output_index = int(query[1])
            for family, (family_columns, smoothing) in FAMILIES.items():
                family_positions = tuple(positions[column] for column in family_columns)
                key = make_key(query, family_positions)
                total, count = stats[family].get(key, (0.0, 0))
                rate = (total + smoothing * prior) / (count + smoothing)
                features[f"online_{family}_{lag_name}"][output_index] = rate - prior
            history = user_history.get(str(query[positions["user_id"]]))
            for k in RECENT_K:
                if history:
                    values = list(history)[-k:]
                    features[f"online_user_recent{k}_{lag_name}"][output_index] = (
                        float(np.mean(values)) - prior
                    )
            if query_number and query_number % 30_000 == 0:
                print(json.dumps({
                    "lag": lag_name,
                    "query_rows": query_number,
                    "feedback_rows": history_index,
                }), flush=True)
    return features


def main() -> None:
    rows, training_prior = load_rows()
    features = build_features(rows, training_prior)
    np.savez_compressed(CACHE, **features)
    manifest = {
        "experiment": "prequential online standard-feed feedback features",
        "evaluation_mode": "online/prequential; not static locked-holdout",
        "validation_rows": len(rows),
        "feature_count": len(features),
        "families": list(FAMILIES),
        "lags_ms": LAGS_MS,
        "uses_only_standard_events_before_query_minus_lag": True,
        "hidden_test_accessed": False,
        "cache": str(CACHE),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
