#!/usr/bin/env python3
"""Build strictly earlier-timestamp standard-feed feedback features."""
from __future__ import annotations

import json
import os
from collections import defaultdict, deque
import heapq
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
OUTPUT = HERE / "prequential_immediate_feedback_features.npz"
MANIFEST = HERE / "prequential_immediate_feedback_manifest.json"
WINDOWS_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "30m": 30 * 60_000,
    "2h": 2 * 60 * 60_000,
    "24h": 24 * 60 * 60_000,
}
RECENT_K = (1, 3, 5)
FAMILIES = {
    "user": (("user_id",), 20.0),
    "video": (("video_id",), 20.0),
    "author": (("author_id",), 40.0),
    "music": (("music_id",), 50.0),
    "tag": (("tag_primary",), 80.0),
    "type": (("video_type",), 200.0),
    "duration": (("duration_bucket",), 150.0),
    "user_video": (("user_id", "video_id"), 4.0),
    "user_author": (("user_id", "author_id"), 8.0),
    "user_tag": (("user_id", "tag_primary"), 12.0),
    "user_duration": (("user_id", "duration_bucket"), 12.0),
}
MATCH_FIELDS = (
    "video_id", "author_id", "music_id", "tag_primary", "video_type",
    "duration_bucket",
)


def load_rows() -> tuple[pd.DataFrame, dict[str, float]]:
    metadata = pd.read_csv(
        DATA / "video_features_basic_pure.csv",
        usecols=["video_id", "author_id", "music_id", "tag", "video_type"],
        dtype={column: "string" for column in (
            "video_id", "author_id", "music_id", "tag", "video_type"
        )},
    ).drop_duplicates("video_id", keep="last")
    metadata["tag_primary"] = (
        metadata["tag"].fillna("UNK").astype(str).str.split(",").str[0]
    )
    metadata = metadata.drop(columns="tag")
    columns = [
        "user_id", "video_id", "date", "time_ms", "duration_ms",
        "play_time_ms", "long_view", "is_click", "is_like", "is_follow",
        "is_comment", "is_forward", "is_hate", "is_profile_enter",
    ]
    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=columns,
        dtype={"user_id": "string", "video_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    rows["row_index"] = np.arange(len(rows), dtype=np.int64)
    rows = rows.merge(metadata, how="left", on="video_id", sort=False)
    for column in ("author_id", "music_id", "tag_primary", "video_type"):
        rows[column] = rows[column].fillna("UNK").astype(str)
    rows["user_id"] = rows["user_id"].astype(str)
    rows["video_id"] = rows["video_id"].astype(str)
    rows["duration_bucket"] = (
        rows["duration_ms"].astype(np.int64) // 10_000
    ).clip(0, 60).astype(str)
    rows["watch_ratio"] = np.clip(
        rows["play_time_ms"].to_numpy(dtype=np.float64)
        / np.maximum(rows["duration_ms"].to_numpy(dtype=np.float64), 1.0),
        0.0, 2.0,
    )
    rows["positive_action"] = (
        rows[[
            "is_like", "is_follow", "is_comment", "is_forward",
            "is_profile_enter",
        ]].max(axis=1).fillna(0).astype(np.float32)
    )
    train = pd.read_csv(
        DATA / "log_standard_4_08_to_4_21_pure.csv",
        usecols=["long_view", "play_time_ms", "duration_ms", "is_click",
                 "is_like", "is_follow", "is_comment", "is_forward",
                 "is_profile_enter", "is_hate"],
    )
    train_watch = np.clip(
        train["play_time_ms"].to_numpy(dtype=np.float64)
        / np.maximum(train["duration_ms"].to_numpy(dtype=np.float64), 1.0),
        0.0, 2.0,
    )
    priors = {
        "long": float(train["long_view"].mean()),
        "watch": float(np.mean(train_watch)),
        "click": float(train["is_click"].mean()),
        "positive": float(train[[
            "is_like", "is_follow", "is_comment", "is_forward",
            "is_profile_enter",
        ]].max(axis=1).mean()),
        "hate": float(train["is_hate"].mean()),
    }
    return rows, priors


def key_for(record: dict, fields: tuple[str, ...]):
    if len(fields) == 1:
        return record[fields[0]]
    return tuple(record[field] for field in fields)


def mean_centered(history: list[dict], field: str, prior: float) -> float:
    if not history:
        return 0.0
    return float(np.mean([event[field] for event in history]) - prior)


def main() -> None:
    rows, priors = load_rows()
    ordered_columns = [
        "time_ms", "row_index", "user_id", "video_id", "author_id",
        "music_id", "tag_primary", "video_type", "duration_bucket",
        "play_time_ms", "long_view", "watch_ratio", "is_click",
        "positive_action", "is_hate",
    ]
    ordered_frame = rows.sort_values(["time_ms", "row_index"], kind="stable")[
        ordered_columns
    ]
    records = ordered_frame.to_dict("records")
    features: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        for suffix in ("long_delta", "watch_delta", "log_count"):
            features[f"immediate_{family}_{suffix}"] = np.zeros(
                len(rows), dtype=np.float32
            )
    state_fields = ("long_view", "watch_ratio", "is_click", "positive_action", "is_hate")
    state_prior_names = ("long", "watch", "click", "positive", "hate")
    for window_name in WINDOWS_MS:
        features[f"immediate_user_count_{window_name}"] = np.zeros(
            len(rows), dtype=np.float32
        )
        for field_name in state_prior_names:
            features[f"immediate_user_{field_name}_{window_name}"] = np.zeros(
                len(rows), dtype=np.float32
            )
    for k in RECENT_K:
        for field_name in state_prior_names:
            features[f"immediate_user_{field_name}_last{k}"] = np.zeros(
                len(rows), dtype=np.float32
            )
    features["immediate_previous_gap_log"] = np.zeros(len(rows), dtype=np.float32)
    for field in MATCH_FIELDS:
        for suffix in ("same", "same_long", "same_watch"):
            features[f"immediate_previous_{field}_{suffix}"] = np.zeros(
                len(rows), dtype=np.float32
            )

    stats: dict[str, dict] = {family: {} for family in FAMILIES}
    user_history: dict[str, deque] = defaultdict(deque)
    global_long = 0.0
    global_watch = 0.0
    global_count = 0
    pending: list[tuple[int, int, dict]] = []
    pending_number = 0
    cursor = 0
    while cursor < len(records):
        timestamp = int(records[cursor]["time_ms"])
        end = cursor + 1
        while end < len(records) and int(records[end]["time_ms"]) == timestamp:
            end += 1

        while pending and pending[0][0] < timestamp:
            _, _, available = heapq.heappop(pending)
            outcome_long = float(available["long_view"])
            outcome_watch = float(available["watch_ratio"])
            global_long += outcome_long
            global_watch += outcome_watch
            global_count += 1
            for family, (family_fields, _) in FAMILIES.items():
                key = key_for(available, family_fields)
                long_sum, watch_sum, count = stats[family].get(key, (0.0, 0.0, 0))
                stats[family][key] = (
                    long_sum + outcome_long, watch_sum + outcome_watch, count + 1
                )
            user_history[str(available["user_id"])].append(available)

        current_prior_long = (
            global_long + 500.0 * priors["long"]
        ) / (global_count + 500.0)
        current_prior_watch = (
            global_watch + 500.0 * priors["watch"]
        ) / (global_count + 500.0)
        for record in records[cursor:end]:
            output_index = int(record["row_index"])
            for family, (family_fields, smoothing) in FAMILIES.items():
                key = key_for(record, family_fields)
                long_sum, watch_sum, count = stats[family].get(key, (0.0, 0.0, 0))
                features[f"immediate_{family}_long_delta"][output_index] = (
                    (long_sum + smoothing * current_prior_long) / (count + smoothing)
                    - current_prior_long
                )
                features[f"immediate_{family}_watch_delta"][output_index] = (
                    (watch_sum + smoothing * current_prior_watch) / (count + smoothing)
                    - current_prior_watch
                )
                features[f"immediate_{family}_log_count"][output_index] = np.log1p(count)

            history = user_history[str(record["user_id"])]
            cutoff_24h = timestamp - WINDOWS_MS["24h"]
            while history and int(history[0]["time_ms"]) < cutoff_24h:
                history.popleft()
            history_list = list(history)
            for window_name, window_ms in WINDOWS_MS.items():
                cutoff = timestamp - window_ms
                window_history = [
                    event for event in history_list if int(event["time_ms"]) >= cutoff
                ]
                features[f"immediate_user_count_{window_name}"][output_index] = np.log1p(
                    len(window_history)
                )
                for field, prior_name in zip(state_fields, state_prior_names):
                    features[f"immediate_user_{prior_name}_{window_name}"][output_index] = (
                        mean_centered(window_history, field, priors[prior_name])
                    )
            for k in RECENT_K:
                recent = history_list[-k:]
                for field, prior_name in zip(state_fields, state_prior_names):
                    features[f"immediate_user_{prior_name}_last{k}"][output_index] = (
                        mean_centered(recent, field, priors[prior_name])
                    )
            if history_list:
                previous = history_list[-1]
                features["immediate_previous_gap_log"][output_index] = np.log1p(
                    max(timestamp - int(previous["time_ms"]), 0) / 1000.0
                )
                for field in MATCH_FIELDS:
                    same = float(record[field] == previous[field])
                    features[f"immediate_previous_{field}_same"][output_index] = same
                    features[f"immediate_previous_{field}_same_long"][output_index] = (
                        same * (float(previous["long_view"]) - priors["long"])
                    )
                    features[f"immediate_previous_{field}_same_watch"][output_index] = (
                        same * (float(previous["watch_ratio"]) - priors["watch"])
                    )

        for record in records[cursor:end]:
            available_time = timestamp + max(int(record["play_time_ms"]), 0)
            heapq.heappush(pending, (available_time, pending_number, record))
            pending_number += 1
        cursor = end
        if cursor % 30_000 < end - (cursor - (end - cursor)):
            print(json.dumps({"processed": cursor}), flush=True)

    np.savez_compressed(OUTPUT, **features)
    manifest = {
        "experiment": "strictly earlier-timestamp standard feedback",
        "evaluation_mode": "online/prequential; current and same-timestamp labels excluded",
        "validation_rows": len(rows),
        "feature_count": len(features),
        "families": list(FAMILIES),
        "windows_ms": WINDOWS_MS,
        "same_timestamp_events_visible": False,
        "outcome_available_at": "time_ms + play_time_ms",
        "hidden_test_accessed": False,
        "artifact": str(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
