#!/usr/bin/env python3
"""Measure where the frozen champion leaves recoverable nDCG@5 on validation."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.champion import load_champion_scores


def ndcg_at_5(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")[:5]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.dot(labels[order], discounts))
    ideal_count = min(int(labels.sum()), 5)
    if ideal_count == 0:
        return 0.0
    return dcg / float(discounts[:ideal_count].sum())


def bucket(value: int, boundaries: tuple[tuple[int, str], ...], fallback: str) -> str:
    for upper, label in boundaries:
        if value <= upper:
            return label
    return fallback


def summarize(records: list[dict], key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    total_gap = sum(record["gap"] for record in records)
    output = []
    for name, group in sorted(grouped.items()):
        gaps = [record["gap"] for record in group]
        output.append({
            "regime": name,
            "users": len(group),
            "rows": sum(record["rows"] for record in group),
            "mean_ndcg5": round(float(np.mean([record["ndcg5"] for record in group])), 6),
            "recoverable_gap": round(float(np.mean(gaps)), 6),
            "share_of_total_gap": round(float(sum(gaps) / total_gap), 6) if total_gap else 0.0,
        })
    return output


def main() -> int:
    data_path = ROOT / "external" / "KuaiRand-Pure" / "data" / "log_standard_4_22_to_5_08_pure.csv"
    rows: list[dict] = []
    with data_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["date"]) > 20220428:
                continue
            rows.append({
                "user": row["user_id"],
                "label": 1 if row["long_view"] != "0" else 0,
                "date": int(row["date"]),
                "time_ms": int(row["time_ms"]),
            })

    scores, manifest = load_champion_scores(expected_rows=len(rows))
    indices_by_user: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_user[row["user"]].append(index)

    records = []
    for user, indices_list in indices_by_user.items():
        indices = np.asarray(indices_list, dtype=np.int64)
        labels = np.asarray([rows[index]["label"] for index in indices_list], dtype=np.int8)
        user_scores = scores[indices]
        ndcg = ndcg_at_5(labels, user_scores)
        positives = int(labels.sum())
        timestamps = sorted(rows[index]["time_ms"] for index in indices_list)
        sessions = 1 + sum(
            current - previous > 30 * 60 * 1000
            for previous, current in zip(timestamps, timestamps[1:])
        )
        records.append({
            "user": user,
            "rows": len(indices_list),
            "positives": positives,
            "days": len({rows[index]["date"] for index in indices_list}),
            "sessions": sessions,
            "ndcg5": ndcg,
            "gap": (1.0 if positives else 0.0) - ndcg,
            "slate_bucket": bucket(len(indices_list), ((5, "01-05"), (10, "06-10"), (20, "11-20"), (40, "21-40")), "41+"),
            "positive_bucket": bucket(positives, ((0, "0"), (1, "1"), (2, "2"), (5, "3-5"), (10, "6-10")), "11+"),
            "day_bucket": bucket(len({rows[index]["date"] for index in indices_list}), ((1, "1"), (2, "2"), (4, "3-4")), "5+"),
            "session_bucket": bucket(sessions, ((1, "1"), (3, "2-3"), (6, "4-6")), "7+"),
        })

    observed_ndcg = float(np.mean([record["ndcg5"] for record in records]))
    oracle_ndcg = float(np.mean([1.0 if record["positives"] else 0.0 for record in records]))
    result = {
        "source_primary": manifest["validation_metrics"]["primary"],
        "source_gauc": manifest["validation_metrics"]["gauc"],
        "source_ndcg5": manifest["validation_metrics"]["ndcg5"],
        "recomputed_ndcg5": observed_ndcg,
        "label_oracle_ndcg5": oracle_ndcg,
        "recoverable_ndcg5_gap": oracle_ndcg - observed_ndcg,
        "users": len(records),
        "rows": len(rows),
        "regimes": {
            "slate_size": summarize(records, "slate_bucket"),
            "positive_count": summarize(records, "positive_bucket"),
            "active_days": summarize(records, "day_bucket"),
            "session_count": summarize(records, "session_bucket"),
        },
    }
    output_path = ROOT / "results" / "final-model" / "champion-regime-analysis.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
