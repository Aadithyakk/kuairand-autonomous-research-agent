#!/usr/bin/env python3
"""Independent temporal replication for the exploratory author-only CDM rule."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import audit_cdm_context_grid as audit
from scripts import train_contextual_diversity as cdm


SOURCE = ROOT / "runtime" / "stacked-reranker" / "out-of-time-components-cutoff-20220413.npz"
OUTPUT = ROOT / "results" / "context-distillation" / "cdm-author-replication.json"
WEIGHTS = audit.SIMILARITY_TEMPLATES["author"]
POOL_SIZE = 5
PENALTY = 0.04


def measured(rows: list[tuple], scores: np.ndarray) -> dict:
    users = [row[1] for row in rows]
    labels = np.asarray([row[7] for row in rows], dtype=np.float32)
    base, prepared = audit.prepare_context(rows, users, scores)
    candidate = audit.rerank(base, prepared, WEIGHTS, PENALTY, POOL_SIZE)
    control_metrics = cdm.metric_record(users, labels, base)
    candidate_metrics = cdm.metric_record(users, labels, candidate)
    delta = audit.metric_delta(candidate_metrics, control_metrics)
    return {
        "control": control_metrics,
        "candidate": candidate_metrics,
        "delta": delta,
        "all_metrics_nonnegative": audit.all_nonnegative(delta),
        "primary_positive": delta["primary"] > audit.EPSILON,
        "changed_rows": int(np.count_nonzero(candidate != base)),
    }


def main() -> int:
    tracker = ProcessResourceTracker()
    rows = cdm.load_rows(ROOT / "external" / "KuaiRand-Pure" / "data")
    meta_rows = [row for row in rows if row[0] > 20220413]
    with np.load(SOURCE, allow_pickle=False) as archive:
        scores = np.asarray(archive["deep"], dtype=np.float64)
    if len(scores) != len(meta_rows):
        raise RuntimeError(f"Out-of-time score alignment failed: {len(scores)} != {len(meta_rows)}")

    april14_mask = np.asarray([row[0] == 20220414 for row in meta_rows], dtype=bool)
    later_mask = np.asarray([row[0] >= 20220415 for row in meta_rows], dtype=bool)
    april14_rows = [row for row, keep in zip(meta_rows, april14_mask, strict=True) if keep]
    later_rows = [row for row, keep in zip(meta_rows, later_mask, strict=True) if keep]
    april14 = measured(april14_rows, scores[april14_mask])
    april15_21 = measured(later_rows, scores[later_mask])
    replicated = bool(
        april14["all_metrics_nonnegative"]
        and april14["primary_positive"]
        and april15_21["all_metrics_nonnegative"]
        and april15_21["primary_positive"]
    )
    report = {
        "experiment": "fixed author-only CDM/MMR temporal replication",
        "status": "replicated" if replicated else "failed_replication",
        "configuration": {
            "similarity": "same non-UNK author only",
            "weights": list(WEIGHTS),
            "pool_size": POOL_SIZE,
            "lambda": PENALTY,
        },
        "protocol": {
            "source_model_fit_through": "2022-04-13",
            "first_replication": "2022-04-14",
            "second_replication": "2022-04-15..2022-04-21",
            "configuration_locked_from": "exploratory April 22-28 context grid",
            "outcomes_used_as_features": False,
            "hidden_test_accessed": False,
            "promotion_requirement": "positive primary and nonnegative GAUC/nDCG@5 on both earlier windows",
        },
        "april14": april14,
        "april15_21": april15_21,
        "replicated": replicated,
        "resources": tracker.finish(),
        "recommendation": (
            "Eligible for recipe integration review with explicit post-hoc provenance."
            if replicated else
            "Reject the exploratory author-only micro-gain and retain the frozen champion."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
