#!/usr/bin/env python3
"""Learned contextual gate for the tag-only CDM/MMR correction.

The deterministic tag penalty showed a sizeable GAUC gain but inconsistent
nDCG.  This script learns, from earlier user slates only, which structurally
defined slates should receive that correction.  The learner never receives a
user ID or any outcome at inference.  Its features are score margins, slate
size, rank displacement, and static tag/author redundancy.

Screen: fit/OOF-select on April 14, then lock the gate for April 15-21.
Confirmation: only after a passed screen, refit on April 14-21 and evaluate the
same gate once on April 22-28 against the checksum-verified champion.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import audit_cdm_context_grid as audit
from scripts import train_contextual_diversity as cdm


SOURCE = ROOT / "runtime" / "stacked-reranker" / "out-of-time-components-cutoff-20220413.npz"
RESULTS = ROOT / "results" / "context-distillation"
RUNTIME = ROOT / "runtime" / "context-distillation"
SCREEN_REPORT = RESULTS / "cdm-context-gate-screen.json"
CONFIRM_REPORT = RESULTS / "cdm-context-gate-confirmation.json"
SCREEN_SCORES = RUNTIME / "cdm-context-gate-screen.npz"
CONFIRM_SCORES = RUNTIME / "cdm-context-gate-confirmation.npz"
PAPER = "https://arxiv.org/abs/2406.09021"
TEACHER_WEIGHTS = audit.SIMILARITY_TEMPLATES["tags"]
TEACHER_POOL = 5
TEACHER_LAMBDA = 0.09
GATE_FRACTIONS = (0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)
SEED = 24060921


def per_user_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def per_user_ndcg(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")[:5]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2, dtype=np.float64))
    dcg = float(np.sum(labels[order] * discounts))
    ideal = np.sort(labels)[::-1][:5]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def tag_jaccard(left: tuple, right: tuple) -> float:
    union = left[6] | right[6]
    return len(left[6] & right[6]) / len(union) if union else 0.0


def examples(
    rows: list[tuple], base: np.ndarray, teacher: np.ndarray,
) -> list[dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row[1])].append(index)
    records = []
    for user, raw_indices in grouped.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        changed = indices[teacher[indices] != base[indices]]
        if not len(changed):
            continue
        descending = indices[np.argsort(-base[indices], kind="stable")]
        top = descending[: min(5, len(descending))]
        pairwise_tags = [
            tag_jaccard(rows[int(top[left])], rows[int(top[right])])
            for left in range(len(top)) for right in range(left)
        ]
        tag_counts: dict[str, int] = defaultdict(int)
        for index in top:
            for tag in rows[int(index)][6]:
                tag_counts[tag] += 1
        top_values = base[top]
        displacement = teacher[indices] - base[indices]
        features = np.asarray([
            np.log1p(len(indices)),
            len(changed),
            float(np.sum(np.abs(displacement))),
            float(np.max(displacement)),
            float(np.min(displacement)),
            float(top_values[0] - top_values[1]) if len(top_values) > 1 else 1.0,
            float(top_values[0] - top_values[-1]) if len(top_values) > 1 else 1.0,
            float(np.std(top_values)),
            float(np.mean(pairwise_tags)) if pairwise_tags else 0.0,
            float(np.max(pairwise_tags)) if pairwise_tags else 0.0,
            len(tag_counts) / max(sum(tag_counts.values()), 1),
            max(tag_counts.values(), default=0) / max(len(top), 1),
            len({rows[int(index)][3] for index in top}) / max(len(top), 1),
            len({rows[int(index)][4] for index in top}) / max(len(top), 1),
            float(np.mean([len(rows[int(index)][6]) for index in top])),
        ], dtype=np.float32)

        labels = np.asarray([rows[int(index)][7] for index in indices], dtype=np.float32)
        base_auc = per_user_auc(labels, base[indices])
        teacher_auc = per_user_auc(labels, teacher[indices])
        auc_gain = 0.0 if base_auc is None else float(teacher_auc - base_auc)
        ndcg_gain = per_user_ndcg(labels, teacher[indices]) - per_user_ndcg(labels, base[indices])
        target = 0.5 * auc_gain + 0.5 * ndcg_gain
        records.append({
            "user": user,
            "indices": indices,
            "features": features,
            "target": target,
            "sample_weight": max(int(np.sum(labels)), 1),
        })
    return records


def fit_model(records: list[dict]):
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = np.stack([record["features"] for record in records])
    y = np.asarray([record["target"] for record in records], dtype=np.float64)
    weights = np.asarray([record["sample_weight"] for record in records], dtype=np.float64)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=10.0,
        random_state=SEED,
    )
    model.fit(x, y, sample_weight=weights)
    return model


def oof_predictions(records: list[dict]) -> np.ndarray:
    output = np.zeros(len(records), dtype=np.float64)
    folds = np.asarray([cdm.user_fold(record["user"]) for record in records], dtype=np.int8)
    for fold in range(4):
        train_indices = np.flatnonzero(folds != fold)
        valid_indices = np.flatnonzero(folds == fold)
        if not len(train_indices) or not len(valid_indices):
            continue
        model = fit_model([records[int(index)] for index in train_indices])
        x = np.stack([records[int(index)]["features"] for index in valid_indices])
        output[valid_indices] = model.predict(x)
    return output


def gated_scores(
    base: np.ndarray, teacher: np.ndarray, records: list[dict],
    predictions: np.ndarray, fraction: float,
) -> tuple[np.ndarray, int]:
    output = base.copy()
    if fraction <= 0 or not len(records):
        return output, 0
    count = max(1, int(np.ceil(fraction * len(records))))
    selected = np.argsort(-predictions, kind="stable")[:count]
    for record_index in selected:
        indices = records[int(record_index)]["indices"]
        output[indices] = teacher[indices]
    return output, count


def measured(rows: list[tuple], scores: np.ndarray) -> dict:
    return cdm.metric_record(
        [row[1] for row in rows],
        np.asarray([row[7] for row in rows], dtype=np.float32),
        scores,
    )


def load_train_only() -> tuple[list[tuple], np.ndarray, list[tuple], np.ndarray]:
    rows = cdm.load_rows(ROOT / "external" / "KuaiRand-Pure" / "data")
    meta = [row for row in rows if row[0] > 20220413]
    with np.load(SOURCE, allow_pickle=False) as archive:
        scores = np.asarray(archive["deep"], dtype=np.float64)
    if len(scores) != len(meta):
        raise RuntimeError(f"Out-of-time score alignment failed: {len(scores)} != {len(meta)}")
    fit_mask = np.asarray([row[0] == 20220414 for row in meta], dtype=bool)
    screen_mask = np.asarray([row[0] >= 20220415 for row in meta], dtype=bool)
    fit_rows = [row for row, keep in zip(meta, fit_mask, strict=True) if keep]
    screen_rows = [row for row, keep in zip(meta, screen_mask, strict=True) if keep]
    return fit_rows, scores[fit_mask], screen_rows, scores[screen_mask]


def teacher_scores(rows: list[tuple], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    users = [row[1] for row in rows]
    base, prepared = audit.prepare_context(rows, users, scores)
    teacher = audit.rerank(base, prepared, TEACHER_WEIGHTS, TEACHER_LAMBDA, TEACHER_POOL)
    return base, teacher


def select_fraction(
    rows: list[tuple], base: np.ndarray, teacher: np.ndarray,
    records: list[dict], predictions: np.ndarray,
) -> tuple[float, list[dict]]:
    control = measured(rows, base)
    grid = []
    for fraction in GATE_FRACTIONS:
        scores, selected = gated_scores(base, teacher, records, predictions, fraction)
        metrics = measured(rows, scores)
        delta = audit.metric_delta(metrics, control)
        grid.append({
            "gate_fraction": fraction,
            "selected_changed_users": selected,
            "metrics": metrics,
            "delta": delta,
            "all_metrics_nonnegative": audit.all_nonnegative(delta),
        })
    eligible = [
        item for item in grid
        if item["gate_fraction"] > 0
        and item["all_metrics_nonnegative"]
        and item["delta"]["primary"] >= 1e-5
    ]
    if not eligible:
        return 0.0, grid
    selected = max(eligible, key=lambda item: (item["metrics"]["primary"], -item["gate_fraction"]))
    return float(selected["gate_fraction"]), grid


def run_screen() -> dict:
    tracker = ProcessResourceTracker()
    fit_rows, fit_raw, screen_rows, screen_raw = load_train_only()
    fit_base, fit_teacher = teacher_scores(fit_rows, fit_raw)
    screen_base, screen_teacher = teacher_scores(screen_rows, screen_raw)
    fit_records = examples(fit_rows, fit_base, fit_teacher)
    screen_records = examples(screen_rows, screen_base, screen_teacher)
    oof = oof_predictions(fit_records)
    fraction, grid = select_fraction(fit_rows, fit_base, fit_teacher, fit_records, oof)
    model = fit_model(fit_records)
    screen_predictions = model.predict(np.stack([record["features"] for record in screen_records]))
    screen_scores, selected_users = gated_scores(
        screen_base, screen_teacher, screen_records, screen_predictions, fraction
    )
    control = measured(screen_rows, screen_base)
    candidate = measured(screen_rows, screen_scores)
    delta = audit.metric_delta(candidate, control)
    passed = bool(
        fraction > 0
        and delta["primary"] >= 1e-5
        and audit.all_nonnegative(delta)
    )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCREEN_SCORES,
        control=screen_base.astype(np.float32),
        teacher=screen_teacher.astype(np.float32),
        candidate=screen_scores.astype(np.float32),
    )
    report = {
        "experiment": "learned CDM tag-context gate",
        "paper": PAPER,
        "status": "screen_passed" if passed else "screen_rejected",
        "teacher": {
            "similarity": "tag Jaccard only",
            "pool_size": TEACHER_POOL,
            "lambda": TEACHER_LAMBDA,
        },
        "protocol": {
            "source_score_fit_through": "2022-04-13",
            "OOF_gate_fit_and_fraction_selection": "2022-04-14",
            "locked_screen": "2022-04-15..2022-04-21",
            "confirmation_file_opened": False,
            "outcome_features_at_inference": False,
            "hidden_test_accessed": False,
            "retrospective_origin": "teacher configuration came from an exploratory April22-28 grid; this screen is an independent earlier-time replication",
        },
        "selection_examples": len(fit_records),
        "screen_examples": len(screen_records),
        "fraction_grid": grid,
        "selected_gate_fraction": fraction,
        "screen_selected_changed_users": selected_users,
        "control": control,
        "candidate": candidate,
        "delta": delta,
        "merits_confirmation": passed,
        "resources": tracker.finish(),
        "artifacts": {"scores": str(SCREEN_SCORES.relative_to(ROOT))},
        "recommendation": "run fixed confirmation" if passed else "reject learned context gate",
    }
    SCREEN_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def run_confirmation() -> dict:
    if not SCREEN_REPORT.exists():
        raise RuntimeError("Run the train-only screen first")
    screen_report = json.loads(SCREEN_REPORT.read_text(encoding="utf-8"))
    if not screen_report.get("merits_confirmation"):
        raise RuntimeError("The locked screen did not permit confirmation")
    fraction = float(screen_report["selected_gate_fraction"])
    tracker = ProcessResourceTracker()

    fit_rows, fit_raw, screen_rows, screen_raw = load_train_only()
    train_rows = fit_rows + screen_rows
    train_raw = np.concatenate([fit_raw, screen_raw])
    train_base, train_teacher = teacher_scores(train_rows, train_raw)
    train_records = examples(train_rows, train_base, train_teacher)
    model = fit_model(train_records)

    valid_rows = cdm.load_confirmation_rows(ROOT / "external" / "KuaiRand-Pure" / "data")
    users = [row[1] for row in valid_rows]
    labels = np.asarray([row[7] for row in valid_rows], dtype=np.float32)
    champion, manifest = cdm.load_champion_scores(expected_rows=len(valid_rows))
    control = measured(valid_rows, champion)
    expected = manifest["validation_metrics"]
    if any(abs(control[key] - float(expected[key])) > 1e-9 for key in ("primary", "gauc", "ndcg5")):
        raise RuntimeError("Frozen champion alignment failed")
    valid_base, valid_teacher = teacher_scores(valid_rows, champion)
    valid_records = examples(valid_rows, valid_base, valid_teacher)
    predictions = model.predict(np.stack([record["features"] for record in valid_records]))
    candidate_scores, selected_users = gated_scores(
        valid_base, valid_teacher, valid_records, predictions, fraction
    )
    candidate = measured(valid_rows, candidate_scores)
    delta = audit.metric_delta(candidate, control)

    fold_ids = np.asarray([cdm.user_fold(user) for user in users], dtype=np.int8)
    user_array = np.asarray(users, dtype=object)
    folds = []
    for fold in range(4):
        mask = fold_ids == fold
        fold_rows = [row for row, keep in zip(valid_rows, mask, strict=True) if keep]
        fold_control = cdm.metric_record(user_array[mask].tolist(), labels[mask], valid_base[mask])
        fold_candidate = cdm.metric_record(user_array[mask].tolist(), labels[mask], candidate_scores[mask])
        fold_delta = audit.metric_delta(fold_candidate, fold_control)
        folds.append({
            "fold": fold,
            "rows": len(fold_rows),
            "control": fold_control,
            "candidate": fold_candidate,
            "delta": fold_delta,
            "all_metrics_nonnegative": audit.all_nonnegative(fold_delta),
        })
    promotable = bool(
        delta["primary"] > 0
        and audit.all_nonnegative(delta)
        and all(item["all_metrics_nonnegative"] for item in folds)
    )
    RUNTIME.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CONFIRM_SCORES,
        champion=champion.astype(np.float32),
        candidate=candidate_scores.astype(np.float32),
        fold_ids=fold_ids,
    )
    report = {
        "experiment": "locked learned CDM tag-context gate confirmation",
        "paper": PAPER,
        "status": "promotable" if promotable else "rejected",
        "selected_gate_fraction_locked": fraction,
        "selected_changed_users": selected_users,
        "protocol": {
            "gate_refit": "2022-04-14..2022-04-21",
            "confirmation": "2022-04-22..2022-04-28",
            "retuned_on_confirmation": False,
            "hidden_test_outcomes_parsed": False,
            "outcomes_used_as_inference_features": False,
        },
        "control": control,
        "candidate": candidate,
        "delta": delta,
        "folds": folds,
        "promotable": promotable,
        "resources": tracker.finish(),
        "artifacts": {"scores": str(CONFIRM_SCORES.relative_to(ROOT))},
        "recommendation": "promote after recipe integration review" if promotable else "retain frozen champion",
    }
    CONFIRM_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    run_confirmation() if args.confirmation else run_screen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
