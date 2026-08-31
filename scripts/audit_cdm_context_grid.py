#!/usr/bin/env python3
"""Robust CDM/MMR content-context audit against the frozen champion.

The earlier locked MMR confirmation improved the global primary score but lost
one component metric in two actual-user-ID folds.  This follow-up asks a single
question: can a predeclared family of outcome-free content similarities and
top-pool sizes retain that gain in every fold?

The search never uses an outcome as an input feature.  Each leave-one-fold-out
selector sees only the other three folds.  Promotion additionally requires the
four selectors to agree on one similarity family and pool size, a narrow lambda
range, and a fixed consensus configuration that improves GAUC, nDCG@5, and
primary globally and in every fold.  Hidden-test outcomes remain untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from scripts import train_contextual_diversity as cdm


DEFAULT_OUTPUT = ROOT / "results" / "context-distillation" / "cdm-context-grid.json"
DEFAULT_SCORES = ROOT / "runtime" / "context-distillation" / "cdm-context-consensus.npz"
PAPER = "https://arxiv.org/abs/2406.09021"

# All configurations are declared before outcomes are loaded.  They isolate
# which static content relation supplies a useful competing-item context.
SIMILARITY_TEMPLATES: dict[str, tuple[float, float, float, float]] = {
    "author": (1.0, 0.0, 0.0, 0.0),
    "music": (0.0, 1.0, 0.0, 0.0),
    "video_type": (0.0, 0.0, 1.0, 0.0),
    "tags": (0.0, 0.0, 0.0, 1.0),
    "author_tags": (0.5, 0.0, 0.0, 0.5),
    "author_music_tags": (0.4, 0.2, 0.0, 0.4),
    "original": (0.35, 0.20, 0.10, 0.35),
    "uniform": (0.25, 0.25, 0.25, 0.25),
}
POOL_GRID = (5, 8, 10, 12, 15)
LAMBDA_GRID = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12)
EPSILON = 1e-12


def similarity_components(left: tuple, right: tuple) -> np.ndarray:
    """Return author, music, type, and tag-Jaccard similarities."""
    union = left[6] | right[6]
    return np.asarray(
        [
            float(left[3] == right[3] and left[3] != "UNK"),
            float(left[4] == right[4] and left[4] != "UNK"),
            float(left[5] == right[5] and left[5] != "UNK"),
            len(left[6] & right[6]) / len(union) if union else 0.0,
        ],
        dtype=np.float32,
    )


def prepare_context(
    rows: list[tuple], users: list[str], base_scores: np.ndarray,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Cache top-15 content-relation matrices once for the full grid."""
    base = cdm.fractional_user_rank(users, base_scores)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        grouped[str(user)].append(index)
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    maximum_pool = max(POOL_GRID)
    for raw_indices in grouped.values():
        indices = np.asarray(raw_indices, dtype=np.int64)
        descending = indices[np.argsort(-base[indices], kind="stable")]
        top = descending[: min(maximum_pool, len(descending))]
        components = np.zeros((len(top), len(top), 4), dtype=np.float32)
        for left in range(len(top)):
            for right in range(left):
                value = similarity_components(rows[int(top[left])], rows[int(top[right])])
                components[left, right] = value
                components[right, left] = value
        prepared.append((top, components))
    return base, prepared


def rerank(
    base: np.ndarray,
    prepared: list[tuple[np.ndarray, np.ndarray]],
    weights: tuple[float, float, float, float],
    penalty: float,
    pool_size: int,
) -> np.ndarray:
    """Apply deterministic greedy contextual redundancy inside each top pool."""
    output = base.copy()
    weight_array = np.asarray(weights, dtype=np.float32)
    for top, components in prepared:
        count = min(pool_size, len(top))
        if count < 2:
            continue
        pool = list(range(count))
        selected: list[int] = []
        similarity = components[:count, :count] @ weight_array
        while pool:
            best_position = pool[0]
            best_value = -np.inf
            for position in pool:
                redundancy = (
                    max(float(similarity[position, prior]) for prior in selected)
                    if selected else 0.0
                )
                value = float(base[int(top[position])]) - penalty * redundancy
                if value > best_value:
                    best_value = value
                    best_position = position
            selected.append(best_position)
            pool.remove(best_position)
        original_values = np.sort(base[top[:count]])[::-1]
        output[top[np.asarray(selected, dtype=np.int64)]] = original_values
    return output


def metric_delta(candidate: dict, control: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in ("primary", "gauc", "ndcg5")
    }


def all_nonnegative(delta: dict) -> bool:
    return all(delta[key] >= -EPSILON for key in ("primary", "gauc", "ndcg5"))


def evaluate_configuration(
    users: list[str], labels: np.ndarray, scores: np.ndarray,
    folds: np.ndarray, control: dict, fold_controls: list[dict],
) -> tuple[dict, list[dict]]:
    metrics = cdm.metric_record(users, labels, scores)
    fold_metrics = []
    user_array = np.asarray(users, dtype=object)
    for fold in range(4):
        mask = folds == fold
        measured = cdm.metric_record(user_array[mask].tolist(), labels[mask], scores[mask])
        delta = metric_delta(measured, fold_controls[fold])
        fold_metrics.append({"fold": fold, "metrics": measured, "delta": delta})
    return ({
        "metrics": metrics,
        "delta": metric_delta(metrics, control),
        "all_global_metrics_nonnegative": all_nonnegative(metric_delta(metrics, control)),
        "all_four_folds_all_metrics_nonnegative": all(
            all_nonnegative(item["delta"]) for item in fold_metrics
        ),
    }, fold_metrics)


def select_leave_one_fold_out(candidates: list[dict]) -> list[dict]:
    """Select robustly on three folds, then report the untouched fourth fold."""
    selections = []
    for holdout in range(4):
        eligible = []
        for candidate in candidates:
            training = [item for item in candidate["folds"] if item["fold"] != holdout]
            if not all(all_nonnegative(item["delta"]) for item in training):
                continue
            primary = [item["delta"]["primary"] for item in training]
            if min(primary) <= EPSILON:
                continue
            eligible.append((min(primary), float(np.mean(primary)), candidate))
        if eligible:
            _, _, selected = max(
                eligible,
                key=lambda item: (
                    item[0], item[1], -item[2]["lambda"], -item[2]["pool_size"]
                ),
            )
            holdout_record = selected["folds"][holdout]
            selections.append({
                "holdout_fold": holdout,
                "selected": {
                    "template": selected["template"],
                    "weights": selected["weights"],
                    "pool_size": selected["pool_size"],
                    "lambda": selected["lambda"],
                },
                "held_out": holdout_record,
            })
        else:
            selections.append({
                "holdout_fold": holdout,
                "selected": "control",
                "held_out": {
                    "fold": holdout,
                    "delta": {"primary": 0.0, "gauc": 0.0, "ndcg5": 0.0},
                },
            })
    return selections


def consensus_candidate(selections: list[dict], candidates: list[dict]) -> dict | None:
    """Require exact semantic/pool agreement and a narrow selected lambda range."""
    if any(item["selected"] == "control" for item in selections):
        return None
    templates = {item["selected"]["template"] for item in selections}
    pools = {item["selected"]["pool_size"] for item in selections}
    lambdas = [float(item["selected"]["lambda"]) for item in selections]
    if len(templates) != 1 or len(pools) != 1 or max(lambdas) - min(lambdas) > 0.04:
        return None
    target_lambda = min(LAMBDA_GRID, key=lambda value: abs(value - float(np.median(lambdas))))
    template = next(iter(templates))
    pool = next(iter(pools))
    return next(
        candidate for candidate in candidates
        if candidate["template"] == template
        and candidate["pool_size"] == pool
        and candidate["lambda"] == target_lambda
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()
    started = time.monotonic()

    data_dir = ROOT / "external" / "KuaiRand-Pure" / "data"
    rows = cdm.load_confirmation_rows(data_dir)
    users = [row[1] for row in rows]
    labels = np.asarray([row[7] for row in rows], dtype=np.float32)
    champion, manifest = cdm.load_champion_scores(expected_rows=len(rows))
    control = cdm.metric_record(users, labels, champion)
    expected = manifest["validation_metrics"]
    if any(abs(control[key] - float(expected[key])) > 1e-9 for key in ("primary", "gauc", "ndcg5")):
        raise RuntimeError(f"Frozen champion alignment failed: {control} != {expected}")

    fold_ids = np.asarray([cdm.user_fold(user) for user in users], dtype=np.int8)
    user_array = np.asarray(users, dtype=object)
    fold_controls = []
    for fold in range(4):
        mask = fold_ids == fold
        fold_controls.append(cdm.metric_record(
            user_array[mask].tolist(), labels[mask], champion[mask]
        ))

    base, prepared = prepare_context(rows, users, champion)
    candidates = []
    best_scores: dict[tuple[str, int, float], np.ndarray] = {}
    total = len(SIMILARITY_TEMPLATES) * len(POOL_GRID) * len(LAMBDA_GRID)
    completed = 0
    for template, weights in SIMILARITY_TEMPLATES.items():
        for pool_size in POOL_GRID:
            for penalty in LAMBDA_GRID:
                scores = rerank(base, prepared, weights, penalty, pool_size)
                summary, fold_metrics = evaluate_configuration(
                    users, labels, scores, fold_ids, control, fold_controls
                )
                candidate = {
                    "template": template,
                    "weights": list(weights),
                    "pool_size": pool_size,
                    "lambda": penalty,
                    **summary,
                    "folds": fold_metrics,
                    "changed_rows": int(np.count_nonzero(scores != base)),
                }
                candidates.append(candidate)
                best_scores[(template, pool_size, penalty)] = scores
                completed += 1
                if completed % 60 == 0:
                    print(f"evaluated {completed}/{total} configurations", flush=True)

    selections = select_leave_one_fold_out(candidates)
    consensus = consensus_candidate(selections, candidates)
    promotable = bool(
        consensus is not None
        and consensus["all_global_metrics_nonnegative"]
        and consensus["all_four_folds_all_metrics_nonnegative"]
        and consensus["delta"]["primary"] > EPSILON
        and all(all_nonnegative(item["held_out"]["delta"]) for item in selections)
    )
    robust = [
        candidate for candidate in candidates
        if candidate["all_global_metrics_nonnegative"]
        and candidate["all_four_folds_all_metrics_nonnegative"]
        and candidate["delta"]["primary"] > EPSILON
    ]
    top = sorted(candidates, key=lambda item: item["metrics"]["primary"], reverse=True)[:20]
    robust_top = sorted(robust, key=lambda item: item["metrics"]["primary"], reverse=True)[:20]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    if consensus is not None:
        key = (consensus["template"], consensus["pool_size"], consensus["lambda"])
        consensus_scores = best_scores[key]
    else:
        consensus_scores = base
    np.savez_compressed(
        args.scores_output,
        champion_scores=champion.astype(np.float32),
        consensus_scores=consensus_scores.astype(np.float32),
        users=np.asarray(users, dtype="U"),
        fold_ids=fold_ids,
    )

    report = {
        "experiment": "CDM/MMR content-context stability grid",
        "paper": PAPER,
        "status": "promotable" if promotable else "rejected_or_exploratory",
        "hypothesis": (
            "A content-context penalty with a different similarity family or top-pool "
            "scope can retain the prior global MMR gain without fold regressions."
        ),
        "protocol": {
            "candidate_features": "static author, music, video type, and tag overlap only",
            "outcomes_used_as_features": False,
            "confirmation_window": "2022-04-22..2022-04-28",
            "hidden_test_outcomes_parsed": False,
            "selection": "four leave-one-actual-user-ID-fold-out selectors",
            "promotion_gate": (
                "all selectors agree on template and pool, lambda range <= 0.04, "
                "and every held-out/global/fixed-consensus metric is nonnegative"
            ),
            "configurations": total,
        },
        "control": control,
        "leave_one_fold_out": selections,
        "selectors_all_held_out_metrics_nonnegative": all(
            all_nonnegative(item["held_out"]["delta"]) for item in selections
        ),
        "consensus": consensus,
        "promotable": promotable,
        "robust_configuration_count": len(robust),
        "top_robust_configurations": robust_top,
        "top_global_configurations": top,
        "resources": tracker.finish(),
        "elapsed_wall_seconds": round(time.monotonic() - started, 3),
        "artifacts": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "scores": str(args.scores_output.relative_to(ROOT)),
        },
        "recommendation": (
            "Promote the fixed consensus after checksum and recipe integration review."
            if promotable else
            "Retain the frozen champion; use the audit to design a learned contextual-distillation ablation."
        ),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "control": control,
        "robust_configuration_count": len(robust),
        "leave_one_fold_out": selections,
        "consensus": consensus,
        "promotable": promotable,
        "top_global": top[0],
        "resources": report["resources"],
        "report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
