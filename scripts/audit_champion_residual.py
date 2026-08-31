#!/usr/bin/env python3
"""Cross-fit a frozen candidate residual before considering promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.champion import load_champion_scores, within_user_rank
from scripts import kuairand_runner as runner


def user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", type=Path, nargs="+",
        help=(
            "One or more .npz files with candidate_scores or scores; multiple "
            "seeds are rank-ensembled."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gate",
        choices=(
            "all", "long-video", "low-training-history", "sessions-7plus",
            "slate-6plus",
        ),
        default="all",
        help="Apply the residual only to a pre-specified outcome-free regime.",
    )
    args = parser.parse_args()

    splits = runner.load_development_splits(ROOT / "external" / "KuaiRand-Pure" / "data")
    _, valid_y, valid_users = runner.data_module.encode(splits)[0]["valid"]
    champion_scores, manifest = load_champion_scores(expected_rows=len(valid_y))
    champion = within_user_rank(valid_users, champion_scores)
    candidates = []
    for artifact in args.artifact:
        with np.load(artifact, allow_pickle=False) as archive:
            score_key = "candidate_scores" if "candidate_scores" in archive else "scores"
            scores = np.asarray(archive[score_key], dtype=np.float64)
        candidates.append(within_user_rank(valid_users, scores))
    candidate = within_user_rank(valid_users, np.mean(np.stack(candidates), axis=0))
    folds = np.asarray([user_fold(str(user)) for user in valid_users], dtype=np.int8)
    gate = np.ones(len(valid_y), dtype=np.float64)
    if args.gate == "sessions-7plus":
        rows_by_user: dict[str, list[tuple[int, int]]] = {}
        for index, row in enumerate(splits["valid"]):
            rows_by_user.setdefault(str(row[1]), []).append((int(row[8]), index))
        gate.fill(0.0)
        for user_rows in rows_by_user.values():
            ordered = sorted(user_rows)
            session_count = 1 + sum(
                current[0] - previous[0] > 1_800_000
                for previous, current in zip(ordered, ordered[1:])
            )
            if session_count >= 7:
                gate[[index for _, index in ordered]] = 1.0
    elif args.gate == "slate-6plus":
        user_counts: dict[str, int] = {}
        for user in valid_users:
            user_key = str(user)
            user_counts[user_key] = user_counts.get(user_key, 0) + 1
        gate = np.asarray(
            [user_counts[str(user)] >= 6 for user in valid_users],
            dtype=np.float64,
        )
    elif args.gate == "long-video":
        gate = np.asarray(
            [float(row[5]) >= 18_000.0 for row in splits["valid"]],
            dtype=np.float64,
        )
    elif args.gate == "low-training-history":
        training_counts: dict[str, int] = {}
        for row in splits["train"]:
            user_key = str(row[1])
            training_counts[user_key] = training_counts.get(user_key, 0) + 1
        validation_user_counts = {
            str(user): training_counts.get(str(user), 0) for user in valid_users
        }
        threshold = float(np.quantile(list(validation_user_counts.values()), 0.25))
        gate = np.asarray(
            [validation_user_counts[str(user)] <= threshold for user in valid_users],
            dtype=np.float64,
        )
    grid = np.round(np.linspace(-0.25, 0.25, 101), 6)

    def metrics(mask: np.ndarray, weight: float) -> dict:
        score = champion[mask] + weight * gate[mask] * (candidate[mask] - champion[mask])
        result = runner.evaluate_module.evaluate(
            np.asarray(valid_users, dtype=object)[mask].tolist(), valid_y[mask], score,
        )
        return {
            "primary": float(result["primary"]),
            "gauc": float(result["GAUC"]),
            "ndcg5": float(result["nDCG@5"]),
        }

    fold_results = []
    selected_weights = []
    for fold in range(4):
        selection_mask = folds != fold
        holdout_mask = folds == fold
        baseline_selection = metrics(selection_mask, 0.0)
        scans = [(float(weight), metrics(selection_mask, float(weight))) for weight in grid]
        weight, selected = max(
            scans,
            key=lambda item: (
                round(item[1]["primary"] - baseline_selection["primary"], 12),
                -abs(item[0]),
            ),
        )
        if selected["primary"] <= baseline_selection["primary"] + 1e-12:
            weight = 0.0
        selected_weights.append(weight)
        baseline_holdout = metrics(holdout_mask, 0.0)
        selected_holdout = metrics(holdout_mask, weight)
        fold_results.append({
            "fold": fold,
            "selected_weight": weight,
            "selection_gain": selected["primary"] - baseline_selection["primary"],
            "holdout_gain": selected_holdout["primary"] - baseline_holdout["primary"],
            "holdout_gauc_gain": selected_holdout["gauc"] - baseline_holdout["gauc"],
            "holdout_ndcg5_gain": selected_holdout["ndcg5"] - baseline_holdout["ndcg5"],
        })

    fixed_weight = float(np.mean(selected_weights))
    all_mask = np.ones(len(valid_y), dtype=bool)
    baseline = metrics(all_mask, 0.0)
    fixed = metrics(all_mask, fixed_weight)
    result = {
        "artifacts": [str(path) for path in args.artifact],
        "gate": args.gate,
        "gated_rows": int(gate.sum()),
        "champion_primary": manifest["validation_metrics"]["primary"],
        "candidate_primary": metrics(all_mask, 1.0)["primary"],
        "protocol": "four actual-user-ID folds; select on three, report held-out fourth; zero-preferring grid",
        "selected_weights": selected_weights,
        "folds": fold_results,
        "fixed_weight": fixed_weight,
        "fixed_metrics": fixed,
        "fixed_gain": fixed["primary"] - baseline["primary"],
        "all_holdout_folds_nonnegative": all(item["holdout_gain"] >= -1e-12 for item in fold_results),
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
