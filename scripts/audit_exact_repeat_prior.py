#!/usr/bin/env python3
"""Screen and confirm an exact user-video Bayesian history correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.champion import load_champion_scores, within_user_rank
from backend.kuailab.resources import ProcessResourceTracker
from scripts import kuairand_runner as runner


def user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def exact_repeat_signal(train_rows: list[tuple], rows: list[tuple], alpha: float) -> np.ndarray:
    global_sum = 0.0
    global_count = 0
    user_sum: Counter[str] = Counter()
    user_count: Counter[str] = Counter()
    pair_sum: Counter[tuple[str, str]] = Counter()
    pair_count: Counter[tuple[str, str]] = Counter()
    for row in train_rows:
        user, video, label = str(row[1]), str(row[2]), float(row[6])
        global_sum += label
        global_count += 1
        user_sum[user] += label
        user_count[user] += 1
        pair_sum[(user, video)] += label
        pair_count[(user, video)] += 1
    global_prior = global_sum / max(global_count, 1)
    signal = np.zeros(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        user, video = str(row[1]), str(row[2])
        user_prior = (user_sum[user] + 10.0 * global_prior) / (user_count[user] + 10.0)
        count = pair_count[(user, video)]
        if count:
            pair_prior = (pair_sum[(user, video)] + alpha * user_prior) / (count + alpha)
            signal[index] = pair_prior - user_prior
    return signal


def user_standardize(users: list[str], values: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        groups.setdefault(str(user), []).append(index)
    output = np.zeros(len(values), dtype=np.float64)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        group = values[indices]
        scale = float(group.std())
        if scale > 1e-8:
            output[indices] = (group - group.mean()) / scale
    return output


def as_metrics(result: dict) -> dict[str, float]:
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()
    tracker = ProcessResourceTracker()

    splits = runner.load_development_splits(ROOT / "external" / "KuaiRand-Pure" / "data")
    train_rows = splits["train"]
    valid_rows = splits["valid"]
    valid_y = np.asarray([row[6] for row in valid_rows], dtype=np.float32)
    valid_users = [str(row[1]) for row in valid_rows]

    early_rows = [row for row in train_rows if int(row[0]) <= 20220414]
    screen_rows = [row for row in train_rows if int(row[0]) >= 20220415]
    screen_y = np.asarray([row[6] for row in screen_rows], dtype=np.float32)
    screen_users = [str(row[1]) for row in screen_rows]
    meta_rows = [row for row in train_rows if int(row[0]) >= 20220414]
    meta_base = np.load(
        ROOT / "runtime" / "stacked-reranker" / "out-of-time-base-cutoff-20220413.npz",
        allow_pickle=False,
    )["scores"].astype(np.float64)
    if len(meta_rows) != len(meta_base):
        raise RuntimeError(f"Screen base alignment failed: {len(meta_rows)} != {len(meta_base)}")
    screen_base = meta_base[np.asarray([int(row[0]) >= 20220415 for row in meta_rows])]
    screen_base = within_user_rank(screen_users, screen_base)
    screen_baseline = as_metrics(
        runner.evaluate_module.evaluate(screen_users, screen_y, screen_base)
    )

    alphas = (0.5, 1.0, 2.0, 4.0, 8.0)
    weights = np.round(np.linspace(-0.5, 0.5, 41), 6)
    screen_trials = []
    for alpha in alphas:
        signal = user_standardize(
            screen_users, exact_repeat_signal(early_rows, screen_rows, alpha)
        )
        for weight in weights:
            metrics = as_metrics(
                runner.evaluate_module.evaluate(
                    screen_users, screen_y, screen_base + float(weight) * signal,
                )
            )
            screen_trials.append((metrics["primary"], -abs(float(weight)), alpha, float(weight), metrics))
    _, _, selected_alpha, selected_weight, screen_metrics = max(screen_trials)

    champion_scores, manifest = load_champion_scores(expected_rows=len(valid_rows))
    champion = within_user_rank(valid_users, champion_scores)
    valid_signal = user_standardize(
        valid_users, exact_repeat_signal(train_rows, valid_rows, selected_alpha)
    )
    candidate = champion + selected_weight * valid_signal
    baseline_metrics = as_metrics(
        runner.evaluate_module.evaluate(valid_users, valid_y, champion)
    )
    candidate_metrics = as_metrics(
        runner.evaluate_module.evaluate(valid_users, valid_y, candidate)
    )
    folds = np.asarray([user_fold(user) for user in valid_users], dtype=np.int8)
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        baseline_fold = as_metrics(
            runner.evaluate_module.evaluate(
                np.asarray(valid_users, dtype=object)[mask].tolist(), valid_y[mask], champion[mask]
            )
        )
        candidate_fold = as_metrics(
            runner.evaluate_module.evaluate(
                np.asarray(valid_users, dtype=object)[mask].tolist(), valid_y[mask], candidate[mask]
            )
        )
        fold_results.append({
            "fold": fold,
            "primary_gain": candidate_fold["primary"] - baseline_fold["primary"],
            "gauc_gain": candidate_fold["gauc"] - baseline_fold["gauc"],
            "ndcg5_gain": candidate_fold["ndcg5"] - baseline_fold["ndcg5"],
        })

    result = {
        "hypothesis": "Bayesian-smoothed exact user-video history adds signal beyond identifier coverage.",
        "protocol": {
            "screen_train": "2022-04-08..2022-04-14",
            "screen_validation": "2022-04-15..2022-04-21",
            "confirmation_train": "2022-04-08..2022-04-21",
            "confirmation_validation": "2022-04-22..2022-04-28",
            "hidden_test_accessed": False,
        },
        "screen_baseline": screen_baseline,
        "screen_selected": {
            "alpha": selected_alpha,
            "weight": selected_weight,
            "metrics": screen_metrics,
            "gain": screen_metrics["primary"] - screen_baseline["primary"],
        },
        "confirmation": {
            "champion_manifest_primary": manifest["validation_metrics"]["primary"],
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "gain": candidate_metrics["primary"] - baseline_metrics["primary"],
            "changed_rows": int(np.count_nonzero(valid_signal)),
            "folds": fold_results,
            "all_folds_nonnegative": all(item["primary_gain"] >= -1e-12 for item in fold_results),
            "both_metrics_nonnegative": (
                candidate_metrics["gauc"] >= baseline_metrics["gauc"]
                and candidate_metrics["ndcg5"] >= baseline_metrics["ndcg5"]
            ),
        },
        "resource_usage": tracker.finish(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.scores_output,
        scores=candidate.astype(np.float32),
        exact_repeat_signal=valid_signal.astype(np.float32),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
