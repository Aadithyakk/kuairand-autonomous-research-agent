#!/usr/bin/env python3
"""Daily causal meta-logistic model over frozen rankers and online features."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
HERE = Path(os.environ.get(
    "KUAI_PREQUENTIAL_WORKDIR",
    str(ROOT / "runtime" / "prequential-teacher"),
)).resolve()
HERE.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
RUNTIME = Path(os.environ.get(
    "KUAI_STATIC_MODEL_ROOT",
    str(ROOT / "runtime"),
)).resolve()
BASE = HERE / "cross_panel_on_online_logistic_pass3_scores.npz"
OUTPUT = HERE / "prequential_daily_static_stack_logistic_scores.npz"
REPORT = HERE / "prequential_daily_static_stack_logistic_results.json"
CACHES = (
    HERE / "prequential_standard_feedback_features.npz",
    HERE / "causal_streaming_random_features.npz",
    HERE / "causal_random_watch_features.npz",
    HERE / "causal_random_user_state_features.npz",
    HERE / "causal_decayed_random_features.npz",
    HERE / "causal_random_action_state_features.npz",
    HERE / "causal_random_transition_features.npz",
)
STATIC_MODELS = (
    "final-sessionx-consensus.npz",
    "stacked-reranker-scores-history-rank_xendcg-cutoff20220413-rolling0-"
    "components0-session1-sessionx1-sessionw1.npz",
    "history-catboost-classifier-probe-i250-seq40-plain-s151.npz",
    "history-deepfm-k16-h128-lr0.001-session-fullmeta-watchratio0.4-s293.npz",
    "history-deepfm-k16-h128-lr0.001-session-fullmeta-"
    "sessionmargin0.05m1.0f0.1-s293.npz",
    "history-catboost-classifier-refit-i500-seq40-session-fullmeta-s257.npz",
    "batch-slate-meta-catboost-s751.npz",
    "history-deepfm-k16-h128-lr0.001-session-fullmeta-s293.npz",
    "history-deepfm-k16-h128-lr0.001-session-fullmeta-aux0.15-s337.npz",
    "multitask-a0.2-k16-h128-lr0.001-s71.npz",
    "din-h20-k16-hidden128-lr0.001-s0.npz",
    "cross-session-gru-e8-h64-head128-s431.npz",
    "history-catboost-ranker-probe-i220-seq40-session-s157.npz",
    "lgbm-lambdarank-r500-l31-lr0.03-leaf100-s0.npz",
    "lightgcn-k32-l2-lr0.01-s0.npz",
    "ffm-base-star-k16-lr0.001-s401.npz",
    "ordinal-watch-a0.3-k16-h128-lr0.001-s121.npz",
    "threshold-deepfm-a0.3-k16-h128-lr0.001-s131.npz",
    "stacked-catboost-classifier-scores-extended-cutoff20220413-session1-"
    "sessionx1-sessionw0-multir0-trend0-affinity0-ctx0.npz",
    "stacked-linear-scores-history-cutoff20220413-rolling0-components0-"
    "session1-sessionx1-sessionw0-multir0-trend0-affinity0-ctx0-"
    "transition1.npz",
)
ALPHAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
WEIGHTS = (-0.10, -0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02,
           0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

sys.path.insert(0, str(SCRIPT_DIR))
from joint_terminal_gate_search import (  # noqa: E402
    exact_metrics, factorize, fast_metrics, rank_ordinal,
)


def user_balanced_weights(users: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(users, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def main() -> None:
    rows = pd.read_csv(
        DATA / "log_standard_4_22_to_5_08_pure.csv",
        usecols=["user_id", "date", "long_view"],
        dtype={"user_id": "string"},
    )
    rows = rows.loc[rows["date"] <= 20220428].reset_index(drop=True)
    users = rows["user_id"].astype(str).to_numpy()
    dates = rows["date"].to_numpy(dtype=np.int64)
    labels = rows["long_view"].to_numpy(dtype=np.int8)
    codes, _, counts = factorize(users)

    with np.load(BASE) as archive:
        base = np.asarray(archive["selected"], dtype=np.float64)
    base_rank = rank_ordinal(base, codes, counts).astype(np.float32)
    columns = [base_rank]
    names = ["current_online_champion_rank"]

    for model_name in STATIC_MODELS:
        path = RUNTIME / model_name
        with np.load(path) as archive:
            values = np.asarray(archive["scores"], dtype=np.float64)
        if len(values) != len(rows):
            raise ValueError(f"Unexpected row count in {path}: {len(values)}")
        columns.append(rank_ordinal(values, codes, counts).astype(np.float32))
        names.append(f"static_rank:{model_name}")

    for cache in CACHES:
        with np.load(cache) as archive:
            for name in archive.files:
                values = np.asarray(archive[name], dtype=np.float32)
                if len(values) != len(rows):
                    raise ValueError(f"Unexpected row count in {cache}:{name}")
                columns.append(values)
                names.append(f"{cache.stem}:{name}")
    matrix = np.column_stack(columns).astype(np.float32)

    predictions: dict[float, np.ndarray] = {}
    fit_log = []
    unique_dates = np.unique(dates)
    for alpha in ALPHAS:
        score = base_rank.astype(np.float64).copy()
        for current_date in unique_dates[1:]:
            train_mask = dates < current_date
            test_mask = dates == current_date
            model = make_pipeline(
                SimpleImputer(strategy="constant", fill_value=0.0),
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss", penalty="l2", alpha=alpha, max_iter=100,
                    tol=1e-4, average=True, random_state=2026,
                ),
            )
            model.fit(
                matrix[train_mask], labels[train_mask],
                sgdclassifier__sample_weight=user_balanced_weights(users[train_mask]),
            )
            score[test_mask] = model.decision_function(matrix[test_mask])
            entry = {
                "alpha": alpha, "test_date": int(current_date),
                "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
            }
            fit_log.append(entry)
            print(json.dumps(entry), flush=True)
        predictions[alpha] = score

    base_fast = fast_metrics(labels, base_rank, codes, counts)
    base_exact = exact_metrics(users, labels, base_rank)
    fold_ids = np.asarray([int(user) % 4 for user in users], dtype=np.int8)
    fold_cache = {}
    for fold in range(4):
        mask = fold_ids == fold
        fc, _, fn = factorize(users[mask])
        fold_cache[fold] = (
            mask, fc, fn,
            fast_metrics(labels[mask], base_rank[mask], fc, fn),
        )

    trials = []
    for alpha, raw_score in predictions.items():
        model_rank = rank_ordinal(raw_score, codes, counts)
        for weight in WEIGHTS:
            candidate = base_rank + weight * model_rank
            metrics = fast_metrics(labels, candidate, codes, counts)
            delta = {key: metrics[key] - base_fast[key] for key in base_fast}
            stable = (
                metrics["primary"] > base_fast["primary"]
                and metrics["gauc"] >= base_fast["gauc"]
                and metrics["ndcg5"] >= base_fast["ndcg5"]
            )
            folds = []
            if stable:
                for fold, (mask, fc, fn, fold_base) in fold_cache.items():
                    fold_metrics = fast_metrics(
                        labels[mask], candidate[mask], fc, fn,
                    )
                    fold_delta = {
                        key: fold_metrics[key] - fold_base[key] for key in fold_base
                    }
                    stable &= all(value >= -1e-12 for value in fold_delta.values())
                    folds.append({"fold": fold, "delta": fold_delta})
            trials.append({
                "alpha": alpha, "weight": weight, "metrics": metrics,
                "delta": delta, "stable": bool(stable), "folds": folds,
            })

    stable_trials = [trial for trial in trials if trial["stable"]]
    selected = (
        max(stable_trials, key=lambda trial: trial["metrics"]["primary"])
        if stable_trials else None
    )
    selected_scores = base_rank.astype(np.float64)
    if selected is not None:
        model_rank = rank_ordinal(predictions[selected["alpha"]], codes, counts)
        selected_scores = base_rank + selected["weight"] * model_rank
        selected["exact_metrics"] = exact_metrics(users, labels, selected_scores)
        selected["exact_delta"] = {
            key: selected["exact_metrics"][key] - base_exact[key]
            for key in base_exact
        }

    np.savez_compressed(
        OUTPUT,
        base=base.astype(np.float32),
        selected=selected_scores.astype(np.float32),
        **{
            f"alpha_{alpha:g}": score.astype(np.float32)
            for alpha, score in predictions.items()
        },
    )
    report = {
        "experiment": "daily causal meta-logistic over 20 frozen rankers",
        "evaluation_mode": "online/prequential; prior dates only",
        "decision": "promote_online" if selected else "retain_online_champion",
        "base_exact": base_exact,
        "selected": selected,
        "best_unconstrained": max(trials, key=lambda x: x["metrics"]["primary"]),
        "stable_trial_count": len(stable_trials),
        "feature_count": int(matrix.shape[1]),
        "feature_names": names,
        "fit_log": fit_log,
        "static_models": list(STATIC_MODELS),
        "uses_only_prior_dates_for_each_daily_model": True,
        "hidden_test_accessed": False,
        "artifacts": {"scores": str(OUTPUT), "report": str(REPORT)},
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
