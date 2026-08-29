#!/usr/bin/env python3
"""Trusted KuaiRand validation adapter.

The research agent receives aggregate validation metrics only. This process owns
label access and never evaluates the test date range during development.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from backend.kuailab.pairwise import sample_pair_indices


STARTER = Path(os.getenv("KUAI_STARTER_KIT_DIR", ROOT / "external" / "kuairand-starter-kit")).resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("official_data", STARTER / "data.py")
evaluate_module = load_module("official_evaluate", STARTER / "evaluate.py")
sys.modules["data"] = data_module
sys.modules["evaluate"] = evaluate_module
baseline_module = load_module("official_baseline", STARTER / "baseline.py")


def load_development_splits(data_dir: Path) -> dict:
    """Load train/validation only. Rows after 2022-04-28 are discarded."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    train, valid = [], []
    for filename in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with (data_dir / filename).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                date = int(row["date"])
                if date > 20220428:
                    continue
                item = (
                    date, row["user_id"], row["video_id"], video_to_author.get(row["video_id"], "UNK"),
                    row["tab"], float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0,
                    int(row["hourmin"]) // 100,
                )
                (train if date <= 20220421 else valid).append(item)
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Unexpected split sizes: train={len(train)}, valid={len(valid)}")
    return {"train": train, "valid": valid, "test": []}


class WeightedFM(baseline_module.FM):
    def __init__(self, *args, positive_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_weight = positive_weight

    def step(self, X, y):
        z, E, S = self.logits(X)
        weights = np.where(y > 0.5, self.positive_weight, 1.0).astype(np.float32)
        g = ((baseline_module.sigmoid(z) - y) * weights / weights.sum()).astype(np.float32)
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for parameter, gradient, mean, variance in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            mean *= b1
            mean += (1 - b1) * gradient
            variance *= b2
            variance += (1 - b2) * (gradient * gradient)
            parameter -= self.lr * (mean / (1 - b1 ** self.t)) / (np.sqrt(variance / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        probabilities = baseline_module.sigmoid(z)
        loss = -(weights * (y * np.log(probabilities + 1e-9) + (1 - y) * np.log(1 - probabilities + 1e-9))).sum() / weights.sum()
        return float(loss)


class PairwiseFM(WeightedFM):
    """Factorization Machine optimized for within-user BPR comparisons."""

    def pair_step(self, positive_x: np.ndarray, negative_x: np.ndarray) -> float:
        positive_z, positive_e, positive_s = self.logits(positive_x)
        negative_z, negative_e, negative_s = self.logits(negative_x)
        difference = positive_z - negative_z
        gradient = ((baseline_module.sigmoid(difference) - 1.0) / len(difference)).astype(np.float32)
        gradient_v = np.zeros_like(self.V)
        gradient_w = np.zeros_like(self.W)
        np.add.at(gradient_w, positive_x, gradient[:, None])
        np.add.at(gradient_w, negative_x, -gradient[:, None])
        np.add.at(gradient_v, positive_x, gradient[:, None, None] * (positive_s[:, None, :] - positive_e))
        np.add.at(gradient_v, negative_x, -gradient[:, None, None] * (negative_s[:, None, :] - negative_e))
        gradient_v += self.l2 * self.V
        gradient_w += self.l2 * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, update, mean, variance in (
            (self.V, gradient_v, self.mV, self.vV),
            (self.W, gradient_w, self.mW, self.vW),
        ):
            mean *= beta1
            mean += (1 - beta1) * update
            variance *= beta2
            variance += (1 - beta2) * (update * update)
            parameter -= self.lr * (mean / (1 - beta1 ** self.t)) / (np.sqrt(variance / (1 - beta2 ** self.t)) + epsilon)
        return float(np.mean(np.logaddexp(0.0, -difference)))


def train_one(
    train_x, train_y, valid_x, valid_y, valid_users, dimension: int, output_dir: Path,
    parameters: dict, seed: int, positive_weight: float, checkpoint_prefix: str = "checkpoint",
):
    k = max(4, min(int(parameters.get("k", 16)), 32))
    lr = max(0.0001, min(float(parameters.get("lr", 0.001)), 0.01))
    epochs = max(5, min(int(parameters.get("epochs", 40)), 60))
    batch = max(2048, min(int(parameters.get("batch_size", 8192)), 32768))
    patience = max(2, min(int(parameters.get("patience", 4)), 8))
    model = WeightedFM(dimension, k=k, lr=lr, seed=seed, positive_weight=positive_weight)
    random = np.random.default_rng(seed)
    best_score, best_state, bad = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        order = random.permutation(len(train_y))
        losses = [model.step(train_x[order[start:start + batch]], train_y[order[start:start + batch]]) for start in range(0, len(order), batch)]
        scores = model.predict(valid_x)
        metrics = evaluate_module.evaluate(valid_users, valid_y, scores)
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)), "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]), "primary": float(metrics["primary"]),
            "users": int(metrics["users"]), "rows": int(metrics["rows"]),
        })
        if metrics["primary"] > best_score + 1e-5:
            best_score, bad = metrics["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b), scores.copy())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("FM training produced no checkpoint")
    model.V, model.W, model.b, best_scores = best_state
    np.savez_compressed(output_dir / f"{checkpoint_prefix}-seed-{seed}.npz", V=model.V, W=model.W, b=model.b)
    return best_scores, {"objective": "weighted_bce", "seed": seed, "positive_weight": positive_weight, "epochs": history}


def train_pairwise(
    train_x, train_y, train_users, valid_x, valid_y, valid_users,
    dimension: int, output_dir: Path, parameters: dict,
):
    k = max(4, min(int(parameters.get("k", 16)), 32))
    lr = max(0.00005, min(float(parameters.get("pairwise_lr", 0.002)), 0.01))
    epochs = max(3, min(int(parameters.get("pairwise_epochs", 12)), 30))
    batch = max(2048, min(int(parameters.get("batch_size", 8192)), 32768))
    patience = max(2, min(int(parameters.get("pairwise_patience", 4)), 8))
    seed = int(parameters.get("pairwise_seed", parameters.get("seed", 0)))
    model = PairwiseFM(dimension, k=k, lr=lr, seed=seed)
    random = np.random.default_rng(seed)
    best_score, best_state, bad = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        positive_indices, negative_indices = sample_pair_indices(train_users, train_y, random)
        if not len(positive_indices):
            raise RuntimeError("Pairwise FM requires at least one user with both positive and negative impressions")
        losses = [
            model.pair_step(
                train_x[positive_indices[start:start + batch]],
                train_x[negative_indices[start:start + batch]],
            )
            for start in range(0, len(positive_indices), batch)
        ]
        scores = model.predict(valid_x)
        metrics = evaluate_module.evaluate(valid_users, valid_y, scores)
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)), "pairs": int(len(positive_indices)),
            "GAUC": float(metrics["GAUC"]), "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]), "users": int(metrics["users"]), "rows": int(metrics["rows"]),
        })
        if metrics["primary"] > best_score + 1e-5:
            best_score, bad = metrics["primary"], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b), scores.copy())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("Pairwise FM training produced no checkpoint")
    model.V, model.W, model.b, best_scores = best_state
    np.savez_compressed(output_dir / f"pairwise-checkpoint-seed-{seed}.npz", V=model.V, W=model.W, b=model.b)
    return best_scores, {"objective": "bpr_pairwise", "seed": seed, "epochs": history}


def train_fm(splits: dict, output_dir: Path, proposal: dict) -> dict:
    encoded, dimension = data_module.encode(splits)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    parameters = proposal.get("parameters", {})
    experiment_type = proposal.get("experiment_type", "fm_config")
    seed = int(parameters.get("seed", 0))
    ensemble_types = {"fm_ensemble", "fm_pairwise_blend", "fm_deep_blend", "fm_temporal_deep_blend"}
    seeds = parameters.get("ensemble_seeds", [seed]) if experiment_type in ensemble_types else [seed]
    seeds = [int(value) for value in seeds[:3]]
    positive_weight = float(parameters.get("positive_weight", 1.0))
    positive_weight = max(1.0, min(positive_weight, 10.0))
    predictions, histories, pairwise_histories, deep_histories, temporal_histories = [], [], [], [], []
    if experiment_type != "fm_pairwise":
        for model_seed in seeds:
            scores, history = train_one(train_x, train_y, valid_x, valid_y, valid_users, dimension, output_dir, parameters, model_seed, positive_weight)
            predictions.append(scores)
            histories.append(history)
    if experiment_type in {"fm_pairwise", "fm_pairwise_blend", "fm_deep_blend", "fm_temporal_deep_blend"}:
        pairwise_scores, pairwise_history = train_pairwise(
            train_x, train_y, train_users, valid_x, valid_y, valid_users,
            dimension, output_dir, parameters,
        )
        pairwise_histories.append(pairwise_history)
        if experiment_type == "fm_pairwise":
            best_scores = pairwise_scores
        else:
            blend_weight = max(0.0, min(float(parameters.get("blend_weight", 0.455)), 1.0))
            pointwise_scores = np.mean(np.stack(predictions), axis=0)
            best_scores = (1.0 - blend_weight) * pointwise_scores + blend_weight * pairwise_scores
    else:
        blend_weight = 0.0
        best_scores = np.mean(np.stack(predictions), axis=0)
    if experiment_type in {"fm_deep_blend", "fm_temporal_deep_blend"}:
        from backend.kuailab.deepfm import train_deepfm

        deep_scores, deep_history = train_deepfm(
            train_x, train_y, valid_x, valid_y, valid_users,
            dimension, output_dir, parameters, evaluate_module.evaluate,
        )
        deep_histories.append(deep_history)
        deep_blend_weight = max(0.0, min(float(parameters.get("deep_blend_weight", 0.23)), 1.0))
        best_scores = (1.0 - deep_blend_weight) * best_scores + deep_blend_weight * deep_scores
    else:
        deep_blend_weight = 0.0
    if experiment_type == "fm_temporal_deep_blend":
        from backend.kuailab.temporal import encode_clock_context

        temporal_encoded, temporal_dimension = encode_clock_context(splits)
        temporal_train_x, temporal_train_y, _ = temporal_encoded["train"]
        temporal_valid_x, temporal_valid_y, temporal_valid_users = temporal_encoded["valid"]
        temporal_scores, temporal_history = train_one(
            temporal_train_x, temporal_train_y, temporal_valid_x, temporal_valid_y,
            temporal_valid_users, temporal_dimension, output_dir, parameters, seed,
            positive_weight, checkpoint_prefix="temporal-checkpoint",
        )
        temporal_history["features"] = ["hour", "daypart", "weekday"]
        temporal_histories.append(temporal_history)
        temporal_blend_weight = max(0.0, min(float(parameters.get("temporal_blend_weight", 0.024)), 0.2))
        base_standardized = (best_scores - best_scores.mean()) / max(float(best_scores.std()), 1e-8)
        temporal_standardized = (temporal_scores - temporal_scores.mean()) / max(float(temporal_scores.std()), 1e-8)
        best_scores = (1.0 - temporal_blend_weight) * base_standardized + temporal_blend_weight * temporal_standardized
    else:
        temporal_blend_weight = 0.0
    final = evaluate_module.evaluate(valid_users, valid_y, best_scores)
    (output_dir / "training-history.json").write_text(json.dumps({
        "experiment_type": experiment_type,
        "blend_weight": blend_weight if experiment_type in {"fm_pairwise_blend", "fm_deep_blend", "fm_temporal_deep_blend"} else None,
        "deep_blend_weight": deep_blend_weight if experiment_type in {"fm_deep_blend", "fm_temporal_deep_blend"} else None,
        "temporal_blend_weight": temporal_blend_weight if experiment_type == "fm_temporal_deep_blend" else None,
        "runs": histories,
        "pairwise_runs": pairwise_histories,
        "deep_runs": deep_histories,
        "temporal_runs": temporal_histories,
    }, indent=2), encoding="utf-8")
    return {"primary": float(final["primary"]), "gauc": float(final["GAUC"]), "ndcg5": float(final["nDCG@5"])}


def main() -> int:
    request_path = os.getenv("KUAI_RUNNER_REQUEST")
    if not request_path:
        raise RuntimeError("KUAI_RUNNER_REQUEST is required")
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    output_dir = Path(request_path).resolve().parent
    data_dir = Path(request["dataset_path"]).resolve()
    tracker = ProcessResourceTracker()
    started = time.monotonic()
    proposal = {"experiment_type": "fm_config", "parameters": {}}
    if request.get("action") == "experiment":
        proposal_path = Path(request["proposal_path"])
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    load_started = time.monotonic()
    splits = load_development_splits(data_dir)
    data_loading_seconds = time.monotonic() - load_started
    train_started = time.monotonic()
    metrics = train_fm(splits, output_dir, proposal)
    train_seconds = time.monotonic() - train_started
    metrics["runtime_seconds"] = round(time.monotonic() - started, 3)
    usage = tracker.finish(train_seconds=train_seconds)
    usage["data_loading_seconds"] = round(data_loading_seconds, 3)
    metrics["resource_usage"] = usage
    (output_dir / "resource-usage.json").write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
    Path(request["metrics_path"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"action": request.get("action"), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
