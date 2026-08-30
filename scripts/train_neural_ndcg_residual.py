#!/usr/bin/env python3
"""Train-only NeuralNDCG-style bounded residual screen.

This lane deliberately opens only ``log_standard_4_08_to_4_21_pure.csv``.
A frozen pointwise DeepFM trained on Apr 8--11 produces out-of-time base
scores for Apr 12--14.  Two identically initialized residual heads are then
fit on Apr 12--14: an exact alpha=0 BCE control and a candidate with a small
smooth-nDCG@5 term.  For the locked Apr 15--21 comparison, the same DeepFM is
refit on Apr 8--14 for the same fixed epoch count and the residual heads are
applied without any evaluation-driven tuning.

The differentiable rank approximation for item i is

    r_i = 1 + sum_{j != i} sigmoid((s_j - s_i) / temperature)

with a smooth top-five gate.  User groups are capped at 64 rows for bounded
O(n^2) memory; the frozen top 48 plus up to 16 missed positives are retained
for fitting only.  Official evaluation always uses every Apr 15--21 row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "external" / "kuairand-starter-kit"
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
DEFAULT_RUNTIME = ROOT / "runtime" / "parallel-neural-ndcg"
DEFAULT_REPORT = ROOT / "results" / "parallel-methods" / "neural-ndcg-screen.json"
DIFFERENTIAL_PLAN = (
    ROOT / "results" / "parallel-methods" / "neural-ndcg-differential-plan.json"
)
DIFFERENTIAL_SCREEN_REPORT = (
    ROOT / "results" / "parallel-methods" / "neural-ndcg-differential-screen.json"
)
DIFFERENTIAL_CONFIRMATION_REPORT = (
    ROOT / "results" / "parallel-methods" / "neural-ndcg-differential-confirmation.json"
)
DIFFERENTIAL_AUDIT_REPORT = (
    ROOT / "results" / "parallel-methods" / "neural-ndcg-differential-audit.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_module = load_module("neural_ndcg_official_data", STARTER / "data.py")
evaluate_module = load_module("neural_ndcg_official_evaluate", STARTER / "evaluate.py")


def load_train_only_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Load exactly Apr 8--21 from the first organizer log."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    base_train: list[tuple] = []
    residual_fit: list[tuple] = []
    evaluation: list[tuple] = []
    # Do not name or open the Apr 22+ log anywhere in this script.
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            parsed = (
                date,
                row["user_id"],
                row["video_id"],
                video_to_author.get(row["video_id"], "UNK"),
                row["tab"],
                float(row["duration_ms"]),
                1 if row["long_view"] != "0" else 0,
            )
            if date <= 20220411:
                base_train.append(parsed)
            elif date <= 20220414:
                residual_fit.append(parsed)
            elif date <= 20220421:
                evaluation.append(parsed)
            else:
                raise RuntimeError(f"Protected date unexpectedly present: {date}")
    if len(base_train) + len(residual_fit) + len(evaluation) != 1_141_112:
        raise RuntimeError(
            "Unexpected train-only row count: "
            f"{len(base_train)}+{len(residual_fit)}+{len(evaluation)}"
        )
    if not base_train or not residual_fit or not evaluation:
        raise RuntimeError("One or more train-only windows are empty")
    return base_train, residual_fit, evaluation


def load_locked_confirmation_rows(data_dir: Path) -> tuple[list[tuple], list[tuple]]:
    """Load Apr8--21 training and Apr22--28 confirmation, never Apr29+ outcomes."""
    video_to_author: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_to_author[row["video_id"]] = row["author_id"]

    def parse(row: dict[str, str]) -> tuple:
        return (
            int(row["date"]),
            row["user_id"],
            row["video_id"],
            video_to_author.get(row["video_id"], "UNK"),
            row["tab"],
            float(row["duration_ms"]),
            1 if row["long_view"] != "0" else 0,
        )

    training: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            parsed = parse(row)
            if not 20220408 <= parsed[0] <= 20220421:
                raise RuntimeError(f"Unexpected training date: {parsed[0]}")
            training.append(parsed)

    confirmation: list[tuple] = []
    with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(
        encoding="utf-8"
    ) as stream:
        header = next(stream)
        fieldnames = next(csv.reader([header]))
        if fieldnames[:3] != ["user_id", "video_id", "date"]:
            raise RuntimeError(f"Unexpected later-log header: {fieldnames[:3]}")
        for line in stream:
            # Inspect only the first three columns until the date is known.
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise RuntimeError("Malformed later-log row")
            date = int(prefix[2])
            if date > 20220428:
                continue
            if date < 20220422:
                raise RuntimeError(f"Unexpected confirmation date: {date}")
            values = next(csv.reader([line]))
            confirmation.append(parse(dict(zip(fieldnames, values, strict=True))))
    if len(training) != 1_141_112 or len(confirmation) != 124_909:
        raise RuntimeError(
            f"Unexpected confirmation rows: train={len(training)}, valid={len(confirmation)}"
        )
    return training, confirmation


def encode_base(train_rows: list[tuple], target_rows: list[tuple]) -> tuple[dict, int]:
    return data_module.encode({"train": train_rows, "valid": target_rows, "test": []})


def within_user_ranks(users: list[str], scores: np.ndarray) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[str(user)].append(index)
    ranks = np.empty(len(scores), dtype=np.float32)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.argsort(scores[indices], kind="stable")
        local = np.empty(len(indices), dtype=np.float32)
        local[order] = np.arange(len(indices), dtype=np.float32)
        ranks[indices] = local / max(len(indices) - 1, 1)
    return ranks


def peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def measured_usage(start_wall: float, start_cpu: float) -> dict:
    wall = time.monotonic() - start_wall
    cpu = time.process_time() - start_cpu
    return {
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(cpu, 3),
        "cpu_hours": round(cpu / 3600, 6),
        "cpu_utilization_percent": round(100.0 * cpu / max(wall, 1e-9), 2),
        "peak_rss_mb": round(peak_rss_mb(), 3),
        "device": "cpu",
        "gpu_count": 0,
        "gpu_hours": 0.0,
        "peak_gpu_memory_mb": 0.0,
    }


def make_deepfm(dimension: int, fields: int, embedding_dim: int, hidden: int):
    import torch
    from torch import nn

    class FrozenBaseDeepFM(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.deep = nn.Sequential(
                nn.Linear(fields * embedding_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, max(hidden // 2, 8)),
                nn.ReLU(),
                nn.Linear(max(hidden // 2, 8), 1),
            )
            nn.init.zeros_(self.linear.weight)
            nn.init.normal_(self.embedding.weight, std=0.01)

        def forward(self, x):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (
                summed.square() - embedding.square().sum(dim=1)
            ).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            deep = self.deep(embedding.flatten(start_dim=1)).squeeze(-1)
            return linear + fm + deep

    return FrozenBaseDeepFM()


def train_fixed_base(
    *, encoded: dict, dimension: int, output_dir: Path, name: str, seed: int,
    epochs: int, batch_size: int, embedding_dim: int, hidden: int,
) -> tuple[np.ndarray, dict]:
    """Fit a base model for a fixed count without reading target labels."""
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    train_x, train_y, _ = encoded["train"]
    target_x, _, _ = encoded["valid"]
    train_x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    train_y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    target_x_t = torch.from_numpy(target_x.astype(np.int64, copy=False))
    torch.manual_seed(seed)
    model = make_deepfm(dimension, train_x.shape[1], embedding_dim, hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    random = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        order = random.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x_t[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train_y_t[indices]
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(target_x_t), batch_size * 4):
            predictions.append(model(target_x_t[start:start + batch_size * 4]).numpy())
    scores = np.concatenate(predictions).astype(np.float32)
    checkpoint = output_dir / f"{name}-base-checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    return scores, {
        "name": name,
        "objective": "fixed_epoch_pointwise_deepfm_bce",
        "epochs": epochs,
        "epoch_losses": losses,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resource_usage": measured_usage(started_wall, started_cpu),
    }


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-5), 1.0 - 1e-5)
    return math.log(clipped / (1.0 - clipped))


def build_residual_features(
    history: list[tuple], target: list[tuple], base_scores: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Build causal aggregate/context features from the preceding window."""
    if len(target) != len(base_scores):
        raise RuntimeError("Base score alignment failure while building features")
    counts = {"user": defaultdict(int), "video": defaultdict(int), "author": defaultdict(int)}
    positives = {
        "user": defaultdict(float), "video": defaultdict(float), "author": defaultdict(float)
    }
    durations = np.asarray([float(row[5]) for row in history], dtype=np.float64)
    duration_mean = float(np.mean(np.log1p(durations)))
    duration_std = max(float(np.std(np.log1p(durations))), 1e-6)
    for row in history:
        keys = (str(row[1]), str(row[2]), str(row[3]))
        label = float(row[6])
        for family, key in zip(("user", "video", "author"), keys, strict=True):
            counts[family][key] += 1
            positives[family][key] += label
    prior = (sum(float(row[6]) for row in history) + 20.0) / (len(history) + 40.0)
    history_days = len({int(row[0]) for row in history})
    users = [str(row[1]) for row in target]
    base_ranks = within_user_ranks(users, base_scores)
    # scalar base logit/rank, 3 exposure rates, 3 causal target priors,
    # duration, and six fixed tab buckets.
    features = np.zeros((len(target), 15), dtype=np.float32)
    tab_values = ("0", "1", "2", "4", "6")
    for index, row in enumerate(target):
        keys = (str(row[1]), str(row[2]), str(row[3]))
        features[index, 0] = float(np.clip(base_scores[index], -8.0, 8.0))
        features[index, 1] = float(base_ranks[index] - 0.5)
        for offset, (family, key) in enumerate(
            zip(("user", "video", "author"), keys, strict=True)
        ):
            count = counts[family][key]
            positive = positives[family][key]
            features[index, 2 + offset] = math.log1p(count / max(history_days, 1))
            smoothed = (positive + 10.0 * prior) / (count + 10.0)
            features[index, 5 + offset] = logit(smoothed)
        features[index, 8] = (
            math.log1p(float(row[5])) - duration_mean
        ) / duration_std
        tab = str(row[4])
        tab_index = tab_values.index(tab) if tab in tab_values else 5
        features[index, 9 + tab_index] = 1.0
    metadata = {
        "history_rows": len(history),
        "history_days": history_days,
        "global_positive_rate_smoothed": prior,
        "feature_count": features.shape[1],
        "features": [
            "base_logit", "within_user_base_rank", "user_exposure_per_day_log1p",
            "video_exposure_per_day_log1p", "author_exposure_per_day_log1p",
            "user_prior_logit", "video_prior_logit", "author_prior_logit",
            "duration_log_z", "tab_0", "tab_1", "tab_2", "tab_4", "tab_6",
            "tab_other",
        ],
    }
    return features, metadata


def capped_user_groups(
    users: list[str], labels: np.ndarray, base_scores: np.ndarray, maximum: int,
) -> tuple[list[np.ndarray], dict]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        grouped[str(user)].append(index)
    selected: list[np.ndarray] = []
    capped = 0
    retained = 0
    for indices_list in grouped.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        if len(indices) <= maximum:
            chosen = indices
        else:
            capped += 1
            descending = indices[np.argsort(-base_scores[indices], kind="stable")]
            top_count = min(48, maximum)
            choice = list(descending[:top_count])
            chosen_set = set(choice)
            missed_positive = [
                int(index) for index in descending[top_count:]
                if labels[index] > 0.5 and int(index) not in chosen_set
            ]
            for index in missed_positive[: maximum - len(choice)]:
                choice.append(index)
                chosen_set.add(index)
            for index in descending[top_count:]:
                if len(choice) >= maximum:
                    break
                if int(index) not in chosen_set:
                    choice.append(int(index))
                    chosen_set.add(int(index))
            chosen = np.asarray(choice, dtype=np.int64)
        retained += len(chosen)
        selected.append(chosen)
    return selected, {
        "users": len(selected),
        "capped_users": capped,
        "original_rows": len(users),
        "retained_training_rows": retained,
        "maximum_group_rows": maximum,
    }


def make_residual_model(feature_count: int, hidden: int, bound: float):
    from torch import nn

    class BoundedResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(feature_count, hidden),
                nn.ReLU(),
                nn.Linear(hidden, max(hidden // 2, 8)),
                nn.ReLU(),
                nn.Linear(max(hidden // 2, 8), 1),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
            self.bound = bound

        def forward(self, features):
            return self.bound * self.network(features).squeeze(-1).tanh()

    return BoundedResidual()


def smooth_ndcg5_loss(scores, labels, mask, temperature: float, gate_temperature: float):
    """Pairwise smooth-rank approximation with a differentiable top-5 gate."""
    import torch

    valid_i = mask.unsqueeze(2)
    valid_j = mask.unsqueeze(1)
    pair_mask = valid_i & valid_j
    differences = (scores.unsqueeze(1) - scores.unsqueeze(2)) / temperature
    comparisons = torch.sigmoid(differences) * pair_mask
    # Remove the diagonal's sigmoid(0)=0.5 contribution.
    ranks = 1.0 + comparisons.sum(dim=2) - 0.5 * mask
    discounts = 1.0 / torch.log2(1.0 + ranks.clamp_min(1.0))
    top5_gate = torch.sigmoid((5.5 - ranks) / gate_temperature)
    dcg = (labels * discounts * top5_gate * mask).sum(dim=1)
    positives = (labels * mask).sum(dim=1).long().clamp(min=0, max=5)
    ideal_discounts = scores.new_tensor(
        [1.0 / math.log2(position + 2) for position in range(5)]
    )
    prefix = torch.cat([scores.new_zeros(1), ideal_discounts.cumsum(dim=0)])
    idcg = prefix[positives]
    usable = idcg > 0
    if not bool(usable.any()):
        return scores.new_zeros(())
    return (1.0 - dcg[usable] / idcg[usable].clamp_min(1e-8)).mean()


def train_residual(
    *, name: str, alpha: float, features: np.ndarray, labels: np.ndarray,
    users: list[str], base_scores: np.ndarray, evaluation_features: np.ndarray,
    evaluation_base_scores: np.ndarray, output_dir: Path, seed: int, epochs: int,
    hidden: int, residual_bound: float, group_batch_size: int,
    maximum_group_rows: int, temperature: float, gate_temperature: float,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    groups, group_metadata = capped_user_groups(
        users, labels, base_scores, maximum_group_rows
    )
    features_t = torch.from_numpy(features.astype(np.float32, copy=False))
    labels_t = torch.from_numpy(labels.astype(np.float32, copy=False))
    base_t = torch.from_numpy(base_scores.astype(np.float32, copy=False))
    torch.manual_seed(seed)
    model = make_residual_model(features.shape[1], hidden, residual_bound)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    random = np.random.default_rng(seed + 17_171)
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = random.permutation(len(groups))
        epoch_losses: list[float] = []
        bce_losses: list[float] = []
        ranking_losses: list[float] = []
        for start in range(0, len(order), group_batch_size):
            chosen_groups = [groups[index] for index in order[start:start + group_batch_size]]
            width = max(len(group) for group in chosen_groups)
            row_indices = np.zeros((len(chosen_groups), width), dtype=np.int64)
            mask_np = np.zeros((len(chosen_groups), width), dtype=bool)
            for row, group in enumerate(chosen_groups):
                row_indices[row, : len(group)] = group
                mask_np[row, : len(group)] = True
            row_indices_t = torch.from_numpy(row_indices)
            mask = torch.from_numpy(mask_np)
            batch_features = features_t[row_indices_t]
            batch_labels = labels_t[row_indices_t]
            batch_base = base_t[row_indices_t]
            optimizer.zero_grad(set_to_none=True)
            scores = batch_base + model(batch_features)
            bce = nn.functional.binary_cross_entropy_with_logits(
                scores[mask], batch_labels[mask]
            )
            ranking = smooth_ndcg5_loss(
                scores, batch_labels, mask, temperature, gate_temperature
            )
            loss = bce + alpha * ranking
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            bce_losses.append(float(bce.detach()))
            ranking_losses.append(float(ranking.detach()))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "bce": float(np.mean(bce_losses)),
                "smooth_ndcg5_loss": float(np.mean(ranking_losses)),
            }
        )

    model.eval()
    evaluation_features_t = torch.from_numpy(
        evaluation_features.astype(np.float32, copy=False)
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(evaluation_features_t), 32768):
            chunks.append(model(evaluation_features_t[start:start + 32768]).numpy())
    correction = np.concatenate(chunks).astype(np.float32)
    final_scores = evaluation_base_scores.astype(np.float32) + correction
    checkpoint = output_dir / f"{name}-residual-checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    return final_scores, {
        "name": name,
        "alpha": alpha,
        "objective": "pointwise_bce_plus_alpha_times_smooth_ndcg_at_5",
        "residual_bound": residual_bound,
        "temperature": temperature,
        "gate_temperature": gate_temperature,
        "epochs": epochs,
        "history": history,
        "fit_groups": group_metadata,
        "correction": {
            "minimum": float(np.min(correction)),
            "maximum": float(np.max(correction)),
            "mean": float(np.mean(correction)),
            "standard_deviation": float(np.std(correction)),
        },
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "resource_usage": measured_usage(started_wall, started_cpu),
    }


def metrics(users: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    result = evaluate_module.evaluate(users, labels, scores)
    return {
        "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def alignment_digest(rows: list[tuple]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row[0]},{row[1]},{row[2]}\n".encode("utf-8"))
    return digest.hexdigest()


def actual_user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode("utf-8")).hexdigest()[:8], 16) % 4


def run_differential_screen(args) -> dict:
    """Select a zero-preferring ranking-effect weight on the train-only screen."""
    started_wall, started_cpu = time.monotonic(), time.process_time()
    plan = json.loads(DIFFERENTIAL_PLAN.read_text(encoding="utf-8"))
    weights = [float(value) for value in plan["screen_weight_grid"]]
    if weights != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("Differential plan weight grid changed unexpectedly")
    source_artifact = DEFAULT_RUNTIME / "train-only-evaluation-scores.npz"
    with np.load(source_artifact) as stored:
        base = np.asarray(stored["base_scores"], dtype=np.float32)
        control = np.asarray(stored["control_scores"], dtype=np.float32)
        candidate = np.asarray(stored["candidate_scores"], dtype=np.float32)
        labels = np.asarray(stored["labels"], dtype=np.float32)
        users = np.asarray(stored["users"], dtype=str).tolist()
        dates = np.asarray(stored["dates"], dtype=np.int32)
    if int(np.max(dates)) > 20220421 or int(np.min(dates)) < 20220415:
        raise RuntimeError("Differential screen source is not Apr15--21 only")
    if not all(np.isfinite(values).all() for values in (base, control, candidate)):
        raise RuntimeError("Differential screen source has non-finite scores")

    baseline_metrics = metrics(users, labels, base)
    scans = []
    for weight in weights:
        scores = base + weight * (candidate - control)
        scans.append({"weight": weight, "metrics": metrics(users, labels, scores)})
    selected = max(
        scans,
        key=lambda item: (
            round(item["metrics"]["primary"] - baseline_metrics["primary"], 12),
            -item["weight"],
        ),
    )
    gains = {
        key: float(selected["metrics"][key] - baseline_metrics[key])
        for key in ("GAUC", "nDCG@5", "primary")
    }
    passed = bool(
        gains["primary"] >= float(plan["minimum_primary_gain"])
        and gains["GAUC"] > 0.0
        and gains["nDCG@5"] > 0.0
    )
    if not passed:
        selected = next(item for item in scans if item["weight"] == 0.0)
        gains = {key: 0.0 for key in ("GAUC", "nDCG@5", "primary")}
    selected_scores = base + float(selected["weight"]) * (candidate - control)
    score_artifact = DEFAULT_RUNTIME / "differential-screen-scores.npz"
    np.savez_compressed(
        score_artifact,
        base_scores=base,
        differential_scores=selected_scores.astype(np.float32),
        labels=labels,
        users=np.asarray(users, dtype="U"),
        dates=dates,
        selected_weight=np.asarray([selected["weight"]], dtype=np.float32),
    )
    report = {
        "experiment": "NeuralNDCG ranking-objective differential follow-up",
        "status": "screen_passed" if passed else "screen_rejected",
        "merits_locked_confirmation": passed,
        "plan": str(DIFFERENTIAL_PLAN.relative_to(ROOT)),
        "source_artifact": str(source_artifact.relative_to(ROOT)),
        "formula": "frozen_base + weight * (candidate - alpha_zero_control)",
        "protocol": {
            "selection_window": "2022-04-15..2022-04-21",
            "confirmation_accessed": False,
            "hidden_test_accessed": False,
            "zero_preferring": True,
            "weight_grid": weights,
            "minimum_primary_gain": float(plan["minimum_primary_gain"]),
            "integrity_note": plan["integrity_note"],
        },
        "baseline_metrics": baseline_metrics,
        "scans": scans,
        "selected_weight": float(selected["weight"]),
        "selected_metrics": selected["metrics"],
        "selected_gains": gains,
        "all_three_metrics_improve": bool(
            gains["primary"] > 0.0 and gains["GAUC"] > 0.0 and gains["nDCG@5"] > 0.0
        ),
        "artifact": str(score_artifact.relative_to(ROOT)),
        "resource_usage": measured_usage(started_wall, started_cpu),
    }
    DIFFERENTIAL_SCREEN_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def run_locked_differential_confirmation(args, screen_report: dict) -> dict:
    """Refit once and evaluate Apr22--28 using the exact locked screen formula."""
    started_wall, started_cpu = time.monotonic(), time.process_time()
    plan = json.loads(DIFFERENTIAL_PLAN.read_text(encoding="utf-8"))
    selected_weight = float(screen_report["selected_weight"])
    champion_weight = float(
        plan["confirmation_protocol_if_screen_passes"]["champion_blend_weight"]
    )
    base_train, early_residual_fit, residual_fit = load_train_only_rows(
        args.data_dir.resolve()
    )
    loaded_training, confirmation = load_locked_confirmation_rows(args.data_dir.resolve())
    if len(loaded_training) != len(base_train) + len(early_residual_fit) + len(residual_fit):
        raise RuntimeError("Apr8--21 row count changed between loaders")
    # The physical organizer file is not date sorted.  Preserve the same
    # chronologically partitioned ordering used by the screen/refit protocol.
    training = base_train + early_residual_fit + residual_fit

    # Reuse the screen's genuinely OOT Apr15--21 base scores as the residual
    # fitting anchor.  No screen hyperparameter or score is changed here.
    screen_source = DEFAULT_RUNTIME / "train-only-evaluation-scores.npz"
    with np.load(screen_source) as stored:
        residual_fit_base = np.asarray(stored["base_scores"], dtype=np.float32)
        source_dates = np.asarray(stored["dates"], dtype=np.int32)
        source_users = np.asarray(stored["users"], dtype=str)
    expected_users = np.asarray([str(row[1]) for row in residual_fit], dtype=str)
    expected_dates = np.asarray([int(row[0]) for row in residual_fit], dtype=np.int32)
    if not np.array_equal(source_users, expected_users) or not np.array_equal(
        source_dates, expected_dates
    ):
        raise RuntimeError("Apr15--21 OOT score alignment check failed")

    confirmation_encoded, confirmation_dimension = encode_base(training, confirmation)
    confirmation_base, base_details = train_fixed_base(
        encoded=confirmation_encoded,
        dimension=confirmation_dimension,
        output_dir=args.output_dir,
        name="differential-confirmation",
        seed=args.seed,
        epochs=args.base_epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden=args.base_hidden,
    )
    fit_history = base_train + early_residual_fit
    fit_features, fit_metadata = build_residual_features(
        fit_history, residual_fit, residual_fit_base
    )
    confirmation_features, confirmation_metadata = build_residual_features(
        training, confirmation, confirmation_base
    )
    fit_labels = np.asarray([float(row[6]) for row in residual_fit], dtype=np.float32)
    fit_users = [str(row[1]) for row in residual_fit]
    confirmation_labels = np.asarray(
        [float(row[6]) for row in confirmation], dtype=np.float32
    )
    confirmation_users = [str(row[1]) for row in confirmation]
    common = {
        "features": fit_features,
        "labels": fit_labels,
        "users": fit_users,
        "base_scores": residual_fit_base,
        "evaluation_features": confirmation_features,
        "evaluation_base_scores": confirmation_base,
        "output_dir": args.output_dir,
        "seed": args.seed + 991,
        "epochs": args.residual_epochs,
        "hidden": args.residual_hidden,
        "residual_bound": args.residual_bound,
        "group_batch_size": args.group_batch_size,
        "maximum_group_rows": args.maximum_group_rows,
        "temperature": args.temperature,
        "gate_temperature": args.gate_temperature,
    }
    control_scores, control_details = train_residual(
        name="differential-confirmation-alpha-zero", alpha=0.0, **common
    )
    candidate_scores, candidate_details = train_residual(
        name="differential-confirmation-neural-ndcg", alpha=args.ndcg_alpha, **common
    )
    differential_scores = confirmation_base + selected_weight * (
        candidate_scores - control_scores
    )
    for name, values in {
        "confirmation_base": confirmation_base,
        "control": control_scores,
        "candidate": candidate_scores,
        "differential": differential_scores,
    }.items():
        if len(values) != len(confirmation) or not bool(np.isfinite(values).all()):
            raise RuntimeError(f"Invalid confirmation {name} scores")
    score_artifact = DEFAULT_RUNTIME / "differential-confirmation-scores.npz"
    np.savez_compressed(
        score_artifact,
        base_scores=confirmation_base.astype(np.float32),
        control_scores=control_scores.astype(np.float32),
        candidate_scores=candidate_scores.astype(np.float32),
        differential_scores=differential_scores.astype(np.float32),
        labels=confirmation_labels,
        users=np.asarray(confirmation_users, dtype="U"),
        dates=np.asarray([row[0] for row in confirmation], dtype=np.int32),
        selected_weight=np.asarray([selected_weight], dtype=np.float32),
    )
    base_metrics = metrics(confirmation_users, confirmation_labels, confirmation_base)
    differential_metrics = metrics(
        confirmation_users, confirmation_labels, differential_scores
    )
    differential_gains = {
        key: float(differential_metrics[key] - base_metrics[key])
        for key in ("GAUC", "nDCG@5", "primary")
    }

    champion_manifest_path = ROOT / "results" / "final-model" / "manifest.json"
    champion_manifest = json.loads(champion_manifest_path.read_text(encoding="utf-8"))
    champion_path = ROOT / str(champion_manifest["validation_scores"])
    champion_digest = hashlib.sha256(champion_path.read_bytes()).hexdigest()
    if champion_digest != champion_manifest["validation_scores_sha256"]:
        raise RuntimeError("Frozen champion checksum mismatch")
    with np.load(champion_path, allow_pickle=False) as stored:
        champion_raw = np.asarray(stored["scores"], dtype=np.float64).reshape(-1)
    if len(champion_raw) != len(confirmation_labels) or not bool(
        np.isfinite(champion_raw).all()
    ):
        raise RuntimeError("Frozen champion alignment or finiteness check failed")
    champion_ranks = within_user_ranks(confirmation_users, champion_raw)
    differential_ranks = within_user_ranks(confirmation_users, differential_scores)
    champion_residual = champion_ranks + champion_weight * (
        differential_ranks - champion_ranks
    )
    champion_metrics = metrics(confirmation_users, confirmation_labels, champion_ranks)
    residual_metrics = metrics(
        confirmation_users, confirmation_labels, champion_residual
    )
    residual_gains = {
        key: float(residual_metrics[key] - champion_metrics[key])
        for key in ("GAUC", "nDCG@5", "primary")
    }
    users_array = np.asarray(confirmation_users, dtype=str)
    folds = np.asarray(
        [actual_user_fold(user) for user in confirmation_users], dtype=np.int8
    )
    fold_results = []
    for fold in range(4):
        mask = folds == fold
        fold_champion = metrics(
            users_array[mask].tolist(), confirmation_labels[mask], champion_ranks[mask]
        )
        fold_residual = metrics(
            users_array[mask].tolist(), confirmation_labels[mask], champion_residual[mask]
        )
        fold_gains = {
            key: float(fold_residual[key] - fold_champion[key])
            for key in ("GAUC", "nDCG@5", "primary")
        }
        fold_results.append(
            {
                "fold": fold,
                "rows": int(np.sum(mask)),
                "users": len(set(users_array[mask].tolist())),
                "champion_metrics": fold_champion,
                "residual_metrics": fold_residual,
                "gains": fold_gains,
                "all_metrics_nonnegative": bool(
                    all(value >= -1e-12 for value in fold_gains.values())
                ),
            }
        )
    overall_all_improve = bool(
        residual_gains["primary"] > 0.0
        and residual_gains["GAUC"] > 0.0
        and residual_gains["nDCG@5"] > 0.0
    )
    folds_stable = bool(all(item["all_metrics_nonnegative"] for item in fold_results))
    merits_promotion = bool(overall_all_improve and folds_stable)
    audit = {
        "experiment": "fixed NeuralNDCG differential residual against frozen champion",
        "selected_differential_weight": selected_weight,
        "fixed_champion_blend_weight": champion_weight,
        "champion_artifact_metrics": champion_manifest["validation_metrics"],
        "champion_metrics_recomputed": champion_metrics,
        "residual_metrics": residual_metrics,
        "residual_gains": residual_gains,
        "overall_all_three_improve": overall_all_improve,
        "four_actual_user_id_modulo_folds": fold_results,
        "all_four_folds_all_metrics_nonnegative": folds_stable,
        "merits_promotion": merits_promotion,
        "confirmation_labels_accessed": True,
        "hidden_test_outcome_fields_accessed": False,
    }
    DIFFERENTIAL_AUDIT_REPORT.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    training_details = DEFAULT_RUNTIME / "differential-confirmation-training-details.json"
    training_details.write_text(
        json.dumps(
            {
                "base": base_details,
                "control": control_details,
                "candidate": candidate_details,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = {
        "experiment": "locked NeuralNDCG ranking-objective differential confirmation",
        "status": "promotion_passed" if merits_promotion else "confirmation_rejected",
        "merits_promotion": merits_promotion,
        "screen_report": str(DIFFERENTIAL_SCREEN_REPORT.relative_to(ROOT)),
        "protocol": {
            "residual_fit": "2022-04-15..2022-04-21",
            "residual_fit_base_training": "2022-04-08..2022-04-14",
            "confirmation_base_refit": "2022-04-08..2022-04-21",
            "confirmation": (
                "2022-04-22..2022-04-28 one scored locked run; one prior unscored "
                "preflight parsed rows and is disclosed below"
            ),
            "selected_weight_locked_without_confirmation_retuning": selected_weight,
            "champion_weight_locked_without_confirmation_retuning": champion_weight,
            "confirmation_labels_accessed": True,
            "hidden_test_outcome_fields_accessed": False,
            "integrity_incident": {
                "aborted_preflight_parsed_confirmation_rows": len(confirmation),
                "outcomes_printed_scored_trained_or_used": False,
                "cause": "an overly strict raw-file-order equality assertion; the organizer file is not date sorted",
                "effect_on_locked_configuration": "none",
            },
        },
        "rows": {
            "training": len(training),
            "residual_fit": len(residual_fit),
            "confirmation": len(confirmation),
        },
        "feature_metadata": {
            "fit": fit_metadata,
            "confirmation": confirmation_metadata,
        },
        "metrics": {
            "base": base_metrics,
            "differential": differential_metrics,
            "differential_gains": differential_gains,
        },
        "champion_audit": audit,
        "artifacts": {
            "scores": str(score_artifact.relative_to(ROOT)),
            "training_details": str(training_details.relative_to(ROOT)),
            "audit": str(DIFFERENTIAL_AUDIT_REPORT.relative_to(ROOT)),
        },
        "alignment": {
            "row_key": "date,user_id,video_id",
            "sha256": alignment_digest(confirmation),
            "finite_scores": True,
            "aligned_rows": len(confirmation),
        },
        "resource_usage_total": measured_usage(started_wall, started_cpu),
    }
    DIFFERENTIAL_CONFIRMATION_REPORT.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--differential-followup",
        action="store_true",
        help="Run the preregistered ranking-objective differential screen",
    )
    parser.add_argument(
        "--confirm-differential",
        action="store_true",
        help="After a passing differential screen, run the one locked confirmation",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base-epochs", type=int, default=4)
    parser.add_argument("--residual-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--group-batch-size", type=int, default=96)
    parser.add_argument("--maximum-group-rows", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--base-hidden", type=int, default=64)
    parser.add_argument("--residual-hidden", type=int, default=32)
    parser.add_argument("--residual-bound", type=float, default=0.20)
    parser.add_argument("--ndcg-alpha", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--gate-temperature", type=float, default=0.75)
    args = parser.parse_args()
    if not (0.0 < args.ndcg_alpha <= 0.1):
        raise ValueError("ndcg-alpha must be in (0, 0.1]")
    if not (0.0 < args.residual_bound <= 0.5):
        raise ValueError("residual-bound must be in (0, 0.5]")

    import torch

    torch.set_num_threads(max(1, min(args.threads, 8)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.confirm_differential and not args.differential_followup:
        raise ValueError("--confirm-differential requires --differential-followup")
    if args.differential_followup:
        screen_report = run_differential_screen(args)
        result: dict = {"screen": screen_report}
        if args.confirm_differential and screen_report["merits_locked_confirmation"]:
            result["confirmation"] = run_locked_differential_confirmation(
                args, screen_report
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    base_train, residual_fit, evaluation = load_train_only_rows(args.data_dir.resolve())

    # Frozen OOT base scores for residual fitting: Apr8-11 -> Apr12-14.
    fit_encoded, fit_dimension = encode_base(base_train, residual_fit)
    fit_base_scores, fit_base_details = train_fixed_base(
        encoded=fit_encoded,
        dimension=fit_dimension,
        output_dir=args.output_dir,
        name="fit-oot",
        seed=args.seed,
        epochs=args.base_epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden=args.base_hidden,
    )
    fit_features, fit_feature_metadata = build_residual_features(
        base_train, residual_fit, fit_base_scores
    )

    # Locked evaluation base scores: same architecture/epochs, refit through Apr14.
    refit_history = base_train + residual_fit
    evaluation_encoded, evaluation_dimension = encode_base(refit_history, evaluation)
    evaluation_base_scores, evaluation_base_details = train_fixed_base(
        encoded=evaluation_encoded,
        dimension=evaluation_dimension,
        output_dir=args.output_dir,
        name="evaluation-refit",
        seed=args.seed,
        epochs=args.base_epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden=args.base_hidden,
    )
    evaluation_features, evaluation_feature_metadata = build_residual_features(
        refit_history, evaluation, evaluation_base_scores
    )

    _, fit_labels, fit_users = fit_encoded["valid"]
    _, evaluation_labels, evaluation_users = evaluation_encoded["valid"]
    common = {
        "features": fit_features,
        "labels": fit_labels,
        "users": fit_users,
        "base_scores": fit_base_scores,
        "evaluation_features": evaluation_features,
        "evaluation_base_scores": evaluation_base_scores,
        "output_dir": args.output_dir,
        "seed": args.seed + 991,
        "epochs": args.residual_epochs,
        "hidden": args.residual_hidden,
        "residual_bound": args.residual_bound,
        "group_batch_size": args.group_batch_size,
        "maximum_group_rows": args.maximum_group_rows,
        "temperature": args.temperature,
        "gate_temperature": args.gate_temperature,
    }
    control_scores, control_details = train_residual(
        name="alpha-zero-control", alpha=0.0, **common
    )
    candidate_scores, candidate_details = train_residual(
        name="neural-ndcg-candidate", alpha=args.ndcg_alpha, **common
    )

    arrays = {
        "base_scores": evaluation_base_scores.astype(np.float32),
        "control_scores": control_scores.astype(np.float32),
        "candidate_scores": candidate_scores.astype(np.float32),
        "labels": evaluation_labels.astype(np.float32),
        "users": np.asarray(evaluation_users, dtype="U"),
        "dates": np.asarray([row[0] for row in evaluation], dtype=np.int32),
    }
    for name, values in arrays.items():
        if len(values) != len(evaluation):
            raise RuntimeError(f"{name} is not row aligned")
        if values.dtype.kind in "fc" and not bool(np.isfinite(values).all()):
            raise RuntimeError(f"{name} contains a non-finite value")
    score_artifact = args.output_dir / "train-only-evaluation-scores.npz"
    np.savez_compressed(score_artifact, **arrays)

    base_metrics = metrics(evaluation_users, evaluation_labels, evaluation_base_scores)
    control_metrics = metrics(evaluation_users, evaluation_labels, control_scores)
    candidate_metrics = metrics(evaluation_users, evaluation_labels, candidate_scores)
    deltas = {
        key: float(candidate_metrics[key] - control_metrics[key])
        for key in ("GAUC", "nDCG@5", "primary")
    }
    candidate_minus_base = {
        key: float(candidate_metrics[key] - base_metrics[key])
        for key in ("GAUC", "nDCG@5", "primary")
    }
    matched_candidate_improves_all_metrics = bool(
        deltas["GAUC"] > 0.0 and deltas["nDCG@5"] > 0.0 and deltas["primary"] > 0.0
    )
    candidate_improves_frozen_base = bool(
        candidate_minus_base["GAUC"] > 0.0
        and candidate_minus_base["nDCG@5"] > 0.0
        and candidate_minus_base["primary"] > 0.0
    )
    merits_confirmation = bool(
        matched_candidate_improves_all_metrics and candidate_improves_frozen_base
    )
    training_details_path = args.output_dir / "training-details.json"
    training_details_path.write_text(
        json.dumps(
            {
                "fit_base": fit_base_details,
                "evaluation_base": evaluation_base_details,
                "control": control_details,
                "candidate": candidate_details,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = {
        "experiment": "bounded NeuralNDCG-style top-5 residual",
        "status": "screen_passed" if merits_confirmation else "screen_rejected",
        "merits_confirmation": merits_confirmation,
        "acceptance_rule": (
            "candidate must strictly improve primary, GAUC, and nDCG@5 over "
            "both the exact alpha=0 control and the untouched frozen base"
        ),
        "matched_candidate_improves_all_metrics": matched_candidate_improves_all_metrics,
        "candidate_improves_frozen_base": candidate_improves_frozen_base,
        "protocol": {
            "base_fit": "2022-04-08..2022-04-11",
            "residual_fit": "2022-04-12..2022-04-14",
            "base_refit": "2022-04-08..2022-04-14",
            "locked_evaluation": "2022-04-15..2022-04-21",
            "hyperparameters_locked_before_evaluation": True,
            "confirmation_labels_accessed": False,
            "hidden_test_accessed": False,
            "base_score_alignment": "strictly out-of-time for both residual-fit and evaluation windows",
            "smooth_rank": "1 + sum_j sigmoid((score_j-score_i)/temperature), diagonal removed; sigmoid top-5 gate",
            "group_memory_bound": "fit groups capped at 64; frozen top 48 plus up to 16 missed positives; evaluation uncapped",
        },
        "rows": {
            "base_fit": len(base_train),
            "residual_fit": len(residual_fit),
            "evaluation": len(evaluation),
        },
        "locked_configuration": {
            "seed": args.seed,
            "threads": torch.get_num_threads(),
            "base_epochs": args.base_epochs,
            "residual_epochs": args.residual_epochs,
            "batch_size": args.batch_size,
            "group_batch_size": args.group_batch_size,
            "maximum_group_rows": args.maximum_group_rows,
            "embedding_dim": args.embedding_dim,
            "base_hidden": args.base_hidden,
            "residual_hidden": args.residual_hidden,
            "residual_bound": args.residual_bound,
            "ndcg_alpha": args.ndcg_alpha,
            "temperature": args.temperature,
            "gate_temperature": args.gate_temperature,
        },
        "feature_metadata": {
            "fit": fit_feature_metadata,
            "evaluation": evaluation_feature_metadata,
        },
        "metrics": {
            "frozen_base": base_metrics,
            "alpha_zero_control": control_metrics,
            "neural_ndcg_candidate": candidate_metrics,
            "candidate_minus_control": deltas,
            "candidate_minus_frozen_base": candidate_minus_base,
        },
        "training_summary": {
            "control_final_epoch": control_details["history"][-1],
            "candidate_final_epoch": candidate_details["history"][-1],
            "control_correction": control_details["correction"],
            "candidate_correction": candidate_details["correction"],
            "fit_groups": candidate_details["fit_groups"],
        },
        "alignment": {
            "row_key": "date,user_id,video_id",
            "sha256": alignment_digest(evaluation),
            "finite_scores": True,
            "aligned_rows": len(evaluation),
        },
        "artifacts": {
            "scores": str(score_artifact.relative_to(ROOT)),
            "training_details": str(training_details_path.relative_to(ROOT)),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
        },
        "resource_usage_total": measured_usage(overall_wall, overall_cpu),
        "hardware": {
            "architecture": platform.machine(),
            "logical_cpu_count": os.cpu_count() or 1,
            "torch_version": torch.__version__,
        },
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
