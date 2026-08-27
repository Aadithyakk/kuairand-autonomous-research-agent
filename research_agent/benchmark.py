from __future__ import annotations

import json
import math
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from .core import write_json_atomic


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def evaluate_ranking(rows: list[dict[str, Any]], scores: list[float], k: int = 5) -> dict[str, float]:
    if len(rows) != len(scores):
        raise ValueError("Prediction count does not match validation rows")
    grouped: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise ValueError("Every prediction must be a finite number")
        grouped[str(row["user_id"])].append((int(row["label"]), float(score), int(row["row_id"])))

    auc_weighted_sum = 0.0
    auc_weight = 0
    ndcgs: list[float] = []
    for entries in grouped.values():
        labels = [entry[0] for entry in entries]
        user_scores = [entry[1] for entry in entries]
        positives = sum(labels)
        negatives = len(labels) - positives
        if positives and negatives:
            ranks = _rankdata(user_scores)
            positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
            auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
            auc_weighted_sum += auc * positives
            auc_weight += positives

        ranked = sorted(entries, key=lambda entry: (-entry[1], entry[2]))[:k]
        dcg = sum(label / math.log2(position + 2) for position, (label, _, _) in enumerate(ranked))
        ideal_labels = sorted(labels, reverse=True)[:k]
        idcg = sum(label / math.log2(position + 2) for position, label in enumerate(ideal_labels))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    gauc = auc_weighted_sum / auc_weight if auc_weight else 0.0
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    return {"GAUC": round(gauc, 6), f"nDCG@{k}": round(ndcg, 6), "primary": round((gauc + ndcg) / 2.0, 6)}


class Benchmark(ABC):
    """Immutable boundary between model-authored code and private evaluation labels."""

    name: str
    label: str
    metrics: list[str]

    @abstractmethod
    def initialize(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def public_context(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_experiment(self, experiment_dir: Path) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_predictions(self, prediction_path: Path) -> dict[str, float]:
        raise NotImplementedError


class ToyRankingBenchmark(Benchmark):
    """Small ranking benchmark used to validate the autonomous method itself.

    Validation labels live outside experiment workspaces. Generated experiment
    code sees training labels and validation features, exactly as a competition
    submission sees public development data and an external evaluator.
    """

    name = "Controlled Ranking Validation"
    label = "long_view"
    metrics = ["GAUC", "nDCG@5", "primary"]

    def __init__(self, root: Path):
        self.root = root
        self.private_dir = root / "private"
        self.public_dir = root / "public"

    def initialize(self) -> dict[str, Any]:
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.public_dir.mkdir(parents=True, exist_ok=True)
        train, validation = self._make_rows()
        validation_features = [{key: value for key, value in row.items() if key != "label"} for row in validation]
        write_json_atomic(self.public_dir / "train.json", train)
        write_json_atomic(self.public_dir / "validation.json", validation_features)
        write_json_atomic(self.private_dir / "validation_labels.json", validation)
        baseline_predictions = self._baseline_predictions(train, validation_features)
        write_json_atomic(self.private_dir / "baseline_predictions.json", baseline_predictions)
        metrics = evaluate_ranking(validation, [row["score"] for row in baseline_predictions], k=5)
        return {
            "title": "Item-popularity baseline",
            "metrics": metrics,
            "status": "passed",
            "artifact": str(self.private_dir / "baseline_predictions.json"),
        }

    def public_context(self) -> dict[str, Any]:
        train = json.loads((self.public_dir / "train.json").read_text(encoding="utf-8"))
        valid = json.loads((self.public_dir / "validation.json").read_text(encoding="utf-8"))
        users = sorted({row["user_id"] for row in train})
        categories = sorted({row["category"] for row in train})
        return {
            "benchmark": self.name,
            "task": "Rank the exposed candidate items independently within each user.",
            "label": self.label,
            "metrics": self.metrics,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "users": len(users),
            "schema": {
                "row_id": "integer alignment key",
                "user_id": "categorical user identifier",
                "item_id": "categorical item identifier",
                "category": "categorical content category",
                "position": "integer exposure position",
                "time_index": "integer chronological index",
                "long_view": "binary training-only label",
            },
            "observations": [
                "The item-popularity baseline cannot represent user-specific category preferences.",
                "Every validation user appears in training.",
                "Validation item IDs are unseen, so transferable interaction features matter.",
                "The evaluator groups predictions by user and rewards top-five ordering.",
            ],
            "constraints": [
                "Generated code may read only data/train.json and data/validation.json.",
                "Generated code must write predictions.json with row_id and score.",
                "Validation labels and evaluator implementation are private.",
                "No network, subprocess, dynamic code execution, or external files.",
            ],
            "program_contract": {
                "language": "Python 3 standard library only",
                "train_path": "data/train.json",
                "validation_path": "data/validation.json",
                "output_path": "predictions.json",
                "output_schema": "ordered JSON list of {row_id, score}",
            },
        }

    def prepare_experiment(self, experiment_dir: Path) -> dict[str, str]:
        data_dir = experiment_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.public_dir / "train.json", data_dir / "train.json")
        shutil.copy2(self.public_dir / "validation.json", data_dir / "validation.json")
        return {
            "train": "data/train.json",
            "validation": "data/validation.json",
            "predictions": "predictions.json",
        }

    def evaluate_predictions(self, prediction_path: Path) -> dict[str, float]:
        if not prediction_path.exists():
            raise ValueError("Experiment did not produce predictions.json")
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        validation = json.loads((self.private_dir / "validation_labels.json").read_text(encoding="utf-8"))
        if not isinstance(predictions, list) or len(predictions) != len(validation):
            raise ValueError("Predictions must contain exactly one record per validation row")
        expected_ids = [row["row_id"] for row in validation]
        received_ids = [row.get("row_id") for row in predictions]
        if received_ids != expected_ids:
            raise ValueError("Prediction row_id values are missing, duplicated, or misaligned")
        scores = [row.get("score") for row in predictions]
        return evaluate_ranking(validation, scores, k=5)

    @staticmethod
    def _make_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        preferences = {
            "u0": "sports", "u1": "technology", "u2": "music",
            "u3": "food", "u4": "sports", "u5": "technology",
        }
        categories = ["sports", "technology", "music", "food"]
        train: list[dict[str, Any]] = []
        row_id = 0
        for user, preferred in preferences.items():
            for cycle in range(4):
                for category in categories:
                    train.append({
                        "row_id": row_id,
                        "user_id": user,
                        "item_id": f"train-{cycle}-{category}",
                        "category": category,
                        "position": categories.index(category),
                        "time_index": cycle * len(categories) + categories.index(category),
                        "long_view": int(category == preferred),
                    })
                    row_id += 1

        validation: list[dict[str, Any]] = []
        row_id = 0
        for user, preferred in preferences.items():
            alternatives = [category for category in categories if category != preferred]
            sequence = [alternatives[0], preferred, alternatives[1], alternatives[2], preferred, alternatives[0]]
            for position, category in enumerate(sequence):
                validation.append({
                    "row_id": row_id,
                    "user_id": user,
                    "item_id": f"valid-{user}-{position}",
                    "category": category,
                    "position": position,
                    "time_index": 100 + position,
                    "label": int(category == preferred),
                })
                row_id += 1
        return train, validation

    @staticmethod
    def _baseline_predictions(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sums: dict[str, int] = defaultdict(int)
        counts: dict[str, int] = defaultdict(int)
        global_rate = sum(row["long_view"] for row in train) / len(train)
        for row in train:
            sums[row["item_id"]] += row["long_view"]
            counts[row["item_id"]] += 1
        return [
            {
                "row_id": row["row_id"],
                "score": (sums[row["item_id"]] + 5 * global_rate) / (counts[row["item_id"]] + 5),
            }
            for row in validation
        ]
