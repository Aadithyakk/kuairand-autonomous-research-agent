#!/usr/bin/env python3
"""Reproduce KuaiLab's original champion and create a blind submission.

The final submission path reads only the permitted feature columns from the
2022-04-29..2022-05-08 rows. It never loads or evaluates their outcome columns.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


TRAIN_RANGE = (20220408, 20220421)
VALID_RANGE = (20220422, 20220428)
TEST_RANGE = (20220429, 20220508)
EXPECTED_ROWS = {"train": 1_141_112, "validation": 124_909, "test": 170_588}
FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
CHAMPION = {
    "k": 16,
    "lr": 0.001,
    "l2": 1e-6,
    "epochs": 8,
    "batch_size": 8192,
    "seed": 0,
    "positive_weight": 2.75,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def video_authors(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result[row["video_id"]] = row["author_id"]
    return result


def load_train(data_dir: Path, authors: dict[str, str]) -> list[tuple]:
    """Load the only rows permitted to enter model training."""
    rows: list[tuple] = []
    path = data_dir / "log_standard_4_08_to_4_21_pure.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            if TRAIN_RANGE[0] <= date <= TRAIN_RANGE[1]:
                rows.append((date, row["user_id"], row["video_id"], authors.get(row["video_id"], "UNK"),
                             row["tab"], float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0))
    require_count("train", rows)
    return rows


def load_validation(data_dir: Path, authors: dict[str, str]) -> list[tuple]:
    """Load validation labels for the reproducibility check only."""
    rows: list[tuple] = []
    path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            date = int(row["date"])
            if VALID_RANGE[0] <= date <= VALID_RANGE[1]:
                rows.append((date, row["user_id"], row["video_id"], authors.get(row["video_id"], "UNK"),
                             row["tab"], float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0))
    require_count("validation", rows)
    return rows


def load_blind_test(data_dir: Path, authors: dict[str, str]) -> list[tuple]:
    """Load test features by column index without reading any outcome value."""
    path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
    rows: list[tuple] = []
    permitted = ("user_id", "video_id", "date", "duration_ms", "tab")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {name: header.index(name) for name in permitted}
        for raw in reader:
            date = int(raw[indices["date"]])
            if TEST_RANGE[0] <= date <= TEST_RANGE[1]:
                user = raw[indices["user_id"]]
                video = raw[indices["video_id"]]
                rows.append((date, user, video, authors.get(video, "UNK"), raw[indices["tab"]],
                             float(raw[indices["duration_ms"]])))
    require_count("test", rows)
    return rows


def require_count(name: str, rows: list[tuple]) -> None:
    expected = EXPECTED_ROWS[name]
    if len(rows) != expected:
        raise RuntimeError(f"Unexpected {name} size: {len(rows):,}; expected {expected:,}")


class Encoder:
    def __init__(self, train: list[tuple]):
        self.edges = np.quantile(np.asarray([row[5] for row in train]), np.linspace(0, 1, 11)[1:-1])
        self.vocabs: list[dict[str, int]] = [dict() for _ in FIELDS]
        for row in train:
            for index, value in enumerate(self.raw(row)):
                if value not in self.vocabs[index]:
                    self.vocabs[index][value] = len(self.vocabs[index])
        self.unknown = [len(vocab) for vocab in self.vocabs]
        self.field_dims = [len(vocab) + 1 for vocab in self.vocabs]
        self.offsets = np.cumsum([0, *self.field_dims[:-1]]).astype(np.int32)

    def raw(self, row: tuple) -> tuple[str, ...]:
        return (row[1], row[2], row[3], row[4], str(int(np.searchsorted(self.edges, row[5]))))

    def transform(self, rows: list[tuple]) -> np.ndarray:
        matrix = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for row_number, row in enumerate(rows):
            for field, value in enumerate(self.raw(row)):
                matrix[row_number, field] = self.vocabs[field].get(value, self.unknown[field]) + self.offsets[field]
        return matrix

    @property
    def dimension(self) -> int:
        return int(sum(self.field_dims))

    def save(self, path: Path) -> None:
        payload = {
            "fields": FIELDS,
            "edges": self.edges.tolist(),
            "vocabs": self.vocabs,
            "unknown": self.unknown,
            "field_dims": self.field_dims,
            "offsets": self.offsets.tolist(),
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class WeightedFM:
    """The original sealed NumPy FM with the iteration-4 class weight."""
    def __init__(self, dimension: int):
        rng = np.random.default_rng(CHAMPION["seed"])
        self.V = rng.normal(0, 0.01, (dimension, CHAMPION["k"])).astype(np.float32)
        self.W = np.zeros(dimension, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV, self.vV = np.zeros_like(self.V), np.zeros_like(self.V)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray):
        embeddings = self.V[X]
        summed = embeddings.sum(1)
        interaction = 0.5 * ((summed ** 2).sum(1) - (embeddings ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + interaction, embeddings, summed

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        logits, embeddings, summed = self.logits(X)
        weights = np.where(y > 0.5, CHAMPION["positive_weight"], 1.0).astype(np.float32)
        gradient = ((sigmoid(logits) - y) * weights / weights.sum()).astype(np.float32)
        grad_v, grad_w = np.zeros_like(self.V), np.zeros_like(self.W)
        np.add.at(grad_w, X, gradient[:, None])
        np.add.at(grad_v, X, gradient[:, None, None] * (summed[:, None, :] - embeddings))
        grad_v += CHAMPION["l2"] * self.V
        grad_w += CHAMPION["l2"] * self.W
        self.t += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, grad, mean, variance in (
            (self.V, grad_v, self.mV, self.vV), (self.W, grad_w, self.mW, self.vW)
        ):
            mean *= beta1
            mean += (1 - beta1) * grad
            variance *= beta2
            variance += (1 - beta2) * (grad * grad)
            parameter -= CHAMPION["lr"] * (mean / (1 - beta1 ** self.t)) / (
                np.sqrt(variance / (1 - beta2 ** self.t)) + epsilon
            )
        self.b -= CHAMPION["lr"] * gradient.sum()
        probabilities = sigmoid(logits)
        loss = -(weights * (y * np.log(probabilities + 1e-9) + (1 - y) * np.log(1 - probabilities + 1e-9))).sum()
        return float(loss / weights.sum())

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate([self.logits(X[start:start + batch_size])[0]
                               for start in range(0, len(X), batch_size)])


def train_model(train_x: np.ndarray, train_y: np.ndarray, dimension: int) -> tuple[WeightedFM, list[float]]:
    model = WeightedFM(dimension)
    random = np.random.default_rng(CHAMPION["seed"])
    losses: list[float] = []
    for epoch in range(1, CHAMPION["epochs"] + 1):
        order = random.permutation(len(train_y))
        batch_losses = []
        for start in range(0, len(order), CHAMPION["batch_size"]):
            indices = order[start:start + CHAMPION["batch_size"]]
            batch_losses.append(model.step(train_x[indices], train_y[indices]))
        losses.append(float(np.mean(batch_losses)))
        print(f"epoch {epoch}/{CHAMPION['epochs']} loss={losses[-1]:.6f}", flush=True)
    return model, losses


def save_artifacts(output_dir: Path, model: WeightedFM, encoder: Encoder, losses: list[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "original-fm-checkpoint.npz", V=model.V, W=model.W, b=model.b)
    encoder.save(output_dir / "encoding.json")
    (output_dir / "training.json").write_text(json.dumps({"config": CHAMPION, "losses": losses}, indent=2), encoding="utf-8")


def reproduce_validation(data_dir: Path, starter_dir: Path, output_dir: Path) -> None:
    started = time.monotonic()
    authors = video_authors(data_dir)
    train, validation = load_train(data_dir, authors), load_validation(data_dir, authors)
    encoder = Encoder(train)
    train_x, valid_x = encoder.transform(train), encoder.transform(validation)
    train_y = np.asarray([row[6] for row in train], dtype=np.float32)
    valid_y = np.asarray([row[6] for row in validation], dtype=np.float32)
    model, losses = train_model(train_x, train_y, encoder.dimension)
    evaluate = load_module("kuailab_official_evaluate", starter_dir / "evaluate.py").evaluate
    metrics = evaluate([row[1] for row in validation], valid_y, model.predict(valid_x))
    result = {
        "split": "validation",
        "GAUC": float(metrics["GAUC"]),
        "nDCG@5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
        "rows": int(metrics["rows"]),
        "users": int(metrics["users"]),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    save_artifacts(output_dir, model, encoder, losses)
    (output_dir / "validation-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def make_submission(data_dir: Path, output_dir: Path) -> None:
    started = time.monotonic()
    authors = video_authors(data_dir)
    train = load_train(data_dir, authors)
    test = load_blind_test(data_dir, authors)
    encoder = Encoder(train)
    train_x, test_x = encoder.transform(train), encoder.transform(test)
    train_y = np.asarray([row[6] for row in train], dtype=np.float32)
    model, losses = train_model(train_x, train_y, encoder.dimension)
    scores = model.predict(test_x)
    if len(scores) != EXPECTED_ROWS["test"] or not np.isfinite(scores).all():
        raise RuntimeError("Submission scores are incomplete or non-finite")
    save_artifacts(output_dir, model, encoder, losses)
    submission = output_dir / "submission.csv"
    with submission.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        for row_id, (row, score) in enumerate(zip(test, scores, strict=True)):
            writer.writerow((row_id, row[1], row[2], f"{float(score):.9g}"))
    manifest = {
        "model": "original KuaiLab weighted FM champion",
        "config": CHAMPION,
        "training_dates": list(TRAIN_RANGE),
        "prediction_dates": list(TEST_RANGE),
        "rows": len(test),
        "schema": ["row_id", "user_id", "video_id", "score"],
        "test_labels_loaded": False,
        "test_labels_scored": False,
        "sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    (output_dir / "submission-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def check_submission(data_dir: Path, submission: Path) -> None:
    test = load_blind_test(data_dir, video_authors(data_dir))
    with submission.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header != ["row_id", "user_id", "video_id", "score"]:
            raise RuntimeError(f"Wrong header: {header}")
        count = 0
        for expected_id, (record, source) in enumerate(zip(reader, test, strict=True)):
            if len(record) != 4 or int(record[0]) != expected_id:
                raise RuntimeError(f"Bad row_id or width at row {expected_id}")
            if record[1] != source[1] or record[2] != source[2]:
                raise RuntimeError(f"Source alignment mismatch at row {expected_id}")
            if not math.isfinite(float(record[3])):
                raise RuntimeError(f"Non-finite score at row {expected_id}")
            count += 1
    if count != EXPECTED_ROWS["test"]:
        raise RuntimeError(f"Wrong row count: {count:,}")
    print(f"PASS: {submission} has the required schema and {count:,} aligned finite scores")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="KuaiRand-Pure data/ directory")
    parser.add_argument("--starter-dir", type=Path, default=root / "external" / "kuairand-starter-kit")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "final-model")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("reproduce-validation")
    make = subparsers.add_parser("make-submission")
    make.add_argument("--confirm-final", action="store_true", help="confirm this is the one-time blind export")
    check = subparsers.add_parser("check-submission")
    check.add_argument("--submission", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "reproduce-validation":
        reproduce_validation(arguments.data_dir.resolve(), arguments.starter_dir.resolve(), arguments.output_dir.resolve())
    elif arguments.command == "make-submission":
        if not arguments.confirm_final:
            parser.error("make-submission requires --confirm-final")
        make_submission(arguments.data_dir.resolve(), arguments.output_dir.resolve())
    else:
        target = arguments.submission or (arguments.output_dir / "submission.csv")
        check_submission(arguments.data_dir.resolve(), target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
