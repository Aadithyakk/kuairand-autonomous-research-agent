"""Trusted, label-boundary-safe building blocks available to generated experiments."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_frames(train_columns: list[str], validation_columns: list[str]):
    train_path = Path("data/train.parquet")
    validation_path = Path("data/validation.parquet")
    train = pd.read_parquet(train_path, columns=train_columns)
    validation = pd.read_parquet(validation_path, columns=validation_columns)
    if "long_view" not in train.columns or "long_view" in validation.columns:
        raise ValueError("Label boundary violation")
    if "row_id" in train.columns or "row_id" not in validation.columns or not validation["row_id"].is_unique:
        raise ValueError("row_id contract violation")
    return train, validation


def chronological_split(frame: pd.DataFrame, holdout_dates: int = 2):
    dates = np.sort(pd.to_numeric(frame["date"], errors="raise").unique())
    if len(dates) <= holdout_dates:
        raise ValueError("Insufficient unique dates for chronological holdout")
    boundary = dates[-holdout_dates]
    fit = pd.to_numeric(frame["date"], errors="raise").to_numpy() < boundary
    holdout = ~fit
    if not fit.any() or not holdout.any():
        raise ValueError("Chronological split is empty")
    return fit, holdout


def save_predictions(scores, validation: pd.DataFrame, path: str = "predictions.npy"):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(validation) or not np.isfinite(values).all():
        raise ValueError("Predictions must be finite and aligned one-to-one with validation rows")
    if "row_id" not in validation or not validation["row_id"].is_unique:
        raise ValueError("Validation row_id is missing or non-unique")
    np.save(path, values)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


class TrustedFM:
    def __init__(self, dimension: int, factors: int = 16, learning_rate: float = 0.001, l2: float = 1e-6, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.v = rng.normal(0, 0.01, (dimension, factors)).astype(np.float32)
        self.w = np.zeros(dimension, dtype=np.float32)
        self.bias = np.float32(0.0)
        self.learning_rate, self.l2 = learning_rate, l2
        self.mv, self.vv = np.zeros_like(self.v), np.zeros_like(self.v)
        self.mw, self.vw = np.zeros_like(self.w), np.zeros_like(self.w)
        self.step_number = 0

    def logits(self, matrix):
        embeddings = self.v[matrix]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * ((summed ** 2).sum(axis=1) - (embeddings ** 2).sum(axis=(1, 2)))
        return self.bias + self.w[matrix].sum(axis=1) + interaction, embeddings, summed

    def step(self, matrix, labels):
        logits, embeddings, summed = self.logits(matrix)
        gradient = ((sigmoid(logits) - labels) / len(labels)).astype(np.float32)
        grad_v, grad_w = np.zeros_like(self.v), np.zeros_like(self.w)
        np.add.at(grad_w, matrix, gradient[:, None])
        np.add.at(grad_v, matrix, gradient[:, None, None] * (summed[:, None, :] - embeddings))
        grad_v += self.l2 * self.v
        grad_w += self.l2 * self.w
        self.step_number += 1
        for parameter, grad, first, second in ((self.v, grad_v, self.mv, self.vv), (self.w, grad_w, self.mw, self.vw)):
            first *= 0.9
            first += 0.1 * grad
            second *= 0.999
            second += 0.001 * grad * grad
            parameter -= self.learning_rate * (first / (1 - 0.9 ** self.step_number)) / (np.sqrt(second / (1 - 0.999 ** self.step_number)) + 1e-8)
        self.bias -= self.learning_rate * gradient.sum()

    def predict(self, matrix, batch_size: int = 200_000):
        return np.concatenate([self.logits(matrix[start:start + batch_size])[0] for start in range(0, len(matrix), batch_size)])


def encode_fm(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]):
    offsets, mappings, offset = [], [], 0
    train_values = train[columns].fillna("UNK").astype(str)
    validation_values = validation[columns].fillna("UNK").astype(str)
    for column in columns:
        values = pd.Index(train_values[column].unique())
        mapping = {value: index for index, value in enumerate(values)}
        mappings.append(mapping)
        offsets.append(offset)
        offset += len(mapping) + 1

    def transform(frame):
        matrix = np.empty((len(frame), len(columns)), dtype=np.int32)
        for index, column in enumerate(columns):
            unknown = len(mappings[index])
            matrix[:, index] = frame[column].map(mappings[index]).fillna(unknown).astype(np.int32) + offsets[index]
        return matrix

    return transform(train_values), transform(validation_values), offset
