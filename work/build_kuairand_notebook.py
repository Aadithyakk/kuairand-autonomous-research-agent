import json
from pathlib import Path
from textwrap import dedent


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip() + "\n",
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


cells = [
md(r'''
# KuaiRand Autonomous Recommender Research Lab

**Kaggle-ready, train/validation-only notebook**

This notebook implements the complete model ladder I would use for Track 2:

1. item-popularity sanity baseline;
2. organizer-style NumPy Factorization Machine;
3. leakage-safe temporal and affinity features;
4. LightGBM pointwise binary ranking;
5. LightGBM LambdaRank;
6. CatBoost pairwise ranking;
7. PyTorch BPR matrix factorization;
8. a compact candidate-aware DIN-style sequence model;
9. within-user rank ensembling and automatic blend search;
10. convergence checks, experiment logs, and submission creation.

The default task follows the **executable starter kit**: predict `long_view`, evaluate GAUC and nDCG@5, and optimize their mean. A configuration switch supports the older `click` + nDCG@10/Recall@50 wording while awaiting organizer confirmation.

### Leakage policy

- Model selection uses only the official training and validation dates.
- Validation outcomes never enter feature construction.
- Same-row watch time, click, like, follow, and other post-exposure outcomes are never inference features.
- Public test outcomes are never loaded for scoring or training.
- Month-level video statistics and random-exposure outcomes are excluded until organizers confirm their permitted use.
'''),
code(r'''
# Kaggle setup. Internet is needed only if the dataset is not attached as a Kaggle Dataset.
import importlib.util
import subprocess
import sys

REQUIRED = {
    "lightgbm": "lightgbm>=4.1",
    "catboost": "catboost>=1.2",
}
missing = [pkg for pkg in REQUIRED if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + [REQUIRED[p] for p in missing])

import collections
import copy
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import tarfile
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
from catboost import CatBoostRanker, Pool

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

print("Python", sys.version.split()[0])
print("pandas", pd.__version__, "LightGBM", lgb.__version__, "torch", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
'''),
code(r'''
@dataclass
class TaskSpec:
    # Executable starter-kit contract. Change `mode` below only if organizers confirm the older wording.
    mode: str = "starter_kit"  # starter_kit | legacy_text
    label: str = "long_view"
    ndcg_k: int = 5
    recall_k: int | None = None
    metrics: tuple = ("GAUC", "nDCG@5")
    primary_name: str = "primary"

    @classmethod
    def from_mode(cls, mode: str):
        if mode == "starter_kit":
            return cls(mode=mode)
        if mode == "legacy_text":
            return cls(
                mode=mode,
                label="is_click",
                ndcg_k=10,
                recall_k=50,
                metrics=("nDCG@10", "Recall@50"),
            )
        raise ValueError(f"Unknown task mode: {mode}")


@dataclass
class Config:
    seed: int = 2026
    task_mode: str = "starter_kit"
    train_start: int = 20220408
    train_end: int = 20220421
    valid_start: int = 20220422
    valid_end: int = 20220428
    fast_mode: bool = False
    fast_train_rows: int = 300_000
    fast_valid_rows: int = 60_000
    prior_strength: float = 20.0
    max_deep_train_rows: int = 700_000
    max_history: int = 40
    experiment_log: str = "/kaggle/working/kuairand_experiments.jsonl"
    # Recommended run ladder. All approaches are implemented below.
    run_fm: bool = True
    run_lgb_binary: bool = True
    run_lgb_ranker: bool = True
    run_catboost_ranker: bool = False  # enable after LightGBM champion is established
    run_bpr: bool = True
    run_din: bool = False              # enable on a Kaggle GPU after the cheap models


CFG = Config()
TASK = TaskSpec.from_mode(CFG.task_mode)

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

print("Task:", asdict(TASK))
print("Config:", asdict(CFG))
'''),
md(r'''
## 1. Locate or download KuaiRand-Pure

Preferred Kaggle setup: attach a private/public Kaggle Dataset containing the extracted `KuaiRand-Pure/data` directory. If it is not attached, the cell downloads the official archive to `/kaggle/working` when internet access is enabled.
'''),
code(r'''
DATA_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
EXPECTED_MD5 = "0820331067a3784d9691136f772b35a7"
REQUIRED_FILES = {
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "video_features_basic_pure.csv",
    "user_features_pure.csv",
}


def is_data_dir(path: Path) -> bool:
    return path.is_dir() and REQUIRED_FILES.issubset({p.name for p in path.iterdir()})


def locate_data_dir() -> Path | None:
    candidates = [
        Path("/kaggle/input/kuairand-pure/KuaiRand-Pure/data"),
        Path("/kaggle/input/kuairand-pure/data"),
        Path("/kaggle/working/KuaiRand-Pure/data"),
        Path("./KuaiRand-Pure/data"),
    ]
    for path in candidates:
        if is_data_dir(path):
            return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for path in kaggle_input.rglob("log_standard_4_08_to_4_21_pure.csv"):
            if is_data_dir(path.parent):
                return path.parent
    return None


def md5sum(path: Path, block_size=2**20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


DATA_DIR = locate_data_dir()
if DATA_DIR is None:
    archive = Path("/kaggle/working/KuaiRand-Pure.tar.gz")
    if not archive.exists():
        print("Dataset not attached; downloading the official archive...")
        urllib.request.urlretrieve(DATA_URL, archive)
    actual_md5 = md5sum(archive)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(f"Archive checksum mismatch: {actual_md5}")
    with tarfile.open(archive, "r:gz") as tf:
        try:
            tf.extractall("/kaggle/working", filter="data")
        except TypeError:  # Python <3.12; archive integrity is pinned by the organizer MD5.
            tf.extractall("/kaggle/working")
    DATA_DIR = locate_data_dir()

if DATA_DIR is None:
    raise FileNotFoundError("Could not locate KuaiRand-Pure/data")
print("Using", DATA_DIR)
'''),
md(r'''
## 2. Load only the official training and validation windows

We preserve source-row order for future submission alignment. The later standard log is filtered to validation dates only; rows after 28 April are deliberately discarded before modeling.
'''),
code(r'''
LOG_EARLY = DATA_DIR / "log_standard_4_08_to_4_21_pure.csv"
LOG_LATE = DATA_DIR / "log_standard_4_22_to_5_08_pure.csv"
VIDEO_FILE = DATA_DIR / "video_features_basic_pure.csv"
USER_FILE = DATA_DIR / "user_features_pure.csv"

video = pd.read_csv(VIDEO_FILE, low_memory=False)
user = pd.read_csv(USER_FILE, low_memory=False)

early = pd.read_csv(LOG_EARLY, low_memory=False)
late = pd.read_csv(LOG_LATE, low_memory=False)
early["_source"] = "early"
late["_source"] = "late"
early["_source_row"] = np.arange(len(early), dtype=np.int64)
late["_source_row"] = np.arange(len(late), dtype=np.int64)

all_dev = pd.concat([early, late], ignore_index=True)
all_dev["date"] = pd.to_numeric(all_dev["date"], errors="raise").astype(np.int32)
all_dev = all_dev[all_dev["date"].between(CFG.train_start, CFG.valid_end)].copy()

if TASK.label not in all_dev.columns:
    raise KeyError(f"Task label {TASK.label!r} not found")
all_dev[TASK.label] = (pd.to_numeric(all_dev[TASK.label], errors="coerce").fillna(0) != 0).astype(np.int8)

# Basic item metadata only. Month-level statistic features are intentionally excluded pending organizer approval.
keep_video = [c for c in [
    "video_id", "author_id", "video_type", "upload_dt", "upload_type",
    "visible_status", "video_duration", "server_width", "server_height",
    "music_id", "music_type", "tag",
] if c in video.columns]
all_dev = all_dev.merge(video[keep_video], on="video_id", how="left", validate="many_to_one")

train_df = all_dev[all_dev["date"].between(CFG.train_start, CFG.train_end)].copy()
valid_df = all_dev[all_dev["date"].between(CFG.valid_start, CFG.valid_end)].copy()

if CFG.fast_mode:
    train_df = train_df.sort_values("time_ms").tail(CFG.fast_train_rows).copy()
    valid_df = valid_df.sort_values("time_ms").head(CFG.fast_valid_rows).copy()

del early, late, all_dev
gc.collect()

print("train", train_df.shape, "valid", valid_df.shape)
print("train users/items", train_df.user_id.nunique(), train_df.video_id.nunique())
print("valid users/items", valid_df.user_id.nunique(), valid_df.video_id.nunique())
print("positive rates", train_df[TASK.label].mean(), valid_df[TASK.label].mean())
'''),
md(r'''
## 3. Immutable evaluators

The starter-kit evaluator is reproduced exactly for GAUC and nDCG@5. The legacy evaluator is supplied only as a switchable compatibility layer; use it only after the organizers define the candidate scope for Recall@50.
'''),
code(r'''
def _auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(label for _, label in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (positive_rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


def _ndcg(labels, k):
    labels = np.asarray(labels, dtype=np.int8)
    top = labels[:k]
    discounts = np.log2(np.arange(2, len(top) + 2))
    dcg = np.sum(top / discounts)
    ideal = np.sort(labels)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return 0.0 if idcg == 0 else float(dcg / idcg)


def evaluate_starter(user_ids, labels, scores, k=5):
    grouped = collections.defaultdict(list)
    for user_id, label, score in zip(user_ids, labels, scores):
        grouped[user_id].append((float(score), int(label)))
    gnum = gden = 0.0
    ndcgs = []
    for rows in grouped.values():
        rows.sort(key=lambda row: -row[0])
        ranked_labels = [label for _, label in rows]
        positives = sum(ranked_labels)
        if 0 < positives < len(rows):
            gnum += positives * _auc(ranked_labels, [score for score, _ in rows])
            gden += positives
        ndcgs.append(_ndcg(ranked_labels, k))
    gauc = gnum / gden if gden else 0.5
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    return {"GAUC": gauc, f"nDCG@{k}": ndcg, "primary": (gauc + ndcg) / 2.0}


def evaluate_legacy(user_ids, labels, scores, ndcg_k=10, recall_k=50):
    frame = pd.DataFrame({"user": user_ids, "label": labels, "score": scores})
    ndcgs, recalls = [], []
    for _, group in frame.groupby("user", sort=False):
        ranked = group.sort_values("score", ascending=False)["label"].to_numpy(dtype=np.int8)
        positives = ranked.sum()
        ndcgs.append(_ndcg(ranked, ndcg_k))
        recalls.append(0.0 if positives == 0 else float(ranked[:recall_k].sum() / positives))
    ndcg, recall = float(np.mean(ndcgs)), float(np.mean(recalls))
    return {f"nDCG@{ndcg_k}": ndcg, f"Recall@{recall_k}": recall, "primary": (ndcg + recall) / 2.0}


def evaluate_frame(frame, scores):
    if len(frame) != len(scores):
        raise ValueError("Prediction length mismatch")
    if not np.isfinite(np.asarray(scores)).all():
        raise ValueError("Predictions contain NaN or Inf")
    if TASK.mode == "starter_kit":
        return evaluate_starter(frame.user_id, frame[TASK.label], scores, TASK.ndcg_k)
    return evaluate_legacy(frame.user_id, frame[TASK.label], scores, TASK.ndcg_k, TASK.recall_k)


# Evaluator self-check: random predictions should be close to the starter kit's published random rung.
rng = np.random.default_rng(CFG.seed)
random_metrics = evaluate_frame(valid_df, rng.random(len(valid_df)))
print("Random self-check:", random_metrics)
'''),
md(r'''
## 4. Minimal EDA and leakage audit

The audit separates information available before an impression from outcomes created after the impression. Outcome columns may be used as historical training signals, but never from the row currently being scored.
'''),
code(r'''
POST_EXPOSURE_COLUMNS = {
    "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
    "long_view", "play_time_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter",
}
SAFE_CONTEXT_COLUMNS = {
    "user_id", "video_id", "author_id", "date", "hourmin", "time_ms", "duration_ms",
    "tab", "video_type", "upload_type", "visible_status", "video_duration",
    "server_width", "server_height", "music_id", "music_type", "tag",
}

print("Available post-exposure outcomes:", sorted(POST_EXPOSURE_COLUMNS.intersection(train_df.columns)))
print("Safe current-row context:", sorted(SAFE_CONTEXT_COLUMNS.intersection(train_df.columns)))

summary = pd.DataFrame({
    "split": ["train", "valid"],
    "rows": [len(train_df), len(valid_df)],
    "users": [train_df.user_id.nunique(), valid_df.user_id.nunique()],
    "items": [train_df.video_id.nunique(), valid_df.video_id.nunique()],
    "positive_rate": [train_df[TASK.label].mean(), valid_df[TASK.label].mean()],
    "mean_rows_per_user": [len(train_df) / train_df.user_id.nunique(), len(valid_df) / valid_df.user_id.nunique()],
})
display(summary)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_df.groupby("date")[TASK.label].mean().plot(ax=axes[0], marker="o", title="Positive rate by date")
train_df.groupby("tab")[TASK.label].mean().sort_values().plot.bar(ax=axes[1], title="Positive rate by tab")
plt.tight_layout()
'''),
md(r'''
## 5. Experiment registry and convergence

Each iteration records the hypothesis, change summary, configuration, metrics, runtime, and recovery information. This is the basis for the competition's required run log and autonomy evidence.
'''),
code(r'''
class ExperimentRegistry:
    def __init__(self, validation_frame, log_path):
        self.validation_frame = validation_frame
        self.log_path = Path(log_path)
        self.records = []
        self.predictions = {}
        self.models = {}

    def run(self, name, hypothesis, change_summary, fn, config=None, recovery="continue to next experiment"):
        started = time.time()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "hypothesis": hypothesis,
            "change_summary": change_summary,
            "config": config or {},
            "status": "running",
            "manual_interventions": 0,
        }
        try:
            model, predictions, extra = fn()
            predictions = np.asarray(predictions, dtype=np.float64)
            metrics = evaluate_frame(self.validation_frame, predictions)
            record.update({"status": "ok", "metrics": metrics, "extra": extra or {}})
            self.predictions[name] = predictions
            self.models[name] = model
            print(name, metrics)
        except Exception as exc:
            record.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "recovery": recovery,
            })
            print(f"{name} failed: {type(exc).__name__}: {exc}")
        record["runtime_seconds"] = round(time.time() - started, 3)
        self.records.append(record)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def leaderboard(self):
        rows = []
        for record in self.records:
            if record["status"] == "ok":
                rows.append({"name": record["name"], **record["metrics"], "seconds": record["runtime_seconds"]})
        return pd.DataFrame(rows).sort_values("primary", ascending=False) if rows else pd.DataFrame()

    def converged(self, epsilon=0.002, n=3):
        scores = [r["metrics"]["primary"] for r in self.records if r["status"] == "ok"]
        if len(scores) < n + 1:
            return False
        best_before = max(scores[: -n])
        return all(score - best_before <= epsilon for score in scores[-n:])


registry = ExperimentRegistry(valid_df, CFG.experiment_log)
'''),
md(r'''
## 6. Baseline 1 — empirical-Bayes item popularity
'''),
code(r'''
def popularity_predictions(train, valid, prior=20.0):
    global_mean = train[TASK.label].mean()
    stats = train.groupby("video_id")[TASK.label].agg(["sum", "count"])
    stats["score"] = (stats["sum"] + prior * global_mean) / (stats["count"] + prior)
    return valid["video_id"].map(stats["score"]).fillna(global_mean).to_numpy()


registry.run(
    "popularity",
    "Smoothed item response rate should reproduce the trivial baseline and validate data alignment.",
    "Add empirical-Bayes video popularity using training labels only.",
    lambda: (None, popularity_predictions(train_df, valid_df, CFG.prior_strength), {}),
    {"prior": CFG.prior_strength},
)
'''),
md(r'''
## 7. Baseline 2 — organizer-style NumPy Factorization Machine

This reproduces the five-field pointwise baseline: user, video, author, tab, and duration bucket. It is intentionally retained as the baseline the autonomous loop must beat.
'''),
code(r'''
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyFM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV, self.vV = np.zeros_like(self.V), np.zeros_like(self.V)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interaction = 0.5 * ((summed ** 2).sum(axis=1) - (embeddings ** 2).sum(axis=(1, 2)))
        return self.b + self.W[X].sum(axis=1) + interaction, embeddings, summed

    def step(self, X, y):
        batch_size = len(y)
        logits, embeddings, summed = self.logits(X)
        gradient = ((sigmoid(logits) - y) / batch_size).astype(np.float32)
        grad_v, grad_w = np.zeros_like(self.V), np.zeros_like(self.W)
        np.add.at(grad_w, X, gradient[:, None])
        np.add.at(grad_v, X, gradient[:, None, None] * (summed[:, None, :] - embeddings))
        grad_v += self.l2 * self.V
        grad_w += self.l2 * self.W
        self.t += 1
        for parameter, grad, first, second in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            first *= 0.9
            first += 0.1 * grad
            second *= 0.999
            second += 0.001 * grad * grad
            corrected_first = first / (1 - 0.9 ** self.t)
            corrected_second = second / (1 - 0.999 ** self.t)
            parameter -= self.lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        self.b -= self.lr * gradient.sum()

    def predict(self, X, batch_size=200_000):
        return np.concatenate([
            self.logits(X[start:start + batch_size])[0]
            for start in range(0, len(X), batch_size)
        ])


def encode_fm(train, valid):
    quantiles = np.unique(np.quantile(train["duration_ms"].fillna(0), np.linspace(0, 1, 11)[1:-1]))

    def raw(frame):
        result = pd.DataFrame(index=frame.index)
        result["user_id"] = frame["user_id"].astype(str)
        result["video_id"] = frame["video_id"].astype(str)
        result["author_id"] = frame["author_id"].fillna("UNK").astype(str)
        result["tab"] = frame["tab"].fillna("UNK").astype(str)
        result["dur_bucket"] = np.searchsorted(quantiles, frame["duration_ms"].fillna(0)).astype(str)
        return result

    train_raw, valid_raw = raw(train), raw(valid)
    offsets, dimensions, maps = [], [], []
    offset = 0
    for column in train_raw:
        values = pd.Index(train_raw[column].unique())
        mapping = {value: idx for idx, value in enumerate(values)}
        maps.append(mapping)
        dimensions.append(len(mapping) + 1)
        offsets.append(offset)
        offset += len(mapping) + 1

    def transform(frame):
        matrix = np.empty((len(frame), len(frame.columns)), dtype=np.int32)
        for idx, column in enumerate(frame):
            unknown = len(maps[idx])
            matrix[:, idx] = frame[column].map(maps[idx]).fillna(unknown).astype(np.int32) + offsets[idx]
        return matrix

    return transform(train_raw), transform(valid_raw), offset


def train_official_fm():
    X_train, X_valid, dimension = encode_fm(train_df, valid_df)
    y_train = train_df[TASK.label].to_numpy(dtype=np.float32)
    y_valid = valid_df[TASK.label].to_numpy(dtype=np.float32)
    model = NumpyFM(dimension, k=16, lr=0.001, seed=CFG.seed)
    rng = np.random.default_rng(CFG.seed)
    best_score, best_state, bad = -np.inf, None, 0
    for epoch in range(1, 41):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), 8192):
            batch = order[start:start + 8192]
            model.step(X_train[batch], y_train[batch])
        predictions = model.predict(X_valid)
        score = evaluate_frame(valid_df, predictions)["primary"]
        print(f"FM epoch {epoch:02d}: {score:.6f}")
        if score > best_score + 1e-5:
            best_score, bad = score, 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= 4:
                break
    model.V, model.W, model.b = best_state
    return model, model.predict(X_valid), {"best_epoch_score": best_score, "dimension": dimension}


if CFG.run_fm:
    registry.run(
        "official_fm",
        "Reproduce the fixed organizer FM before attempting improvements.",
        "Train the five-field k=16 NumPy FM with pointwise log-loss and official early stopping.",
        train_official_fm,
        {"k": 16, "lr": 0.001, "batch_size": 8192, "patience": 4},
    )
'''),
md(r'''
## 8. Leakage-safe temporal and affinity features

For each target-encoded key, training rows see only earlier rows. Validation rows see aggregates from the complete training period, never validation outcomes.

The highest-value interactions are expected to be:

- user × video;
- user × author;
- user × duration regime;
- user × tab;
- video/author trends and smoothed response rates.
'''),
code(r'''
def add_context_features(train, valid):
    train, valid = train.copy(), valid.copy()
    duration_edges = np.unique(np.quantile(train["duration_ms"].fillna(0), np.linspace(0, 1, 21)[1:-1]))
    for frame in (train, valid):
        hourmin = pd.to_numeric(frame["hourmin"], errors="coerce").fillna(0).astype(int)
        frame["hour"] = (hourmin // 100).clip(0, 23).astype(np.int8)
        frame["minute"] = (hourmin % 100).clip(0, 59).astype(np.int8)
        frame["day_index"] = (frame["date"] - CFG.train_start).astype(np.int8)
        frame["day_of_week"] = pd.to_datetime(frame["date"].astype(str)).dt.dayofweek.astype(np.int8)
        duration = pd.to_numeric(frame["duration_ms"], errors="coerce").fillna(0).clip(lower=0)
        frame["duration_log"] = np.log1p(duration).astype(np.float32)
        frame["duration_bucket"] = np.searchsorted(duration_edges, duration).astype(np.int8)
        frame["duration_regime"] = (duration > 18_000).astype(np.int8)
        frame["is_short_video"] = (duration <= 7_000).astype(np.int8)
        width = frame["server_width"] if "server_width" in frame else pd.Series(0, index=frame.index)
        height = frame["server_height"] if "server_height" in frame else pd.Series(1, index=frame.index)
        frame["aspect_ratio"] = (
            pd.to_numeric(width, errors="coerce").fillna(0)
            / pd.to_numeric(height, errors="coerce").replace(0, np.nan).fillna(1)
        ).astype(np.float32)
    return train, valid


def _key_name(keys):
    return "__".join(keys)


def add_causal_target_history(train, valid, keys_list, target, prior_strength=20.0):
    train, valid = train.copy(), valid.copy()
    global_mean = float(train[target].mean())
    time_order = np.argsort(train["time_ms"].to_numpy(), kind="stable")
    chronological = train.iloc[time_order].copy()

    for keys in keys_list:
        keys = list(keys)
        name = _key_name(keys)
        grouped = chronological.groupby(keys, sort=False, dropna=False)[target]
        prior_count = grouped.cumcount().astype(np.float32)
        prior_sum = grouped.cumsum().astype(np.float32) - chronological[target].astype(np.float32)
        chronological[f"hist_{name}_count"] = prior_count
        chronological[f"hist_{name}_rate"] = (
            (prior_sum + prior_strength * global_mean) / (prior_count + prior_strength)
        ).astype(np.float32)

        aggregate = train.groupby(keys, dropna=False)[target].agg(["sum", "count"]).reset_index()
        aggregate[f"hist_{name}_count"] = aggregate["count"].astype(np.float32)
        aggregate[f"hist_{name}_rate"] = (
            (aggregate["sum"] + prior_strength * global_mean) / (aggregate["count"] + prior_strength)
        ).astype(np.float32)
        valid = valid.merge(
            aggregate[keys + [f"hist_{name}_count", f"hist_{name}_rate"]],
            on=keys,
            how="left",
            validate="many_to_one",
        )
        valid[f"hist_{name}_count"] = valid[f"hist_{name}_count"].fillna(0).astype(np.float32)
        valid[f"hist_{name}_rate"] = valid[f"hist_{name}_rate"].fillna(global_mean).astype(np.float32)

    chronological = chronological.sort_index()
    created = [c for c in chronological if c.startswith("hist_")]
    for column in created:
        train[column] = chronological[column]
    return train, valid


def add_auxiliary_history(train, valid, targets=("is_click", "is_like", "is_follow", "is_comment", "is_forward")):
    train, valid = train.copy(), valid.copy()
    available = [target for target in targets if target in train.columns and target != TASK.label]
    chronological = train.sort_values("time_ms", kind="stable").copy()
    for target in available:
        chronological[target] = pd.to_numeric(chronological[target], errors="coerce").fillna(0).astype(np.float32)
        for key in ("video_id", "author_id"):
            name = f"aux_{target}_{key}_rate"
            grouped = chronological.groupby(key, sort=False, dropna=False)[target]
            count = grouped.cumcount().astype(np.float32)
            previous_sum = grouped.cumsum().astype(np.float32) - chronological[target]
            global_mean = float(chronological[target].mean())
            chronological[name] = ((previous_sum + 20 * global_mean) / (count + 20)).astype(np.float32)
            aggregate = train.assign(_aux=pd.to_numeric(train[target], errors="coerce").fillna(0)).groupby(key)["_aux"].agg(["sum", "count"])
            mapping = ((aggregate["sum"] + 20 * global_mean) / (aggregate["count"] + 20)).to_dict()
            valid[name] = valid[key].map(mapping).fillna(global_mean).astype(np.float32)
    chronological = chronological.sort_index()
    for column in chronological:
        if column.startswith("aux_"):
            train[column] = chronological[column]
    return train, valid


feature_train, feature_valid = add_context_features(train_df, valid_df)
history_keys = [
    ("video_id",),
    ("author_id",),
    ("tab",),
    ("duration_bucket",),
    ("video_id", "tab"),
    ("author_id", "tab"),
    ("user_id",),
    ("user_id", "video_id"),
    ("user_id", "author_id"),
    ("user_id", "tab"),
    ("user_id", "duration_regime"),
    ("user_id", "duration_bucket"),
]
feature_train, feature_valid = add_causal_target_history(
    feature_train, feature_valid, history_keys, TASK.label, CFG.prior_strength
)
feature_train, feature_valid = add_auxiliary_history(feature_train, feature_valid)

categorical_source = [
    "user_id", "video_id", "author_id", "tab", "duration_bucket", "duration_regime",
    "hour", "day_of_week", "video_type", "upload_type", "music_id", "music_type",
]
categorical_source = [c for c in categorical_source if c in feature_train.columns]

def train_based_codes(train, valid, columns):
    train, valid = train.copy(), valid.copy()
    encoded = []
    for column in columns:
        name = f"cat_{column}"
        train_values = train[column].fillna("UNK").astype(str)
        valid_values = valid[column].fillna("UNK").astype(str)
        vocabulary = {value: idx for idx, value in enumerate(pd.unique(train_values))}
        train[name] = train_values.map(vocabulary).astype(np.int32)
        valid[name] = valid_values.map(vocabulary).fillna(-1).astype(np.int32)
        encoded.append(name)
    return train, valid, encoded

feature_train, feature_valid, categorical_features = train_based_codes(feature_train, feature_valid, categorical_source)
numeric_features = [
    c for c in feature_train.columns
    if c.startswith("hist_") or c.startswith("aux_")
]
numeric_features += [c for c in [
    "duration_log", "is_short_video", "aspect_ratio", "day_index", "minute",
] if c in feature_train.columns]
MODEL_FEATURES = categorical_features + numeric_features

X_train = feature_train[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
X_valid = feature_valid[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = feature_train[TASK.label].to_numpy(dtype=np.int8)
y_valid = feature_valid[TASK.label].to_numpy(dtype=np.int8)

print(len(MODEL_FEATURES), "features")
display(pd.DataFrame({"feature": MODEL_FEATURES, "dtype": X_train.dtypes.astype(str).values}).head(50))
'''),
md(r'''
## 9. LightGBM pointwise binary model

This provides a strong GAUC-oriented component. It also tests whether historical affinity features solve more of the problem than changing the FM architecture.
'''),
code(r'''
def train_lgb_binary():
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=CFG.seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )
    predictions = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:, 1]
    importance = dict(sorted(zip(MODEL_FEATURES, model.feature_importances_), key=lambda item: -item[1])[:30])
    return model, predictions, {"best_iteration": model.best_iteration_, "top_features": importance}


if CFG.run_lgb_binary:
    registry.run(
        "lgb_binary_history",
        "Leakage-safe user-item/author/duration affinities should outperform the five-field FM on GAUC.",
        "Add causal history features and train a regularized LightGBM binary classifier.",
        train_lgb_binary,
        {"n_estimators": 1800, "learning_rate": 0.025, "num_leaves": 63},
    )
'''),
md(r'''
## 10. LightGBM LambdaRank

Groups are users, matching the evaluator's within-user ranking scope. `lambdarank_truncation_level` is set slightly above nDCG@K so training concentrates on the top of each list.
'''),
code(r'''
def grouped_order(frame):
    order = np.argsort(frame["user_id"].astype(str).to_numpy(), kind="stable")
    sorted_users = frame.iloc[order]["user_id"].astype(str)
    group_sizes = sorted_users.groupby(sorted_users, sort=False).size().to_numpy(dtype=np.int32)
    return order, group_sizes


train_group_order, train_group_sizes = grouped_order(feature_train)
valid_group_order, valid_group_sizes = grouped_order(feature_valid)


def train_lgb_ranker():
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        lambdarank_truncation_level=TASK.ndcg_k + 3,
        lambdarank_norm=True,
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=CFG.seed + 1,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train.iloc[train_group_order],
        y_train[train_group_order],
        group=train_group_sizes,
        eval_set=[(X_valid.iloc[valid_group_order], y_valid[valid_group_order])],
        eval_group=[valid_group_sizes],
        eval_at=[TASK.ndcg_k],
        categorical_feature=categorical_features,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )
    sorted_predictions = model.predict(X_valid.iloc[valid_group_order], num_iteration=model.best_iteration_)
    predictions = np.empty(len(sorted_predictions), dtype=np.float64)
    predictions[valid_group_order] = sorted_predictions
    importance = dict(sorted(zip(MODEL_FEATURES, model.feature_importances_), key=lambda item: -item[1])[:30])
    return model, predictions, {"best_iteration": model.best_iteration_, "top_features": importance}


if CFG.run_lgb_ranker:
    registry.run(
        "lgb_lambdarank_history",
        "A user-grouped LambdaRank loss should improve nDCG by aligning training with the evaluation order.",
        "Replace pointwise log-loss with LambdaRank while retaining leakage-safe history features.",
        train_lgb_ranker,
        {"eval_at": TASK.ndcg_k, "truncation": TASK.ndcg_k + 3, "num_leaves": 63},
    )
'''),
md(r'''
## 11. CatBoost pairwise ranker

CatBoost supplies a structurally different tree ensemble and is valuable primarily as an ensemble component. Enable it after the LightGBM models establish a champion.
'''),
code(r'''
def train_catboost_ranker():
    loss = "YetiRankPairwise"
    train_pool = Pool(
        X_train.iloc[train_group_order],
        y_train[train_group_order],
        group_id=feature_train.iloc[train_group_order]["user_id"].astype(str),
        cat_features=categorical_features,
    )
    valid_pool = Pool(
        X_valid.iloc[valid_group_order],
        y_valid[valid_group_order],
        group_id=feature_valid.iloc[valid_group_order]["user_id"].astype(str),
        cat_features=categorical_features,
    )
    model = CatBoostRanker(
        loss_function=loss,
        eval_metric=f"NDCG:top={TASK.ndcg_k}",
        iterations=1200,
        learning_rate=0.035,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=CFG.seed + 2,
        random_strength=0.5,
        od_type="Iter",
        od_wait=100,
        verbose=100,
        allow_writing_files=False,
        task_type="GPU" if torch.cuda.is_available() else "CPU",
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )
    sorted_predictions = model.predict(valid_pool)
    predictions = np.empty(len(sorted_predictions), dtype=np.float64)
    predictions[valid_group_order] = sorted_predictions
    return model, predictions, {"best_iteration": model.get_best_iteration(), "loss": loss}


if CFG.run_catboost_ranker:
    registry.run(
        "catboost_pairwise_history",
        "A second pairwise tree learner may capture categorical interactions missed by LightGBM and diversify the ensemble.",
        "Train CatBoost YetiRankPairwise on the same grouped, leakage-safe feature matrix.",
        train_catboost_ranker,
        {"loss": "YetiRankPairwise", "depth": 8, "iterations": 1200},
    )
'''),
md(r'''
## 12. Pairwise BPR matrix factorization

BPR directly compares a user's positive and negative exposed items. Negatives are sampled only from impressions actually shown to that user, avoiding arbitrary unobserved-item assumptions.
'''),
code(r'''
def make_id_maps(train, valid):
    user_values = pd.Index(train.user_id.astype(str).unique())
    item_values = pd.Index(train.video_id.astype(str).unique())
    user_map = {value: idx for idx, value in enumerate(user_values)}
    item_map = {value: idx for idx, value in enumerate(item_values)}
    train_users = train.user_id.astype(str).map(user_map).to_numpy(dtype=np.int64)
    train_items = train.video_id.astype(str).map(item_map).to_numpy(dtype=np.int64)
    valid_users = valid.user_id.astype(str).map(user_map).fillna(-1).to_numpy(dtype=np.int64)
    valid_items = valid.video_id.astype(str).map(item_map).fillna(-1).to_numpy(dtype=np.int64)
    return user_map, item_map, train_users, train_items, valid_users, valid_items


user_map, item_map, train_user_codes, train_item_codes, valid_user_codes, valid_item_codes = make_id_maps(train_df, valid_df)


def sample_bpr_pairs(user_codes, item_codes, labels, max_pairs_per_user=64, seed=0):
    rng = np.random.default_rng(seed)
    order = np.argsort(user_codes, kind="stable")
    users_sorted, items_sorted, labels_sorted = user_codes[order], item_codes[order], labels[order]
    unique_users, starts = np.unique(users_sorted, return_index=True)
    ends = np.r_[starts[1:], len(order)]
    out_u, out_pos, out_neg = [], [], []
    for user, start, end in zip(unique_users, starts, ends):
        items = items_sorted[start:end]
        labs = labels_sorted[start:end]
        positives, negatives = items[labs == 1], items[labs == 0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        pair_count = min(max_pairs_per_user, len(positives) * len(negatives))
        out_u.append(np.full(pair_count, user, dtype=np.int64))
        out_pos.append(rng.choice(positives, pair_count, replace=True))
        out_neg.append(rng.choice(negatives, pair_count, replace=True))
    return np.concatenate(out_u), np.concatenate(out_pos), np.concatenate(out_neg)


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users, items):
        return (self.user(users) * self.item(items)).sum(dim=1) + self.item_bias(items).squeeze(1)

    def forward(self, users, positives, negatives):
        return self.score(users, positives), self.score(users, negatives)


def train_bpr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_u, pair_pos, pair_neg = sample_bpr_pairs(
        train_user_codes, train_item_codes, y_train, max_pairs_per_user=96, seed=CFG.seed
    )
    dataset = TensorDataset(
        torch.from_numpy(pair_u), torch.from_numpy(pair_pos), torch.from_numpy(pair_neg)
    )
    loader = DataLoader(dataset, batch_size=8192, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    model = BPRMF(len(user_map), len(item_map), dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    best_score, best_state, bad = -np.inf, None, 0

    def predict():
        model.eval()
        output = np.zeros(len(valid_df), dtype=np.float32)
        known = (valid_user_codes >= 0) & (valid_item_codes >= 0)
        indices = np.flatnonzero(known)
        with torch.no_grad():
            for start in range(0, len(indices), 100_000):
                rows = indices[start:start + 100_000]
                users = torch.from_numpy(valid_user_codes[rows]).to(device)
                items = torch.from_numpy(valid_item_codes[rows]).to(device)
                output[rows] = model.score(users, items).cpu().numpy()
        return output

    for epoch in range(1, 31):
        model.train()
        losses = []
        for users, positives, negatives in loader:
            users, positives, negatives = users.to(device), positives.to(device), negatives.to(device)
            pos_score, neg_score = model(users, positives, negatives)
            loss = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predictions = predict()
        score = evaluate_frame(valid_df, predictions)["primary"]
        print(f"BPR epoch {epoch:02d}: loss={np.mean(losses):.5f} primary={score:.6f}")
        if score > best_score + 1e-5:
            best_score, bad = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    final_predictions = predict()
    return model.cpu(), final_predictions, {"pairs": len(dataset), "best_score": best_score, "device": str(device)}


if CFG.run_bpr:
    registry.run(
        "bpr_matrix_factorization",
        "Within-user positive-negative comparisons should better align collaborative embeddings with ranking metrics than pointwise FM.",
        "Train 64-dimensional BPR MF using only explicit exposed negatives and user-local sampling.",
        train_bpr,
        {"dimension": 64, "pairs_per_user": 96, "learning_rate": 0.002},
    )
'''),
md(r'''
## 13. Candidate-aware DIN-lite sequence model

The candidate video attends over each user's earlier positive videos. Training histories are causal; validation histories end at the training cutoff and are never updated using validation outcomes.

Enable this after the tabular ensemble plateaus. KuaiRand-Pure has incomplete sequences, so this branch is higher-cost and less certain than the historical aggregates.
'''),
code(r'''
def build_positive_histories(train, valid, item_map, max_history=40):
    padding = 0
    unknown_item = len(item_map) + 1
    history_by_user = collections.defaultdict(lambda: collections.deque(maxlen=max_history))
    train_histories = np.zeros((len(train), max_history), dtype=np.int32)
    train_lengths = np.zeros(len(train), dtype=np.int16)
    order = np.argsort(train["time_ms"].to_numpy(), kind="stable")
    user_strings = train.user_id.astype(str).to_numpy()
    item_strings = train.video_id.astype(str).to_numpy()
    labels = train[TASK.label].to_numpy(dtype=np.int8)
    for row in order:
        history = history_by_user[user_strings[row]]
        values = list(history)
        train_lengths[row] = len(values)
        if values:
            train_histories[row, -len(values):] = values
        if labels[row] == 1:
            history.append(item_map.get(item_strings[row], unknown_item) + 1)

    valid_histories = np.zeros((len(valid), max_history), dtype=np.int32)
    valid_lengths = np.zeros(len(valid), dtype=np.int16)
    for row, user in enumerate(valid.user_id.astype(str)):
        values = list(history_by_user[user])
        valid_lengths[row] = len(values)
        if values:
            valid_histories[row, -len(values):] = values
    return train_histories, train_lengths, valid_histories, valid_lengths


class DINLite(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32):
        super().__init__()
        self.user = nn.Embedding(n_users + 1, embedding_dim)
        self.item = nn.Embedding(n_items + 2, embedding_dim, padding_idx=0)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4, 64), nn.PReLU(), nn.Linear(64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128), nn.PReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.PReLU(), nn.Linear(64, 1),
        )

    def forward(self, users, items, histories):
        user_vector = self.user(users)
        item_vector = self.item(items)
        history_vectors = self.item(histories)
        query = item_vector.unsqueeze(1).expand_as(history_vectors)
        attention_input = torch.cat([query, history_vectors, query - history_vectors, query * history_vectors], dim=-1)
        attention_logits = self.attention(attention_input).squeeze(-1)
        mask = histories.ne(0)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(attention_logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = torch.sum(history_vectors * weights.unsqueeze(-1), dim=1)
        features = torch.cat([user_vector, item_vector, interest, item_vector * interest], dim=1)
        return self.head(features).squeeze(1)


def train_din():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: DIN is substantially faster on a Kaggle GPU")
    train_hist, _, valid_hist, _ = build_positive_histories(
        train_df, valid_df, item_map, CFG.max_history
    )
    din_train_users = np.where(train_user_codes >= 0, train_user_codes, len(user_map)).astype(np.int64)
    din_train_items = (train_item_codes + 1).astype(np.int64)
    din_valid_users = np.where(valid_user_codes >= 0, valid_user_codes, len(user_map)).astype(np.int64)
    din_valid_items = np.where(valid_item_codes >= 0, valid_item_codes + 1, len(item_map) + 1).astype(np.int64)

    rng = np.random.default_rng(CFG.seed)
    selected = np.arange(len(train_df))
    if len(selected) > CFG.max_deep_train_rows:
        selected = np.sort(rng.choice(selected, CFG.max_deep_train_rows, replace=False))
    dataset = TensorDataset(
        torch.from_numpy(din_train_users[selected]),
        torch.from_numpy(din_train_items[selected]),
        torch.from_numpy(train_hist[selected]),
        torch.from_numpy(y_train[selected].astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    model = DINLite(len(user_map), len(item_map), embedding_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    best_score, best_state, bad = -np.inf, None, 0

    def predict():
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(valid_df), 8192):
                stop = min(start + 8192, len(valid_df))
                logits = model(
                    torch.from_numpy(din_valid_users[start:stop]).to(device),
                    torch.from_numpy(din_valid_items[start:stop]).to(device),
                    torch.from_numpy(valid_hist[start:stop]).to(device),
                )
                outputs.append(logits.cpu().numpy())
        return np.concatenate(outputs)

    for epoch in range(1, 16):
        model.train()
        losses = []
        for users, items, histories, labels in loader:
            users, items, histories, labels = users.to(device), items.to(device), histories.to(device), labels.to(device)
            logits = model(users, items, histories)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        predictions = predict()
        score = evaluate_frame(valid_df, predictions)["primary"]
        print(f"DIN epoch {epoch:02d}: loss={np.mean(losses):.5f} primary={score:.6f}")
        if score > best_score + 1e-5:
            best_score, bad = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= 3:
                break
    model.load_state_dict(best_state)
    final_predictions = predict()
    return model.cpu(), final_predictions, {"best_score": best_score, "rows": len(dataset), "device": str(device)}


if CFG.run_din:
    registry.run(
        "din_lite_sequence",
        "Candidate-conditioned attention over earlier positive videos may capture user interests absent from the static FM.",
        "Add a causal DIN-lite sequence encoder with candidate-aware attention and early stopping.",
        train_din,
        {"embedding_dim": 32, "max_history": CFG.max_history, "max_rows": CFG.max_deep_train_rows},
    )
'''),
md(r'''
## 14. Within-user rank ensembling

Only relative order within each user matters. Raw model scales are therefore replaced by within-user percentile ranks before blend search. This also makes the final ensemble more stable across model families.
'''),
code(r'''
def within_user_percentile(frame, scores):
    series = pd.Series(np.asarray(scores), index=frame.index)
    return series.groupby(frame["user_id"]).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def search_rank_ensemble(registry, trials=250, seed=2026):
    names = [name for name in registry.predictions if name != "popularity"]
    if len(names) < 2:
        return None, None, None
    ranked = np.column_stack([
        within_user_percentile(registry.validation_frame, registry.predictions[name])
        for name in names
    ])
    rng = np.random.default_rng(seed)
    candidates = [np.eye(len(names))[idx] for idx in range(len(names))]
    candidates.append(np.ones(len(names)) / len(names))
    candidates.extend(rng.dirichlet(np.ones(len(names)), size=trials))
    best_score, best_weights, best_predictions = -np.inf, None, None
    for weights in candidates:
        predictions = ranked @ weights
        score = evaluate_frame(registry.validation_frame, predictions)["primary"]
        if score > best_score:
            best_score, best_weights, best_predictions = score, weights.copy(), predictions.copy()
    return names, best_weights, best_predictions


ensemble_names, ensemble_weights, ensemble_predictions = search_rank_ensemble(registry)
if ensemble_predictions is not None:
    registry.run(
        "rank_normalized_ensemble",
        "Binary, listwise, pairwise, and sequence learners make complementary ordering errors.",
        "Convert each model to within-user percentile ranks and optimize non-negative blend weights.",
        lambda: (
            {name: float(weight) for name, weight in zip(ensemble_names, ensemble_weights)},
            ensemble_predictions,
            {"weights": {name: float(weight) for name, weight in zip(ensemble_names, ensemble_weights)}},
        ),
        {"trials": 250, "models": ensemble_names},
    )

display(registry.leaderboard())
print("Converged under epsilon=0.002, N=3?", registry.converged(epsilon=0.002, n=3))
'''),
md(r'''
## 15. Diagnostics and ablations

Use these cells to decide what the autonomous loop should try next. A new branch should be justified by an observed failure mode, not merely by model novelty.
'''),
code(r'''
leaderboard = registry.leaderboard()
display(leaderboard)

if not leaderboard.empty:
    champion_name = leaderboard.iloc[0]["name"]
    champion_predictions = registry.predictions[champion_name]
    diagnostic = valid_df[["user_id", "video_id", "tab", "duration_ms", TASK.label]].copy()
    diagnostic["score"] = champion_predictions
    diagnostic["duration_regime"] = np.where(diagnostic["duration_ms"] <= 18_000, "<=18s", ">18s")

    slices = []
    for column in ["tab", "duration_regime"]:
        for value, group in diagnostic.groupby(column):
            if len(group) >= 500:
                metrics = evaluate_frame(group, group["score"].to_numpy())
                slices.append({"slice": column, "value": value, "rows": len(group), **metrics})
    display(pd.DataFrame(slices).sort_values(["slice", "primary"]))

    # Save validation champion predictions for reproducibility.
    validation_output = valid_df[["user_id", "video_id"]].copy()
    validation_output.insert(0, "row_id", np.arange(len(validation_output), dtype=np.int64))
    validation_output["score"] = champion_predictions
    validation_output.to_csv("/kaggle/working/validation_champion.csv", index=False)
    print("Champion:", champion_name)
'''),
md(r'''
## 16. Submission generation for a sanitized organizer-provided evaluation set

The notebook intentionally does not read public test outcomes. After the organizer supplies a sanitized evaluation frame, build its features using training aggregates, obtain model scores, and pass them to this writer. The scorer should never require outcome columns.
'''),
code(r'''
def write_submission(evaluation_frame, scores, path="/kaggle/working/submission.csv"):
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(evaluation_frame):
        raise ValueError("Submission score length mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("Submission contains NaN or Inf")
    required = {"user_id", "video_id"}
    if not required.issubset(evaluation_frame.columns):
        raise KeyError(f"Missing submission columns: {required - set(evaluation_frame.columns)}")
    submission = pd.DataFrame({
        "row_id": np.arange(len(evaluation_frame), dtype=np.int64),
        "user_id": evaluation_frame["user_id"].to_numpy(),
        "video_id": evaluation_frame["video_id"].to_numpy(),
        "score": scores,
    })
    submission.to_csv(path, index=False)
    assert submission.columns.tolist() == ["row_id", "user_id", "video_id", "score"]
    assert submission.row_id.iloc[0] == 0 and submission.row_id.iloc[-1] == len(submission) - 1
    print(f"Wrote {len(submission):,} rows to {path}")
    return submission


# Example for validation-format verification only:
if not registry.leaderboard().empty:
    best_name = registry.leaderboard().iloc[0]["name"]
    preview = write_submission(valid_df, registry.predictions[best_name], "/kaggle/working/validation_submission_format.csv")
    display(preview.head())
'''),
md(r'''
## 17. Recommended autonomous iteration ladder

Run the following sequence and let the experiment registry decide whether the added complexity earns its cost:

| Iteration | Hypothesis | Promotion condition |
|---|---|---|
| 0 | Official FM reproduces the baseline | Alignment and score sanity pass |
| 1 | Causal historical affinities beat static IDs | Primary improves by >0.002 |
| 2 | LambdaRank improves top-5 ordering | nDCG gain without unacceptable GAUC loss |
| 3 | Rank ensemble balances GAUC and nDCG | Primary improves by >0.002 |
| 4 | BPR provides collaborative diversity | Ensemble weight is non-trivial and score rises |
| 5 | CatBoost catches different categorical interactions | Gain justifies CPU/GPU cost |
| 6 | DIN adds sequence signal absent from aggregates | Gain exceeds its GPU/token cost |

If three consecutive valid experiments improve the champion by no more than 0.002, record convergence and stop. Do not chase the public test set. Report total runtime, GPU-hours, token usage, failures, recoveries, and manual interventions alongside the validation-best checkpoint.

### Next feature branches after organizer clarification

1. random-exposure priors and inverse-propensity/domain adaptation;
2. month-level video statistics, only if explicitly permitted;
3. tag/category affinity and caption embedding clusters;
4. a duration-conditioned survival/watch-time auxiliary head;
5. rolling chronological cross-validation and final train+validation refit, if allowed.
'''),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "dockerImageVersionId": None,
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output = Path("outputs/kuairand_autonomous_research_lab.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output.resolve())
print(len(cells), "cells")
