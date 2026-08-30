#!/usr/bin/env python3
"""Leak-free DVR/Watch-Time-Gain ablation for KuaiRand-Pure.

The candidate keeps organizer `long_view` BCE as the primary objective. It adds
the paper's duration-conditioned Watch-Time Gain (WTG) auxiliary target and an
optional gradient-reversal duration adversary. The score used for evaluation is
either the BCE probability or a predeclared small within-user rank blend with
the WTG head.

Screen protocol: Apr 8-11 fit -> Apr 12-14 config selection -> Apr 8-14 refit
-> Apr 15-21 locked screen. The Apr 22+ file is never opened by the screen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "external" / "kuairand-starter-kit"
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
RUNTIME = ROOT / "runtime" / "duration-debiasing" / "dvr-wtg"
RESULTS = ROOT / "results" / "duration-debiasing"
SCREEN_REPORT = RESULTS / "dvr-wtg-screen.json"
CONFIRMATION_REPORT = RESULTS / "dvr-wtg-confirmation.json"
AUDIT_REPORT = RESULTS / "dvr-wtg-residual-audit.json"
PAPER = "https://arxiv.org/abs/2208.05190"
OFFICIAL_CODE = "https://github.com/tsinghua-fib-lab/WTG-DVR"
WTG_PRIOR_ROWS = 200.0
MAX_DURATION_SECOND = 600
FIXED_RESIDUAL_WEIGHT = 0.05
CONFIGS = (
    {"wtg_weight": 0.02, "adversary_weight": 0.00, "wtg_rank_blend": 0.00},
    {"wtg_weight": 0.05, "adversary_weight": 0.00, "wtg_rank_blend": 0.05},
    {"wtg_weight": 0.05, "adversary_weight": 0.01, "wtg_rank_blend": 0.05},
    {"wtg_weight": 0.10, "adversary_weight": 0.02, "wtg_rank_blend": 0.10},
    {"wtg_weight": 0.05, "adversary_weight": 0.05, "wtg_rank_blend": 0.00},
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate_module = load_module("dvr_wtg_evaluate", STARTER / "evaluate.py")


def load_authors(data_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[row["video_id"]] = row["author_id"]
    return result


def parse_row(row: dict[str, str], authors: dict[str, str]) -> tuple:
    return (
        int(row["date"]), row["user_id"], row["video_id"],
        authors.get(row["video_id"], "UNK"), row["tab"],
        float(row["duration_ms"]), 1 if row["long_view"] != "0" else 0,
        float(row["play_time_ms"]),
    )


def load_screen_rows(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    authors = load_authors(data_dir)
    core: list[tuple] = []
    dev: list[tuple] = []
    screen: list[tuple] = []
    # Deliberately do not open log_standard_4_22_to_5_08_pure.csv.
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = parse_row(raw, authors)
            if row[0] <= 20220411:
                core.append(row)
            elif row[0] <= 20220414:
                dev.append(row)
            elif row[0] <= 20220421:
                screen.append(row)
            else:
                raise RuntimeError(f"Unexpected train-only date: {row[0]}")
    counts = (len(core), len(dev), len(screen))
    if counts != (559_379, 332_039, 249_694):
        raise RuntimeError(f"Unexpected train-only rows: {counts}")
    return core, dev, screen


def load_confirmation_rows(data_dir: Path) -> tuple[list[tuple], list[tuple]]:
    """Load Apr 8-21 and parse later outcomes only after an Apr 22-28 date check."""
    authors = load_authors(data_dir)
    train: list[tuple] = []
    with (data_dir / "log_standard_4_08_to_4_21_pure.csv").open(encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            train.append(parse_row(raw, authors))
    valid: list[tuple] = []
    with (data_dir / "log_standard_4_22_to_5_08_pure.csv").open(encoding="utf-8") as stream:
        header = next(stream)
        fieldnames = next(csv.reader([header]))
        if fieldnames[:3] != ["user_id", "video_id", "date"]:
            raise RuntimeError(f"Unexpected later-log header: {fieldnames[:3]}")
        for line in stream:
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise RuntimeError("Malformed later-log row")
            date = int(prefix[2])
            if date > 20220428:
                continue
            if date < 20220422:
                raise RuntimeError(f"Unexpected confirmation date: {date}")
            values = next(csv.reader([line]))
            valid.append(parse_row(dict(zip(fieldnames, values, strict=True)), authors))
    if len(train) != 1_141_112 or len(valid) != 124_909:
        raise RuntimeError(f"Unexpected confirmation rows: {len(train)}, {len(valid)}")
    return train, valid


def encode_without_duration(
    train: list[tuple], valid: list[tuple]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Encode user/video/author/tab only, matching DVR's removed duration input."""
    def values(row: tuple) -> tuple[str, str, str, str]:
        return str(row[1]), str(row[2]), str(row[3]), str(row[4])

    vocabs: list[dict[str, int]] = [dict() for _ in range(4)]
    for row in train:
        for field, value in enumerate(values(row)):
            if value not in vocabs[field]:
                vocabs[field][value] = len(vocabs[field])
    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

    def apply(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        x = np.empty((len(rows), 4), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users: list[str] = []
        for index, row in enumerate(rows):
            for field, value in enumerate(values(row)):
                x[index, field] = vocabs[field].get(value, unknown[field]) + offsets[field]
            y[index] = float(row[6])
            users.append(str(row[1]))
        return x, y, users

    train_x, train_y, _ = apply(train)
    valid_x, valid_y, users = apply(valid)
    return train_x, train_y, valid_x, valid_y, users, int(sum(dimensions))


def duration_seconds(rows: list[tuple]) -> np.ndarray:
    values = np.asarray([row[5] for row in rows], dtype=np.float64)
    return np.clip(np.rint(values / 1000.0), 1, MAX_DURATION_SECOND).astype(np.int32)


def truncated_watch_seconds(rows: list[tuple]) -> np.ndarray:
    duration = np.asarray([max(float(row[5]), 1.0) for row in rows], dtype=np.float64)
    play = np.asarray([max(float(row[7]), 0.0) for row in rows], dtype=np.float64)
    return (np.minimum(play, duration) / 1000.0).astype(np.float32)


def fit_wtg_reference(rows: list[tuple]) -> dict:
    buckets = duration_seconds(rows)
    watch = truncated_watch_seconds(rows).astype(np.float64)
    count = np.bincount(buckets, minlength=MAX_DURATION_SECOND + 1).astype(np.float64)
    total = np.bincount(buckets, weights=watch, minlength=MAX_DURATION_SECOND + 1)
    square = np.bincount(buckets, weights=watch * watch, minlength=MAX_DURATION_SECOND + 1)
    global_mean = float(watch.mean())
    global_variance = float(watch.var())
    denominator = count + WTG_PRIOR_ROWS
    mean = (total + WTG_PRIOR_ROWS * global_mean) / denominator
    second = (square + WTG_PRIOR_ROWS * (global_variance + global_mean**2)) / denominator
    standard_deviation = np.sqrt(np.maximum(second - mean * mean, 0.05**2))
    log_duration = np.log1p(np.asarray([row[5] for row in rows], dtype=np.float64) / 1000.0)
    return {
        "mean": mean.astype(np.float32),
        "standard_deviation": standard_deviation.astype(np.float32),
        "duration_log_mean": float(log_duration.mean()),
        "duration_log_std": float(max(log_duration.std(), 1e-6)),
        "observed_duration_buckets": int(np.sum(count > 0)),
        "minimum_bucket_support": int(count[count > 0].min()),
        "maximum_bucket_support": int(count.max()),
    }


def auxiliary_targets(rows: list[tuple], reference: dict) -> tuple[np.ndarray, np.ndarray]:
    buckets = duration_seconds(rows)
    watch = truncated_watch_seconds(rows)
    gain = (watch - reference["mean"][buckets]) / reference["standard_deviation"][buckets]
    gain = np.clip(gain, -5.0, 5.0).astype(np.float32)
    log_duration = np.log1p(np.asarray([row[5] for row in rows], dtype=np.float32) / 1000.0)
    duration_z = (
        (log_duration - reference["duration_log_mean"]) / reference["duration_log_std"]
    ).astype(np.float32)
    return gain, duration_z


def within_user_ranks(users: list[str], scores: np.ndarray) -> np.ndarray:
    grouped: dict[str, list[int]] = {}
    for index, user in enumerate(users):
        grouped.setdefault(str(user), []).append(index)
    output = np.empty(len(scores), dtype=np.float32)
    for row_list in grouped.values():
        rows = np.asarray(row_list, dtype=np.int64)
        order = np.argsort(scores[rows], kind="stable")
        ranks = np.empty(len(rows), dtype=np.float32)
        ranks[order] = np.arange(len(rows), dtype=np.float32)
        output[rows] = ranks / max(len(rows) - 1, 1)
    return output


def final_scores(
    users: list[str], long_probability: np.ndarray, wtg_prediction: np.ndarray,
    blend: float,
) -> np.ndarray:
    if blend <= 0:
        return long_probability.astype(np.float32)
    long_rank = within_user_ranks(users, long_probability)
    wtg_rank = within_user_ranks(users, wtg_prediction)
    return ((1.0 - blend) * long_rank + blend * wtg_rank).astype(np.float32)


def evaluate(users: list[str], labels: np.ndarray, scores: np.ndarray) -> tuple[dict, np.ndarray]:
    ranks = within_user_ranks(users, scores)
    result = evaluate_module.evaluate(users, labels, ranks)
    return {
        "primary": float(result["primary"]), "GAUC": float(result["GAUC"]),
        "nDCG@5": float(result["nDCG@5"]), "rows": int(result["rows"]),
        "users": int(result["users"]),
    }, ranks


def peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw / (1024 * 1024 if sys.platform == "darwin" else 1024)


def usage(start_wall: float, start_cpu: float) -> dict:
    wall = time.monotonic() - start_wall
    cpu = time.process_time() - start_cpu
    return {
        "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
        "cpu_hours": round(cpu / 3600, 6),
        "cpu_utilization_percent": round(100 * cpu / max(wall, 1e-9), 2),
        "peak_rss_mb": round(peak_rss_mb(), 3), "gpu_hours": 0.0,
        "peak_gpu_memory_mb": 0.0, "device": "cpu", "architecture": platform.machine(),
    }


def make_model(dimension: int, fields: int, embedding_dim: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class GradientReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.view_as(value)

        @staticmethod
        def backward(ctx, gradient):
            return -gradient

    class DVRWTG(nn.Module):
        def __init__(self):
            super().__init__()
            shared_width = max(hidden // 2, 8)
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, embedding_dim)
            self.shared = nn.Sequential(
                nn.Linear(fields * embedding_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, shared_width), nn.ReLU(), nn.Dropout(dropout),
            )
            self.long_head = nn.Linear(shared_width, 1)
            self.wtg_head = nn.Linear(shared_width, 1)
            self.duration_adversary = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1),
            )
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            embedded = self.embedding(x)
            summed = embedded.sum(dim=1)
            fm = 0.5 * (summed.square() - embedded.square().sum(dim=1)).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            shared = self.shared(embedded.flatten(start_dim=1))
            long_logit = linear + fm + self.long_head(shared).squeeze(-1)
            wtg = self.wtg_head(shared).squeeze(-1)
            duration = self.duration_adversary(GradientReverse.apply(wtg[:, None])).squeeze(-1)
            return long_logit, wtg, duration

    return DVRWTG()


def train_model(
    *, config: dict, train_x: np.ndarray, train_y: np.ndarray, train_wtg: np.ndarray,
    train_duration: np.ndarray, dimension: int, epochs: int, seed: int,
    embedding_dim: int, hidden: int, dropout: float, learning_rate: float,
    weight_decay: float, batch_size: int, valid_x: np.ndarray | None = None,
    valid_y: np.ndarray | None = None, valid_users: list[str] | None = None,
    patience: int | None = None,
) -> tuple[object, list[dict], int, dict]:
    import torch
    from torch import nn

    started_wall, started_cpu = time.monotonic(), time.process_time()
    x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    wtg_t = torch.from_numpy(train_wtg.astype(np.float32, copy=False))
    duration_t = torch.from_numpy(train_duration.astype(np.float32, copy=False))
    valid_x_t = None if valid_x is None else torch.from_numpy(valid_x.astype(np.int64, copy=False))
    torch.manual_seed(seed)
    model = make_model(dimension, train_x.shape[1], embedding_dim, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    random = np.random.default_rng(seed + 82_202)
    best_primary, best_epoch, stale = -1.0, 0, 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses, bces, wtg_losses, duration_losses = [], [], [], []
        order = random.permutation(len(train_y))
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            long_logit, wtg_prediction, duration_prediction = model(x_t[indices])
            bce = nn.functional.binary_cross_entropy_with_logits(long_logit, y_t[indices])
            wtg_loss = nn.functional.smooth_l1_loss(wtg_prediction, wtg_t[indices])
            duration_loss = nn.functional.smooth_l1_loss(duration_prediction, duration_t[indices])
            loss = (
                bce
                + float(config["wtg_weight"]) * wtg_loss
                + float(config["adversary_weight"]) * duration_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach())); bces.append(float(bce.detach()))
            wtg_losses.append(float(wtg_loss.detach())); duration_losses.append(float(duration_loss.detach()))

        record = {
            "epoch": epoch, "loss": float(np.mean(losses)), "BCE": float(np.mean(bces)),
            "WTG_loss": float(np.mean(wtg_losses)),
            "duration_adversary_loss": float(np.mean(duration_losses)),
        }
        if valid_x_t is not None and valid_y is not None and valid_users is not None:
            long_probability, wtg_prediction = score_model(model, valid_x, batch_size)
            scores = final_scores(valid_users, long_probability, wtg_prediction, float(config["wtg_rank_blend"]))
            measured, _ = evaluate(valid_users, valid_y, scores)
            record.update({key: measured[key] for key in ("primary", "GAUC", "nDCG@5")})
            if measured["primary"] > best_primary + 1e-5:
                best_primary, best_epoch, stale = measured["primary"], epoch, 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    history.append(record)
                    break
        history.append(record)
    if valid_x_t is None:
        best_epoch = epochs
    return model, history, best_epoch, usage(started_wall, started_cpu)


def score_model(model, x: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    x_t = torch.from_numpy(x.astype(np.int64, copy=False))
    long_parts, wtg_parts = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x_t), batch_size * 4):
            long_logit, wtg, _ = model(x_t[start:start + batch_size * 4])
            long_parts.append(torch.sigmoid(long_logit).numpy())
            wtg_parts.append(wtg.numpy())
    return (
        np.concatenate(long_parts).astype(np.float32),
        np.concatenate(wtg_parts).astype(np.float32),
    )


def common(args) -> dict:
    return {
        "seed": args.seed, "embedding_dim": args.embedding_dim, "hidden": args.hidden,
        "dropout": args.dropout, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
    }


def deltas(candidate: dict, control: dict) -> dict[str, float]:
    return {
        key: float(candidate[key] - control[key])
        for key in ("primary", "GAUC", "nDCG@5")
    }


def actual_user_fold(user: str) -> int:
    try:
        return int(user) % 4
    except ValueError:
        return int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 4


def residual_audit(users: list[str], labels: np.ndarray, candidate_ranks: np.ndarray) -> dict:
    started_wall, started_cpu = time.monotonic(), time.process_time()
    with np.load(ROOT / "results" / "final-model" / "validation-scores.npz") as stored:
        champion = within_user_ranks(users, np.asarray(stored["scores"], dtype=np.float32))
    residual = champion + FIXED_RESIDUAL_WEIGHT * (candidate_ranks - champion)
    user_array = np.asarray(users, dtype=object)
    folds = np.asarray([actual_user_fold(str(user)) for user in users], dtype=np.int8)

    def measured(mask: np.ndarray, scores: np.ndarray) -> dict:
        result = evaluate_module.evaluate(user_array[mask].tolist(), labels[mask], scores[mask])
        return {
            "primary": float(result["primary"]), "GAUC": float(result["GAUC"]),
            "nDCG@5": float(result["nDCG@5"]),
        }

    fold_results = []
    for fold in range(4):
        mask = folds == fold
        baseline = measured(mask, champion)
        candidate = measured(mask, residual)
        gains = deltas(candidate, baseline)
        fold_results.append({
            "fold": fold, "rows": int(mask.sum()),
            "users": len(set(user_array[mask].tolist())),
            "baseline_metrics": baseline, "residual_metrics": candidate,
            "gains": gains,
            "all_metrics_nonnegative": bool(all(value >= -1e-12 for value in gains.values())),
        })
    mask = np.ones(len(labels), dtype=bool)
    baseline = measured(mask, champion)
    candidate = measured(mask, residual)
    gains = deltas(candidate, baseline)
    all_folds = bool(all(item["all_metrics_nonnegative"] for item in fold_results))
    all_global = bool(all(value >= -1e-12 for value in gains.values()))
    report = {
        "experiment": "fixed 5% DVR/WTG residual against frozen champion",
        "fixed_weight": FIXED_RESIDUAL_WEIGHT,
        "baseline_metrics": baseline, "fixed_residual_metrics": candidate,
        "fixed_residual_gains": gains, "folds": fold_results,
        "all_four_folds_all_metrics_nonnegative": all_folds,
        "all_global_metrics_nonnegative": all_global,
        "merits_promotion": bool(all_folds and all_global and gains["primary"] > 0),
        "resource_usage": usage(started_wall, started_cpu),
        "hidden_test_outcomes_parsed": False, "integrity_incidents": [],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_screen(args) -> dict:
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    core, dev, screen = load_screen_rows(args.data_dir.resolve())
    control_config = {"wtg_weight": 0.0, "adversary_weight": 0.0, "wtg_rank_blend": 0.0}

    core_x, core_y, dev_x, dev_y, dev_users, selection_dimension = encode_without_duration(core, dev)
    selection_reference = fit_wtg_reference(core)
    core_wtg, core_duration = auxiliary_targets(core, selection_reference)
    control_model, control_history, selected_epoch, control_selection_usage = train_model(
        config=control_config, train_x=core_x, train_y=core_y, train_wtg=core_wtg,
        train_duration=core_duration, dimension=selection_dimension, epochs=args.epochs,
        valid_x=dev_x, valid_y=dev_y, valid_users=dev_users, patience=args.patience,
        **common(args),
    )
    control_probability, control_wtg = score_model(control_model, dev_x, args.batch_size)
    control_selection_metrics, _ = evaluate(dev_users, dev_y, control_probability)

    config_runs = []
    for config_index, config in enumerate(CONFIGS):
        model, history, _, run_usage = train_model(
            config=config, train_x=core_x, train_y=core_y, train_wtg=core_wtg,
            train_duration=core_duration, dimension=selection_dimension,
            epochs=selected_epoch, patience=None, **common(args),
        )
        long_probability, wtg_prediction = score_model(model, dev_x, args.batch_size)
        scores = final_scores(dev_users, long_probability, wtg_prediction, float(config["wtg_rank_blend"]))
        measured, _ = evaluate(dev_users, dev_y, scores)
        config_runs.append({
            "config_index": config_index, "config": config,
            "selection_metrics": measured,
            "deltas_vs_control": deltas(measured, control_selection_metrics),
            "training_history": history, "resource_usage": run_usage,
            "wtg_prediction_duration_correlation": float(np.corrcoef(
                wtg_prediction, auxiliary_targets(dev, selection_reference)[1]
            )[0, 1]),
        })
    selected = max(config_runs, key=lambda item: (item["selection_metrics"]["primary"], -item["config_index"]))
    selected_config = CONFIGS[int(selected["config_index"])]

    full_rows = core + dev
    full_x, full_y, screen_x, screen_y, screen_users, refit_dimension = encode_without_duration(full_rows, screen)
    refit_reference = fit_wtg_reference(full_rows)
    full_wtg, full_duration = auxiliary_targets(full_rows, refit_reference)
    final_runs = []
    arrays = {}
    for name, config in (("control", control_config), ("dvr_wtg", selected_config)):
        model, history, _, run_usage = train_model(
            config=config, train_x=full_x, train_y=full_y, train_wtg=full_wtg,
            train_duration=full_duration, dimension=refit_dimension,
            epochs=selected_epoch, patience=None, **common(args),
        )
        long_probability, wtg_prediction = score_model(model, screen_x, args.batch_size)
        scores = final_scores(screen_users, long_probability, wtg_prediction, float(config["wtg_rank_blend"]))
        measured, ranks = evaluate(screen_users, screen_y, scores)
        arrays[f"{name}_scores"] = scores
        arrays[f"{name}_ranks"] = ranks
        arrays[f"{name}_long_probability"] = long_probability
        arrays[f"{name}_wtg_prediction"] = wtg_prediction
        import torch
        checkpoint = args.output_dir / f"screen-{name}.pt"
        torch.save(model.state_dict(), checkpoint)
        final_runs.append({
            "mode": name, "config": config, "metrics": measured,
            "training_history": history, "resource_usage": run_usage,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        })

    score_path = args.output_dir / "screen-scores.npz"
    np.savez_compressed(score_path, **arrays)
    control, candidate = final_runs
    gains = deltas(candidate["metrics"], control["metrics"])
    gate = bool(gains["primary"] >= 0.0001 and gains["GAUC"] > 0 and gains["nDCG@5"] > 0)
    report = {
        "experiment": "DVR Watch-Time-Gain duration-debiasing ablation",
        "paper": PAPER, "official_implementation": OFFICIAL_CODE,
        "status": "screen_passed" if gate else "screen_rejected",
        "merits_confirmation": gate,
        "selected_config_index": selected["config_index"],
        "selected_config": selected_config,
        "predeclared_confirmation_residual_weight": FIXED_RESIDUAL_WEIGHT,
        "locked_epoch_count": selected_epoch,
        "predeclared_config_grid": list(CONFIGS),
        "control_selection_metrics": control_selection_metrics,
        "config_selection_runs": config_runs,
        "control": control, "candidate": candidate, "deltas_vs_control": gains,
        "protocol": {
            "selection_fit": "Apr8-11", "config_selection": "Apr12-14",
            "refit": "Apr8-14", "locked_screen": "Apr15-21",
            "apr22_plus_file_opened": False, "hidden_test_outcomes_parsed": False,
            "validation_target": "organizer long_view, unchanged", "integrity_incidents": [],
        },
        "matched_configuration": {
            **common(args), "duration_input_removed_for_both_models": True,
            "control_loss": "BCE(long_view)",
            "candidate_loss": "BCE(long_view) + alpha*SmoothL1(WTG) + beta*GRL-duration-SmoothL1",
            "WTG": "z-score of min(play_time_ms,duration_ms) within rounded duration-second, with 200-row global shrinkage",
            "same_initialization_and_row_order": True,
        },
        "source_proposal_audit": {
            "useful_direction": "duration debiasing and chronological features",
            "corrected_target": "paper WTG instead of watch_ratio*log1p(duration)",
            "official_label_retained": "long_view instead of an invented 80%-completion label",
            "pair_scope": "no cross-user pairs; this ablation uses pointwise BCE plus a WTG auxiliary",
            "sequence_code_reused": False,
            "sequence_reason": "causal Transformer, GRU, and contrastive sequence encoders were already rejected",
        },
        "WTG_reference": {
            key: value for key, value in refit_reference.items()
            if key not in {"mean", "standard_deviation"}
        },
        "resources": {
            "control_epoch_selection": control_selection_usage,
            "total": usage(overall_wall, overall_cpu),
        },
        "artifacts": {
            "scores": str(score_path.relative_to(ROOT)),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
        },
        "recommendation": "run locked confirmation" if gate else "reject DVR/WTG; do not access Apr22+",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    SCREEN_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_confirmation(args) -> dict:
    if not 0 <= args.config_index < len(CONFIGS) or args.locked_epochs <= 0:
        raise RuntimeError("Confirmation requires the locked screen config and epoch count")
    overall_wall, overall_cpu = time.monotonic(), time.process_time()
    selected_config = CONFIGS[args.config_index]
    control_config = {"wtg_weight": 0.0, "adversary_weight": 0.0, "wtg_rank_blend": 0.0}
    train, valid = load_confirmation_rows(args.data_dir.resolve())
    train_x, train_y, valid_x, valid_y, valid_users, dimension = encode_without_duration(train, valid)
    reference = fit_wtg_reference(train)
    train_wtg, train_duration = auxiliary_targets(train, reference)
    arrays = {}
    runs = []
    for name, config in (("control", control_config), ("dvr_wtg", selected_config)):
        model, history, _, run_usage = train_model(
            config=config, train_x=train_x, train_y=train_y, train_wtg=train_wtg,
            train_duration=train_duration, dimension=dimension,
            epochs=args.locked_epochs, patience=None, **common(args),
        )
        long_probability, wtg_prediction = score_model(model, valid_x, args.batch_size)
        scores = final_scores(valid_users, long_probability, wtg_prediction, float(config["wtg_rank_blend"]))
        measured, ranks = evaluate(valid_users, valid_y, scores)
        arrays[f"{name}_scores"] = scores
        arrays[f"{name}_ranks"] = ranks
        arrays[f"{name}_long_probability"] = long_probability
        arrays[f"{name}_wtg_prediction"] = wtg_prediction
        import torch
        checkpoint = args.output_dir / f"confirmation-{name}.pt"
        torch.save(model.state_dict(), checkpoint)
        runs.append({
            "mode": name, "config": config, "metrics": measured,
            "training_history": history, "resource_usage": run_usage,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        })
    score_path = args.output_dir / "confirmation-scores.npz"
    np.savez_compressed(score_path, **arrays)
    control, candidate = runs
    gains = deltas(candidate["metrics"], control["metrics"])
    audit = residual_audit(valid_users, valid_y, arrays["dvr_wtg_ranks"])
    report = {
        "experiment": "locked DVR Watch-Time-Gain confirmation",
        "paper": PAPER, "official_implementation": OFFICIAL_CODE,
        "status": "confirmation_evaluated",
        "selected_config_index_locked": args.config_index,
        "selected_config": selected_config, "locked_epoch_count": args.locked_epochs,
        "control": control, "candidate": candidate, "deltas_vs_control": gains,
        "matched_candidate_improves_all_metrics": bool(all(value > 0 for value in gains.values())),
        "protocol": {
            "train": "Apr8-21", "evaluation": "Apr22-28", "retuning": False,
            "screen_artifact": str(SCREEN_REPORT.relative_to(ROOT)),
            "outcome_fields_parsed_only_for": "Apr22-28",
            "hidden_test_outcomes_parsed": False, "integrity_incidents": [],
        },
        "residual_audit": {
            "artifact": str(AUDIT_REPORT.relative_to(ROOT)),
            "fixed_weight": FIXED_RESIDUAL_WEIGHT,
            "fixed_residual_metrics": audit["fixed_residual_metrics"],
            "fixed_residual_gains": audit["fixed_residual_gains"],
            "all_four_folds_all_metrics_nonnegative": audit["all_four_folds_all_metrics_nonnegative"],
            "all_global_metrics_nonnegative": audit["all_global_metrics_nonnegative"],
            "merits_promotion": audit["merits_promotion"],
        },
        "resources": {"total": usage(overall_wall, overall_cpu)},
        "artifacts": {
            "scores": str(score_path.relative_to(ROOT)),
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
        },
        "recommendation": (
            "eligible for integrity review; do not auto-promote"
            if audit["merits_promotion"] else "reject; retain frozen champion"
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    CONFIRMATION_REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--config-index", type=int, default=-1)
    parser.add_argument("--locked-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=220805)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(max(1, min(args.threads, 12)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_confirmation(args) if args.confirmation else run_screen(args)
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
