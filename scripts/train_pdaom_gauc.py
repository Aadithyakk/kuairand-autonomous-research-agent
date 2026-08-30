#!/usr/bin/env python3
"""Leak-free PDAOM screen for personalized GAUC optimization.

The experiment follows Sun et al., "Enhancing Personalized Ranking With
Differentiable Group AUC Optimization".  Complete user groups are packed into
bounded batches.  The candidate adds an exponential maximum-violation term
between the lowest-scoring positive and highest-scoring negative for each
eligible user.  The alpha-zero control has the same model, initialization,
grouped batches, optimizer, and epoch count.

Only the April 8--21 organizer log is opened.  April 8--11 fits the selection
models, April 12--14 selects a predeclared alpha with a zero-preferring rule,
and April 8--14 is refit before the locked April 15--21 screen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "train_neural_ndcg_residual.py"
DATA = ROOT / "external" / "KuaiRand-Pure" / "data"
DEFAULT_RUNTIME = ROOT / "runtime" / "parallel-calibrated-ranking" / "pdaom"
DEFAULT_REPORT = ROOT / "results" / "calibrated-ranking" / "pdaom-screen.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared = load_module("pdaom_shared", SOURCE)


def peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def usage(start_wall: float, start_cpu: float) -> dict:
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


def group_indices(users: list[str] | np.ndarray) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        groups[str(user)].append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def packed_group_batches(
    groups: list[np.ndarray], *, seed: int, epoch: int, row_cap: int
) -> list[list[np.ndarray]]:
    random = np.random.default_rng(seed + 1009 * epoch)
    order = random.permutation(len(groups))
    batches: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    rows = 0
    for group_index in order:
        group = groups[int(group_index)]
        if current and rows + len(group) > row_cap:
            batches.append(current)
            current, rows = [], 0
        current.append(group)
        rows += len(group)
    if current:
        batches.append(current)
    return batches


def score_metrics(users, labels, logits: np.ndarray) -> dict:
    ranks = shared.within_user_ranks(list(users), logits)
    measured = shared.evaluate_module.evaluate(users, labels, ranks)
    return {
        "primary": float(measured["primary"]),
        "GAUC": float(measured["GAUC"]),
        "nDCG@5": float(measured["nDCG@5"]),
        "rows": int(len(labels)),
        "users": int(len(set(map(str, users)))),
    }


def deltas(candidate: dict, control: dict) -> dict:
    return {
        key: float(candidate[key] - control[key])
        for key in ("primary", "GAUC", "nDCG@5")
    }


def train_and_score(
    *, encoded: dict, dimension: int, alpha: float, seed: int,
    embedding_dim: int, hidden: int, learning_rate: float,
    weight_decay: float, row_cap: int, epochs: int, threads: int,
) -> tuple[np.ndarray, dict, list[dict]]:
    import torch
    from torch import nn

    torch.set_num_threads(threads)
    train_x, train_y, train_users = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    train_x_t = torch.from_numpy(train_x.astype(np.int64, copy=False))
    train_y_t = torch.from_numpy(train_y.astype(np.float32, copy=False))
    valid_x_t = torch.from_numpy(valid_x.astype(np.int64, copy=False))
    groups = group_indices(train_users)

    torch.manual_seed(seed)
    model = shared.make_deepfm(
        dimension=dimension,
        fields=int(train_x.shape[1]),
        embedding_dim=embedding_dim,
        hidden=hidden,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        epoch_bce: list[float] = []
        epoch_pdaom: list[float] = []
        eligible_total = 0
        for packed in packed_group_batches(
            groups, seed=seed, epoch=epoch, row_cap=row_cap
        ):
            indices_np = np.concatenate(packed)
            indices = torch.from_numpy(indices_np)
            labels = train_y_t[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x_t[indices])
            bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            violations = []
            offset = 0
            for group in packed:
                length = len(group)
                local_logits = logits[offset:offset + length]
                local_labels = labels[offset:offset + length]
                offset += length
                positive = local_logits[local_labels > 0.5]
                negative = local_logits[local_labels <= 0.5]
                if len(positive) == 0 or len(negative) == 0:
                    continue
                # The paper defines f(x) as a probability.  Sigmoid bounds the
                # exponential surrogate and retains its monotonic ordering.
                hard_margin = torch.sigmoid(positive).min() - torch.sigmoid(
                    negative
                ).max()
                violations.append(torch.exp(-hard_margin))
            if violations:
                pdaom = torch.stack(violations).mean()
                eligible_total += len(violations)
            else:
                pdaom = logits.sum() * 0.0
            loss = bce + alpha * pdaom
            loss.backward()
            optimizer.step()
            epoch_bce.append(float(bce.detach()))
            epoch_pdaom.append(float(pdaom.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                "bce": float(np.mean(epoch_bce)),
                "pdaom": float(np.mean(epoch_pdaom)),
                "eligible_user_batches": int(eligible_total),
            }
        )

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(valid_x_t), row_cap * 2):
            predictions.append(model(valid_x_t[start:start + row_cap * 2]).numpy())
    logits = np.concatenate(predictions).astype(np.float32)
    if len(logits) != len(valid_y) or not np.isfinite(logits).all():
        raise RuntimeError("Invalid PDAOM predictions")
    return logits, score_metrics(valid_users, valid_y, logits), history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=260830)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--row-cap", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    started_wall, started_cpu = time.monotonic(), time.process_time()
    args.runtime.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    core, selection, evaluation = shared.load_train_only_rows(args.data)
    selection_encoded, selection_dimension = shared.encode_base(core, selection)
    refit_encoded, refit_dimension = shared.encode_base(core + selection, evaluation)
    alpha_grid = [0.0, 0.001, 0.003, 0.01]

    selection_runs: list[dict] = []
    selection_scores: dict[float, np.ndarray] = {}
    for alpha in alpha_grid:
        logits, measured, history = train_and_score(
            encoded=selection_encoded,
            dimension=selection_dimension,
            alpha=alpha,
            seed=args.seed,
            embedding_dim=args.embedding_dim,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            row_cap=args.row_cap,
            epochs=args.epochs,
            threads=args.threads,
        )
        selection_scores[alpha] = logits
        selection_runs.append({"alpha": alpha, "metrics": measured, "history": history})

    selection_control = selection_runs[0]["metrics"]
    eligible = []
    for run in selection_runs:
        gain = deltas(run["metrics"], selection_control)
        run["delta_vs_alpha_zero"] = gain
        run["eligible"] = all(gain[key] >= 0.0 for key in gain)
        if run["eligible"]:
            eligible.append(run)
    selected = sorted(
        eligible,
        key=lambda run: (-run["metrics"]["primary"], run["alpha"]),
    )[0]
    selected_alpha = float(selected["alpha"])

    control_logits, locked_control, control_history = train_and_score(
        encoded=refit_encoded,
        dimension=refit_dimension,
        alpha=0.0,
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        row_cap=args.row_cap,
        epochs=args.epochs,
        threads=args.threads,
    )
    if selected_alpha == 0.0:
        candidate_logits = control_logits.copy()
        locked_candidate = dict(locked_control)
        candidate_history = list(control_history)
    else:
        candidate_logits, locked_candidate, candidate_history = train_and_score(
            encoded=refit_encoded,
            dimension=refit_dimension,
            alpha=selected_alpha,
            seed=args.seed,
            embedding_dim=args.embedding_dim,
            hidden=args.hidden,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            row_cap=args.row_cap,
            epochs=args.epochs,
            threads=args.threads,
        )
    locked_gain = deltas(locked_candidate, locked_control)
    passed = (
        selected_alpha > 0.0
        and locked_gain["primary"] >= 0.0001
        and locked_gain["GAUC"] > 0.0
        and locked_gain["nDCG@5"] > 0.0
    )

    artifact = args.runtime / "screen-scores.npz"
    np.savez_compressed(
        artifact,
        control=control_logits,
        candidate=candidate_logits,
        users=np.asarray(refit_encoded["valid"][2]),
    )
    report = {
        "experiment": "PDAOM maximum-violation personalized GAUC loss",
        "status": "screen_passed" if passed else "rejected_at_train_only_screen",
        "protocol": {
            "model_selection_fit": "2022-04-08..2022-04-11",
            "alpha_selection": "2022-04-12..2022-04-14",
            "locked_refit": "2022-04-08..2022-04-14",
            "locked_screen": "2022-04-15..2022-04-21",
            "confirmation_labels_accessed": False,
            "hidden_test_accessed": False,
            "batching": "complete users packed into bounded row batches",
            "selection_rule": "highest selection primary among all-metric-nonnegative candidates; ties prefer smaller alpha",
            "screen_gate": "selected alpha > 0, primary >= +0.0001, GAUC > 0, nDCG@5 > 0",
        },
        "configuration": {
            "alpha_grid": alpha_grid,
            "selected_alpha": selected_alpha,
            "seed": args.seed,
            "epochs": args.epochs,
            "embedding_dim": args.embedding_dim,
            "hidden": args.hidden,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "row_cap": args.row_cap,
            "threads": args.threads,
        },
        "selection_runs": selection_runs,
        "locked_screen": {
            "alpha_zero_control": locked_control,
            "candidate": locked_candidate,
            "candidate_minus_control": locked_gain,
            "passed": passed,
            "control_history": control_history,
            "candidate_history": candidate_history,
        },
        "rows": {
            "core": len(core),
            "selection": len(selection),
            "screen": len(evaluation),
        },
        "resource_usage": usage(started_wall, started_cpu),
        "hardware": {
            "architecture": platform.machine(),
            "logical_cpu_count": __import__("os").cpu_count(),
        },
        "artifacts": {"scores": str(artifact.relative_to(ROOT)), "script": "scripts/train_pdaom_gauc.py"},
        "recommendation": "Run one locked confirmation." if passed else "Reject without reading April 22+ labels.",
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["locked_screen"], indent=2))
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
