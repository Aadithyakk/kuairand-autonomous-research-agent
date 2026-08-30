#!/usr/bin/env python3
"""Leak-free SetRank-style attention residual screen on KuaiRand-Pure.

The experiment first fits a pointwise BCE DeepSets control, freezes it, then
learns a small permutation-equivariant self-attention residual over each
complete user slate.  Epochs and residual strength are selected on Apr 12-14;
the model is refit through Apr 14 and evaluated once on Apr 15-21.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.kuailab.resources import ProcessResourceTracker
from backend.kuailab.slate import _group_batches, build_slate_features
from scripts import kuairand_runner as runner


def metric_record(users: Sequence[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    result = runner.evaluate_module.evaluate(users, labels, scores)
    return {
        "primary": float(result["primary"]),
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "users": int(result["users"]),
        "rows": int(result["rows"]),
    }


def normalize_numeric(
    train_rows: Sequence[tuple], *other_rows: Sequence[tuple],
) -> tuple[list[np.ndarray], list[str]]:
    train, names = build_slate_features(train_rows)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0

    def apply(values: np.ndarray) -> np.ndarray:
        return np.clip((values - mean) / scale, -8.0, 8.0).astype(np.float32)

    arrays = [apply(train)]
    for rows in other_rows:
        values, _ = build_slate_features(rows)
        arrays.append(apply(values))
    return arrays, names


def padded_batches(
    users: Sequence[str], max_cells: int, random: np.random.Generator | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pack complete user groups into length-bucketed padded batches."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        grouped[str(user)].append(index)
    groups = sorted(
        (np.asarray(indices, dtype=np.int64) for indices in grouped.values()),
        key=len,
    )
    packed: list[list[np.ndarray]] = []
    pending: list[np.ndarray] = []
    maximum = 0
    for group in groups:
        proposed_maximum = max(maximum, len(group))
        if pending and proposed_maximum * (len(pending) + 1) > max_cells:
            packed.append(pending)
            pending, maximum = [], 0
        pending.append(group)
        maximum = max(maximum, len(group))
    if pending:
        packed.append(pending)
    if random is not None:
        random.shuffle(packed)

    output = []
    for batch in packed:
        width = max(len(group) for group in batch)
        indices = np.full((len(batch), width), -1, dtype=np.int64)
        mask = np.zeros((len(batch), width), dtype=bool)
        for row, group in enumerate(batch):
            indices[row, : len(group)] = group
            mask[row, : len(group)] = True
        output.append((indices, mask))
    return output


class SetRankResidual(nn.Module):
    """DeepSets pointwise control plus bounded SetRank-style correction."""

    def __init__(
        self, dimension: int, fields: int, numeric: int, embedding_dim: int = 16,
        hidden: int = 64, attention_dim: int = 32, heads: int = 2,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.linear = nn.Embedding(dimension, 1)
        self.embedding = nn.Embedding(dimension, embedding_dim)
        self.row_encoder = nn.Sequential(
            nn.Linear(fields * embedding_dim + numeric, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.base_head = nn.Sequential(
            nn.Linear(hidden * 2 + numeric, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden // 2, 8), 1),
        )
        # These modules are trained only after the pointwise/mean-pool base is
        # frozen. No positional embeddings are used, preserving permutation
        # equivariance within a user slate.
        self.attention_projection = nn.Linear(hidden, attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim, heads, dropout=0.0, batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(attention_dim * 2 + numeric, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.residual_head[-1].weight, std=0.002)
        nn.init.zeros_(self.residual_head[-1].bias)

    def base_parameters(self):
        for module in (self.linear, self.embedding, self.row_encoder, self.base_head):
            yield from module.parameters()

    def residual_parameters(self):
        for module in (
            self.attention_projection, self.attention, self.attention_norm,
            self.residual_head,
        ):
            yield from module.parameters()

    def encode_rows(self, x: torch.Tensor, numeric: torch.Tensor):
        embedding = self.embedding(x)
        summed = embedding.sum(dim=-2)
        fm = 0.5 * (
            summed.square() - embedding.square().sum(dim=-2)
        ).sum(dim=-1)
        linear = self.linear(x).sum(dim=-2).squeeze(-1)
        row = self.row_encoder(
            torch.cat([embedding.flatten(start_dim=-2), numeric], dim=-1)
        )
        return linear + fm, row

    def base_flat(
        self, x: torch.Tensor, numeric: torch.Tensor, group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shallow, row = self.encode_rows(x, numeric)
        groups = int(group_ids.max().item()) + 1
        context = torch.zeros(groups, row.shape[-1], dtype=row.dtype)
        context.index_add_(0, group_ids, row)
        counts = torch.bincount(group_ids, minlength=groups).to(row.dtype).unsqueeze(1)
        context = context / counts.clamp_min(1.0)
        deep = self.base_head(
            torch.cat([row, context[group_ids], numeric], dim=-1)
        ).squeeze(-1)
        return shallow + deep, row

    def base_padded(
        self, x: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shallow, row = self.encode_rows(x, numeric)
        weights = mask.to(row.dtype).unsqueeze(-1)
        context = (row * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        deep = self.base_head(
            torch.cat([row, context.unsqueeze(1).expand_as(row), numeric], dim=-1)
        ).squeeze(-1)
        return shallow + deep, row

    def attention_padded(
        self, x: torch.Tensor, numeric: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            base, row = self.base_padded(x, numeric, mask)
        projected = self.attention_projection(row)
        attended, _ = self.attention(
            projected, projected, projected, key_padding_mask=~mask,
            need_weights=False,
        )
        attended = self.attention_norm(projected + attended)
        residual = self.residual_head(
            torch.cat([projected, attended, numeric], dim=-1)
        ).squeeze(-1)
        return base, base + self.residual_scale * torch.tanh(residual)


def tensor_bundle(x: np.ndarray, y: np.ndarray, numeric: np.ndarray) -> dict:
    return {
        "x": torch.from_numpy(x.astype(np.int64, copy=False)),
        "y": torch.from_numpy(y.astype(np.float32, copy=False)),
        "numeric": torch.from_numpy(numeric.astype(np.float32, copy=False)),
    }


def predict_base(
    model: SetRankResidual, data: dict, users: Sequence[str], batch_rows: int,
) -> np.ndarray:
    model.eval()
    output = np.empty(len(users), dtype=np.float32)
    with torch.no_grad():
        for indices, group_ids in _group_batches(users, batch_rows * 4, None):
            values, _ = model.base_flat(
                data["x"][indices], data["numeric"][indices],
                torch.from_numpy(group_ids),
            )
            output[indices] = values.numpy()
    return output


def predict_attention(
    model: SetRankResidual, data: dict, users: Sequence[str], max_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    base_output = np.empty(len(users), dtype=np.float32)
    attention_output = np.empty(len(users), dtype=np.float32)
    with torch.no_grad():
        for indices, mask in padded_batches(users, max_cells, None):
            safe = np.maximum(indices, 0)
            mask_tensor = torch.from_numpy(mask)
            base, attention = model.attention_padded(
                data["x"][safe], data["numeric"][safe], mask_tensor,
            )
            flat = mask
            base_output[indices[flat]] = base.numpy()[flat]
            attention_output[indices[flat]] = attention.numpy()[flat]
    return base_output, attention_output


def train_base(
    model: SetRankResidual, train: dict, train_users: Sequence[str],
    valid: dict | None, valid_users: Sequence[str] | None, evaluator,
    *, epochs: int, batch_rows: int, seed: int, patience: int | None,
) -> tuple[int, list[dict]]:
    optimizer = torch.optim.AdamW(
        list(model.base_parameters()), lr=0.001, weight_decay=1e-6,
    )
    random = np.random.default_rng(seed)
    best_epoch, best_score, best_state, bad = 1, -1.0, None, 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices, group_ids in _group_batches(train_users, batch_rows, random):
            group = torch.from_numpy(group_ids)
            labels = train["y"][indices]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model.base_flat(
                train["x"][indices], train["numeric"][indices], group,
            )
            counts = torch.bincount(group).to(labels.dtype)
            user_weight = (len(group_ids) / max(len(counts), 1)) / counts[group]
            weights = 0.5 + 0.5 * user_weight
            loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none",
                ) * weights
            ).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        record = {"epoch": epoch, "loss": float(np.mean(losses))}
        if valid is not None and valid_users is not None:
            scores = predict_base(model, valid, valid_users, batch_rows)
            metrics = evaluator(valid_users, valid["y"].numpy(), scores)
            record.update(metric_record(valid_users, valid["y"].numpy(), scores))
            score = float(metrics["primary"])
            if score > best_score + 1e-5:
                best_epoch, best_score, bad = epoch, score, 0
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            else:
                bad += 1
        history.append(record)
        if patience is not None and bad >= patience:
            break
    if valid is not None:
        if best_state is None:
            raise RuntimeError("pointwise control produced no selection checkpoint")
        model.load_state_dict(best_state)
    return best_epoch, history


def set_residual_training_mode(model: SetRankResidual) -> None:
    model.eval()
    for module in (
        model.attention_projection, model.attention, model.attention_norm,
        model.residual_head,
    ):
        module.train()


def train_residual(
    model: SetRankResidual, train: dict, train_users: Sequence[str],
    valid: dict | None, valid_users: Sequence[str] | None,
    *, epochs: int, max_cells: int, seed: int, patience: int | None,
) -> tuple[int, float, list[dict]]:
    for parameter in model.base_parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(model.residual_parameters()), lr=0.0005, weight_decay=1e-5,
    )
    random = np.random.default_rng(seed + 1)
    blend_grid = (0.0, 0.25, 0.5, 1.0)
    best_epoch, best_weight, best_score, best_state, bad = 1, 0.0, -1.0, None, 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        set_residual_training_mode(model)
        losses = []
        for indices, mask in padded_batches(train_users, max_cells, random):
            safe = np.maximum(indices, 0)
            valid_mask = torch.from_numpy(mask)
            labels = train["y"][safe]
            optimizer.zero_grad(set_to_none=True)
            _, logits = model.attention_padded(
                train["x"][safe], train["numeric"][safe], valid_mask,
            )
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[valid_mask], labels[valid_mask],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.residual_parameters()), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        record: dict = {"epoch": epoch, "loss": float(np.mean(losses))}
        if valid is not None and valid_users is not None:
            base, raw = predict_attention(model, valid, valid_users, max_cells)
            raw_metrics = metric_record(valid_users, valid["y"].numpy(), raw)
            scans = []
            for weight in blend_grid:
                scores = base + weight * (raw - base)
                scans.append((weight, metric_record(valid_users, valid["y"].numpy(), scores)))
            weight, selected = max(
                scans,
                key=lambda item: (round(item[1]["primary"], 12), -item[0]),
            )
            record.update({
                "raw_attention": raw_metrics,
                "selected_weight": weight,
                "selected_metrics": selected,
            })
            if selected["primary"] > best_score + 1e-5:
                best_epoch, best_weight, best_score, bad = epoch, weight, selected["primary"], 0
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                    if name.startswith((
                        "attention_projection", "attention.",
                        "attention_norm", "residual_head",
                    ))
                }
            else:
                bad += 1
        history.append(record)
        if patience is not None and bad >= patience:
            break
    if valid is not None:
        if best_state is None:
            raise RuntimeError("attention residual produced no selection checkpoint")
        model.load_state_dict(best_state, strict=False)
    return best_epoch, best_weight, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runtime" / "parallel-setrank",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "results" / "parallel-methods" / "setrank-screen.json",
    )
    parser.add_argument("--seed", type=int, default=1223)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--base-epochs", type=int, default=8)
    parser.add_argument("--residual-epochs", type=int, default=4)
    parser.add_argument("--batch-rows", type=int, default=8192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, min(args.threads, 8)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tracker = ProcessResourceTracker()
    started = time.monotonic()

    screen = runner.load_screen_splits(ROOT / "external" / "KuaiRand-Pure" / "data")
    full_rows, valid_rows = screen["train"], screen["valid"]
    core_rows = [row for row in full_rows if int(row[0]) <= 20220411]
    dev_rows = [row for row in full_rows if int(row[0]) >= 20220412]

    selection_splits = {"train": core_rows, "valid": dev_rows, "test": []}
    selection_encoded, selection_dimension = runner.data_module.encode(selection_splits)
    (core_numeric, dev_numeric), feature_names = normalize_numeric(core_rows, dev_rows)
    core_x, core_y, core_users = selection_encoded["train"]
    dev_x, dev_y, dev_users = selection_encoded["valid"]
    core = tensor_bundle(core_x, core_y, core_numeric)
    dev = tensor_bundle(dev_x, dev_y, dev_numeric)

    model = SetRankResidual(
        selection_dimension, core_x.shape[1], core_numeric.shape[1],
    )
    selected_base_epoch, base_history = train_base(
        model, core, core_users, dev, dev_users, runner.evaluate_module.evaluate,
        epochs=args.base_epochs, batch_rows=args.batch_rows, seed=args.seed,
        patience=3,
    )
    selected_residual_epoch, selected_weight, residual_history = train_residual(
        model, core, core_users, dev, dev_users,
        epochs=args.residual_epochs, max_cells=args.batch_rows,
        seed=args.seed, patience=2,
    )

    # Rebuild all vocabularies and normalization using Apr 8-14 only, refit for
    # exactly the selected epoch counts, and make the sole Apr 15-21 report.
    refit_splits = {"train": full_rows, "valid": valid_rows, "test": []}
    refit_encoded, refit_dimension = runner.data_module.encode(refit_splits)
    (full_numeric, valid_numeric), _ = normalize_numeric(full_rows, valid_rows)
    full_x, full_y, full_users = refit_encoded["train"]
    valid_x, valid_y, valid_users = refit_encoded["valid"]
    full = tensor_bundle(full_x, full_y, full_numeric)
    valid = tensor_bundle(valid_x, valid_y, valid_numeric)

    torch.manual_seed(args.seed)
    refit_model = SetRankResidual(
        refit_dimension, full_x.shape[1], full_numeric.shape[1],
    )
    _, refit_base_history = train_base(
        refit_model, full, full_users, None, None, runner.evaluate_module.evaluate,
        epochs=selected_base_epoch, batch_rows=args.batch_rows, seed=args.seed,
        patience=None,
    )
    _, _, refit_residual_history = train_residual(
        refit_model, full, full_users, None, None,
        epochs=selected_residual_epoch, max_cells=args.batch_rows,
        seed=args.seed, patience=None,
    )
    base_scores, raw_attention_scores = predict_attention(
        refit_model, valid, valid_users, args.batch_rows,
    )
    selected_scores = base_scores + selected_weight * (raw_attention_scores - base_scores)
    base_metrics = metric_record(valid_users, valid_y, base_scores)
    raw_metrics = metric_record(valid_users, valid_y, raw_attention_scores)
    selected_metrics = metric_record(valid_users, valid_y, selected_scores)
    usage = tracker.finish(train_seconds=time.monotonic() - started)

    checkpoint_path = args.output_dir / "setrank-screen-checkpoint.pt"
    torch.save(refit_model.state_dict(), checkpoint_path)
    np.savez_compressed(
        args.output_dir / "setrank-screen-scores.npz",
        base_scores=base_scores,
        raw_attention_scores=raw_attention_scores,
        selected_scores=selected_scores,
    )
    report = {
        "method": "SetRank-style bounded whole-user-slate attention residual",
        "status": (
            "promising" if selected_metrics["primary"] > base_metrics["primary"]
            else "rejected_on_train_only_screen"
        ),
        "protocol": {
            "base_training": "2022-04-08..2022-04-11",
            "selection": "2022-04-12..2022-04-14",
            "refit": "2022-04-08..2022-04-14",
            "evaluation": "2022-04-15..2022-04-21",
            "hidden_confirmation_labels_read": False,
            "unit": "complete user slate; no positional encoding",
            "objective": "pointwise BCE; frozen mean-pool base plus bounded attention residual",
        },
        "configuration": {
            "seed": args.seed,
            "embedding_dim": 16,
            "hidden_dim": 64,
            "attention_dim": 32,
            "attention_heads": 2,
            "maximum_logit_residual": 0.15,
            "selected_base_epoch": selected_base_epoch,
            "selected_residual_epoch": selected_residual_epoch,
            "selected_residual_weight": selected_weight,
            "numeric_features": feature_names,
        },
        "metrics": {
            "matched_mean_pool_control": base_metrics,
            "raw_attention": raw_metrics,
            "selected_attention": selected_metrics,
        },
        "deltas_vs_control": {
            key: selected_metrics[key] - base_metrics[key]
            for key in ("primary", "gauc", "ndcg5")
        },
        "selection_history": {
            "base": base_history,
            "attention": residual_history,
        },
        "refit_history": {
            "base": refit_base_history,
            "attention": refit_residual_history,
        },
        "resource_usage": usage,
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "scores": str((args.output_dir / "setrank-screen-scores.npz").relative_to(ROOT)),
        },
        "recommendation": (
            "Audit stability before confirmation; do not promote from this screen alone."
            if selected_metrics["primary"] > base_metrics["primary"]
            else "Do not confirm this SetRank residual; retain the mean-pool control."
        ),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "setrank-screen.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "metrics": report["metrics"],
        "deltas_vs_control": report["deltas_vs_control"],
        "resource_usage": usage,
        "report": str(args.report),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
