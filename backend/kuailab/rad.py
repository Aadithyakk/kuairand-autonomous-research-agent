from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .champion import within_user_rank


def _empirical_quantiles(
    values: np.ndarray, groups: dict[object, list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed empirical mid-CDF values and group support."""
    quantiles = np.empty(len(values), dtype=np.float32)
    support = np.empty(len(values), dtype=np.float32)
    for indices_list in groups.values():
        indices = np.asarray(indices_list, dtype=np.int64)
        order = np.argsort(values[indices], kind="stable")
        sorted_values = values[indices][order]
        left = np.searchsorted(sorted_values, sorted_values, side="left")
        right = np.searchsorted(sorted_values, sorted_values, side="right")
        # Average tied ranks and divide by n+1. This maps a singleton or an
        # all-tied group to 0.5 and keeps every value strictly inside (0, 1).
        mid_rank = 0.5 * (left + right) + 0.5
        ordered_indices = indices[order]
        quantiles[ordered_indices] = (mid_rank / (len(indices) + 1.0)).astype(np.float32)
        support[indices] = float(len(indices))
    return quantiles, support


def build_rad_labels(rows: Sequence[tuple], duration_bins: int = 4) -> tuple[np.ndarray, dict]:
    """Build train-only Relative Advantage Debiasing preference labels.

    Video-side labels are empirical watch-time CDFs within video ID. User-side
    labels are CDFs within (user ID, duration bin), matching the paper's
    duration-heterogeneity correction. The two views are fused in probit space
    with their group support as reliability weights.
    """
    try:
        from scipy.special import ndtr, ndtri
    except ImportError as error:
        raise RuntimeError("rad_deepfm requires SciPy 1.12 or newer") from error
    if not rows:
        raise ValueError("RAD label estimation requires at least one training row")
    if len(rows[0]) <= 9:
        raise ValueError("RAD label estimation requires play_time_ms at row index 9")

    play_time = np.asarray([max(float(row[9]), 0.0) for row in rows], dtype=np.float64)
    duration = np.asarray([max(float(row[5]), 0.0) for row in rows], dtype=np.float64)
    edges = np.quantile(duration, np.linspace(0.0, 1.0, duration_bins + 1)[1:-1])
    bins = np.searchsorted(edges, duration, side="right")
    video_groups: dict[str, list[int]] = defaultdict(list)
    user_duration_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        video_groups[str(row[2])].append(index)
        user_duration_groups[(str(row[1]), int(bins[index]))].append(index)

    video_quantile, video_support = _empirical_quantiles(play_time, video_groups)
    user_quantile, user_support = _empirical_quantiles(play_time, user_duration_groups)
    clipped_video = np.clip(video_quantile.astype(np.float64), 1e-5, 1.0 - 1e-5)
    clipped_user = np.clip(user_quantile.astype(np.float64), 1e-5, 1.0 - 1e-5)
    video_z = ndtri(clipped_video)
    user_z = ndtri(clipped_user)
    denominator = np.sqrt(np.square(video_support) + np.square(user_support)).clip(1.0)
    fused_z = (video_support * video_z + user_support * user_z) / denominator
    labels = ndtr(fused_z).astype(np.float32)
    stats = {
        "duration_bins": int(duration_bins),
        "duration_edges_ms": [float(value) for value in edges],
        "video_groups": int(len(video_groups)),
        "user_duration_groups": int(len(user_duration_groups)),
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
        "video_support_mean": float(video_support.mean()),
        "user_support_mean": float(user_support.mean()),
    }
    return labels, stats


def train_rad_deepfm(
    splits: dict,
    encoded: dict,
    dimension: int,
    output_dir: Path,
    parameters: dict,
    evaluator: Callable,
) -> tuple[np.ndarray, dict]:
    """Train long_view with a train-only RAD preference auxiliary head."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("rad_deepfm requires PyTorch 2.1 or newer") from error

    class RadDeepFM(nn.Module):
        def __init__(self, fields: int, k: int, hidden: int, dropout: float):
            super().__init__()
            shared_width = max(hidden // 2, 8)
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, k)
            self.shared = nn.Sequential(
                nn.Linear(fields * k, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, shared_width),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.long_view_head = nn.Linear(shared_width, 1)
            self.rad_head = nn.Linear(shared_width, 1)
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (summed.square() - embedding.square().sum(dim=1)).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            shared = self.shared(embedding.flatten(start_dim=1))
            long_view = linear + fm + self.long_view_head(shared).squeeze(-1)
            rad = self.rad_head(shared).squeeze(-1)
            return long_view, rad

    train_rows = splits["train"]
    train_x, train_y, train_users = encoded["train"]
    valid_x, _, valid_users = encoded["valid"]
    maximum_train_date = max(int(row[0]) for row in train_rows)
    if maximum_train_date <= 20220414:
        core_end, dev_start = 20220411, 20220412
        training_split = "2022-04-08..2022-04-11"
        selection_split = "2022-04-12..2022-04-14"
    else:
        core_end, dev_start = 20220414, 20220415
        training_split = "2022-04-08..2022-04-14"
        selection_split = "2022-04-15..2022-04-21"
    core_indices = np.asarray(
        [index for index, row in enumerate(train_rows) if int(row[0]) <= core_end],
        dtype=np.int64,
    )
    dev_indices = np.asarray(
        [
            index for index, row in enumerate(train_rows)
            if dev_start <= int(row[0]) <= maximum_train_date
        ],
        dtype=np.int64,
    )
    core_rows = [train_rows[index] for index in core_indices]
    core_rad, core_rad_stats = build_rad_labels(core_rows)
    core_x, core_y = train_x[core_indices], train_y[core_indices]
    dev_x, dev_y = train_x[dev_indices], train_y[dev_indices]
    dev_users = [train_users[index] for index in dev_indices]

    seed = int(parameters.get("deep_seed", parameters.get("seed", 0)))
    k = max(4, min(int(parameters.get("k", 16)), 32))
    hidden = max(16, min(int(parameters.get("deep_hidden", 64)), 256))
    dropout = max(0.0, min(float(parameters.get("deep_dropout", 0.05)), 0.5))
    learning_rate = max(0.00005, min(float(parameters.get("deep_lr", 0.001)), 0.01))
    epochs = max(3, min(int(parameters.get("deep_epochs", 15)), 40))
    patience = max(2, min(int(parameters.get("deep_patience", 4)), 8))
    batch_size = max(2048, min(int(parameters.get("batch_size", 8192)), 32768))
    positive_weight = max(1.0, min(float(parameters.get("positive_weight", 1.0)), 10.0))
    auxiliary_weight = max(0.0, min(float(parameters.get("rad_aux_weight", 0.2)), 1.0))
    score_weight = max(0.0, min(float(parameters.get("rad_score_weight", 0.0)), 1.0))
    torch.set_num_threads(max(1, min(int(parameters.get("deep_threads", 6)), 16)))

    def tensors(x, y, rad=None):
        values = [
            torch.from_numpy(x.astype(np.int64, copy=False)),
            torch.from_numpy(y.astype(np.float32, copy=False)),
        ]
        if rad is not None:
            values.append(torch.from_numpy(rad.astype(np.float32, copy=False)))
        return values

    core_x_tensor, core_y_tensor, core_rad_tensor = tensors(core_x, core_y, core_rad)
    dev_x_tensor, _ = tensors(dev_x, dev_y)
    valid_x_tensor = torch.from_numpy(valid_x.astype(np.int64, copy=False))

    def predict_heads(model, x_tensor):
        model.eval()
        long_predictions, rad_predictions = [], []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                long_view, rad = model(x_tensor[start:start + batch_size * 4])
                long_predictions.append(long_view.numpy())
                rad_predictions.append(rad.numpy())
        return (
            np.concatenate(long_predictions).astype(np.float32),
            np.concatenate(rad_predictions).astype(np.float32),
        )

    def combine_scores(users, long_scores, rad_scores):
        if score_weight <= 0.0:
            return long_scores
        if score_weight >= 1.0:
            return rad_scores
        long_rank = within_user_rank(users, long_scores)
        rad_rank = within_user_rank(users, rad_scores)
        return (1.0 - score_weight) * long_rank + score_weight * rad_rank

    torch.manual_seed(seed)
    model = RadDeepFM(core_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    random = np.random.default_rng(seed)
    best_score, best_epoch, bad = -1.0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = random.permutation(len(core_y))
        losses, long_losses, rad_losses = [], [], []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            x, y, rad_target = (
                core_x_tensor[indices], core_y_tensor[indices], core_rad_tensor[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            long_view, rad = model(x)
            weights = torch.where(y > 0.5, positive_weight, 1.0)
            long_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    long_view, y, reduction="none"
                ) * weights
            ).sum() / weights.sum()
            rad_loss = nn.functional.binary_cross_entropy_with_logits(rad, rad_target)
            loss = long_loss + auxiliary_weight * rad_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            long_losses.append(float(long_loss.detach()))
            rad_losses.append(float(rad_loss.detach()))
        long_scores, rad_scores = predict_heads(model, dev_x_tensor)
        scores = combine_scores(dev_users, long_scores, rad_scores)
        metrics = evaluator(dev_users, dev_y, scores)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "long_view_loss": float(np.mean(long_losses)),
            "rad_loss": float(np.mean(rad_losses)),
            "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]),
        })
        if metrics["primary"] > best_score + 1e-5:
            best_score, best_epoch, bad = float(metrics["primary"]), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if not best_epoch:
        raise RuntimeError("RAD DeepFM temporal selection produced no checkpoint")

    full_rad, full_rad_stats = build_rad_labels(train_rows)
    full_x_tensor, full_y_tensor, full_rad_tensor = tensors(train_x, train_y, full_rad)
    torch.manual_seed(seed)
    model = RadDeepFM(train_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    refit_random = np.random.default_rng(seed + 7_919)
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        model.train()
        order = refit_random.permutation(len(train_y))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            x, y, rad_target = (
                full_x_tensor[indices], full_y_tensor[indices], full_rad_tensor[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            long_view, rad = model(x)
            weights = torch.where(y > 0.5, positive_weight, 1.0)
            long_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    long_view, y, reduction="none"
                ) * weights
            ).sum() / weights.sum()
            rad_loss = nn.functional.binary_cross_entropy_with_logits(rad, rad_target)
            loss = long_loss + auxiliary_weight * rad_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        refit_history.append({"epoch": epoch, "loss": float(np.mean(losses))})

    valid_long_scores, valid_rad_scores = predict_heads(model, valid_x_tensor)
    valid_scores = combine_scores(valid_users, valid_long_scores, valid_rad_scores)
    torch.save(model.state_dict(), output_dir / f"rad-deep-checkpoint-seed-{seed}.pt")
    np.savez_compressed(
        output_dir / f"rad-head-scores-seed-{seed}.npz",
        long_view_scores=valid_long_scores,
        rad_scores=valid_rad_scores,
        combined_scores=valid_scores,
        rad_score_weight=np.float32(score_weight),
    )
    maximum_date_text = str(maximum_train_date)
    maximum_date_label = (
        f"{maximum_date_text[:4]}-{maximum_date_text[4:6]}-{maximum_date_text[6:]}"
    )
    return valid_scores, {
        "objective": "long_view_bce_plus_rad_uv_quantile_auxiliary",
        "seed": seed,
        "rad_aux_weight": auxiliary_weight,
        "rad_score_weight": score_weight,
        "training_split": training_split,
        "selection_split": selection_split,
        "refit_split": f"2022-04-08..{maximum_date_label}",
        "confirmation_split": (
            "2022-04-15..2022-04-21"
            if maximum_train_date <= 20220414 else "2022-04-22..2022-04-28"
        ),
        "selected_epoch": int(best_epoch),
        "core_rad": core_rad_stats,
        "full_rad": full_rad_stats,
        "epochs": history,
        "refit_epochs": refit_history,
    }
