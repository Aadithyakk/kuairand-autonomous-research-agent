from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def build_slate_features(rows: Sequence[tuple]) -> tuple[np.ndarray, list[str]]:
    """Build outcome-free features in the evaluator's whole-user slate unit."""
    names = [
        "log_slate_rows", "active_days", "log_session_count",
        "session_fraction", "log_session_rows", "session_position_fraction",
        "log_video_count", "video_occurrence_fraction",
        "log_author_count", "author_occurrence_fraction",
        "log_gap_minutes", "relative_day", "hour_sin", "hour_cos",
        "log_duration_seconds",
    ]
    output = np.zeros((len(rows), len(names)), dtype=np.float32)
    by_user: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_user[str(row[1])].append(index)

    for indices_list in by_user.values():
        indices = sorted(indices_list, key=lambda index: (int(rows[index][8]), index))
        video_counts = Counter(str(rows[index][2]) for index in indices)
        author_counts = Counter(str(rows[index][3]) for index in indices)
        video_seen: Counter[str] = Counter()
        author_seen: Counter[str] = Counter()
        dates = sorted({int(rows[index][0]) for index in indices})
        date_rank = {date: rank for rank, date in enumerate(dates)}

        sessions: list[list[int]] = []
        for index in indices:
            if not sessions or int(rows[index][8]) - int(rows[sessions[-1][-1]][8]) > 1_800_000:
                sessions.append([])
            sessions[-1].append(index)

        for session_number, session in enumerate(sessions):
            for position, index in enumerate(session):
                row = rows[index]
                video, author = str(row[2]), str(row[3])
                timestamp = int(row[8])
                previous_timestamp = int(rows[session[position - 1]][8]) if position else timestamp
                hour = int(row[7])
                output[index] = np.asarray([
                    np.log1p(len(indices)),
                    len(dates) / 7.0,
                    np.log1p(len(sessions)),
                    session_number / max(len(sessions) - 1, 1),
                    np.log1p(len(session)),
                    position / max(len(session) - 1, 1),
                    np.log1p(video_counts[video]),
                    video_seen[video] / max(video_counts[video] - 1, 1),
                    np.log1p(author_counts[author]),
                    author_seen[author] / max(author_counts[author] - 1, 1),
                    np.log1p(max(timestamp - previous_timestamp, 0) / 60_000.0),
                    date_rank[int(row[0])] / max(len(dates) - 1, 1),
                    np.sin(2.0 * np.pi * hour / 24.0),
                    np.cos(2.0 * np.pi * hour / 24.0),
                    np.log1p(max(float(row[5]), 0.0) / 1000.0),
                ], dtype=np.float32)
                video_seen[video] += 1
                author_seen[author] += 1
    return output, names


def _group_batches(
    users: Sequence[str], max_rows: int, random: np.random.Generator | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        grouped[str(user)].append(index)
    groups = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
    if random is not None:
        random.shuffle(groups)
    batches: list[tuple[np.ndarray, np.ndarray]] = []
    pending: list[np.ndarray] = []
    pending_rows = 0
    for group in groups:
        if pending and pending_rows + len(group) > max_rows:
            indices = np.concatenate(pending)
            group_ids = np.concatenate([
                np.full(len(value), number, dtype=np.int64)
                for number, value in enumerate(pending)
            ])
            batches.append((indices, group_ids))
            pending, pending_rows = [], 0
        pending.append(group)
        pending_rows += len(group)
    if pending:
        indices = np.concatenate(pending)
        group_ids = np.concatenate([
            np.full(len(value), number, dtype=np.int64)
            for number, value in enumerate(pending)
        ])
        batches.append((indices, group_ids))
    return batches


def train_slate_deepfm(
    splits: dict,
    encoded: dict,
    dimension: int,
    output_dir: Path,
    parameters: dict,
    evaluator: Callable,
) -> tuple[np.ndarray, dict]:
    """Train a user-slate DeepSets/DeepFM hybrid with temporal early stopping."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("slate_context_deepfm requires PyTorch 2.1 or newer") from error

    class SlateDeepFM(nn.Module):
        def __init__(self, fields: int, numeric: int, k: int, hidden: int, dropout: float):
            super().__init__()
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, k)
            self.row_encoder = nn.Sequential(
                nn.Linear(fields * k + numeric, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2 + numeric, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, max(hidden // 2, 8)),
                nn.ReLU(),
                nn.Linear(max(hidden // 2, 8), 1),
            )
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x, numeric, group_ids):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (summed.square() - embedding.square().sum(dim=1)).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            row_state = self.row_encoder(torch.cat([embedding.flatten(start_dim=1), numeric], dim=1))
            groups = int(group_ids.max().item()) + 1
            context = torch.zeros(
                groups, row_state.shape[1], dtype=row_state.dtype, device=row_state.device,
            )
            context.index_add_(0, group_ids, row_state)
            counts = torch.bincount(group_ids, minlength=groups).to(row_state.dtype).unsqueeze(1)
            context = context / counts.clamp_min(1.0)
            deep = self.head(torch.cat([row_state, context[group_ids], numeric], dim=1)).squeeze(-1)
            return linear + fm + deep

    train_rows = splits["train"]
    train_x_all, train_y_all, train_users_all = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
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
        [index for index, row in enumerate(train_rows) if dev_start <= int(row[0]) <= maximum_train_date],
        dtype=np.int64,
    )
    core_rows = [train_rows[index] for index in core_indices]
    dev_rows = [train_rows[index] for index in dev_indices]
    valid_rows = splits["valid"]
    core_numeric, feature_names = build_slate_features(core_rows)
    dev_numeric, _ = build_slate_features(dev_rows)
    valid_numeric, _ = build_slate_features(valid_rows)
    mean = core_numeric.mean(axis=0, keepdims=True)
    scale = core_numeric.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    core_numeric = np.clip((core_numeric - mean) / scale, -8.0, 8.0).astype(np.float32)
    dev_numeric = np.clip((dev_numeric - mean) / scale, -8.0, 8.0).astype(np.float32)
    valid_numeric = np.clip((valid_numeric - mean) / scale, -8.0, 8.0).astype(np.float32)

    core_x = train_x_all[core_indices]
    core_y = train_y_all[core_indices]
    core_users = [train_users_all[index] for index in core_indices]
    dev_x = train_x_all[dev_indices]
    dev_y = train_y_all[dev_indices]
    dev_users = [train_users_all[index] for index in dev_indices]

    seed = int(parameters.get("deep_seed", parameters.get("seed", 0)))
    k = max(4, min(int(parameters.get("k", 16)), 32))
    hidden = max(16, min(int(parameters.get("deep_hidden", 64)), 192))
    dropout = max(0.0, min(float(parameters.get("deep_dropout", 0.08)), 0.5))
    learning_rate = max(0.00005, min(float(parameters.get("deep_lr", 0.001)), 0.01))
    epochs = max(3, min(int(parameters.get("deep_epochs", 15)), 30))
    patience = max(2, min(int(parameters.get("deep_patience", 4)), 8))
    batch_rows = max(2048, min(int(parameters.get("batch_size", 8192)), 32768))
    positive_weight = max(1.0, min(float(parameters.get("positive_weight", 1.0)), 10.0))
    torch.set_num_threads(max(1, min(int(parameters.get("deep_threads", 6)), 16)))
    torch.manual_seed(seed)
    random = np.random.default_rng(seed)

    tensors = {
        "core_x": torch.from_numpy(core_x.astype(np.int64, copy=False)),
        "core_y": torch.from_numpy(core_y),
        "core_numeric": torch.from_numpy(core_numeric),
        "dev_x": torch.from_numpy(dev_x.astype(np.int64, copy=False)),
        "dev_numeric": torch.from_numpy(dev_numeric),
        "valid_x": torch.from_numpy(valid_x.astype(np.int64, copy=False)),
        "valid_numeric": torch.from_numpy(valid_numeric),
    }
    model = SlateDeepFM(core_x.shape[1], core_numeric.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    best_score, best_state, bad = -1.0, None, 0
    history = []

    def predict(prefix: str, users: Sequence[str]) -> np.ndarray:
        model.eval()
        scores = np.empty(len(users), dtype=np.float32)
        with torch.no_grad():
            for indices, group_ids in _group_batches(users, batch_rows * 4, None):
                predictions = model(
                    tensors[f"{prefix}_x"][indices],
                    tensors[f"{prefix}_numeric"][indices],
                    torch.from_numpy(group_ids),
                )
                scores[indices] = predictions.numpy()
        return scores

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices, group_ids in _group_batches(core_users, batch_rows, random):
            group_tensor = torch.from_numpy(group_ids)
            y = tensors["core_y"][indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                tensors["core_x"][indices], tensors["core_numeric"][indices], group_tensor,
            )
            group_counts = torch.bincount(group_tensor).to(y.dtype)
            user_weight = (len(group_ids) / max(len(group_counts), 1)) / group_counts[group_tensor]
            weights = (0.5 + 0.5 * user_weight) * torch.where(y > 0.5, positive_weight, 1.0)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights
            ).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_scores = predict("dev", dev_users)
        metrics = evaluator(dev_users, dev_y, dev_scores)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selection_split": selection_split,
            "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]),
            "users": int(metrics["users"]),
            "rows": int(metrics["rows"]),
        })
        if metrics["primary"] > best_score + 1e-5:
            best_score, bad = float(metrics["primary"]), 0
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("Slate-context DeepFM training produced no checkpoint")

    # Refit both complete seven-day slates for exactly the temporally selected
    # epoch count. The confirmation labels remain unavailable to model choice.
    selected_epoch = max(history, key=lambda item: item["primary"])["epoch"]
    full_x = np.concatenate([core_x, dev_x])
    full_y = np.concatenate([core_y, dev_y])
    full_numeric = np.concatenate([core_numeric, dev_numeric])
    full_users = [f"week1:{user}" for user in core_users] + [
        f"week2:{user}" for user in dev_users
    ]
    tensors.update({
        "full_x": torch.from_numpy(full_x.astype(np.int64, copy=False)),
        "full_y": torch.from_numpy(full_y),
        "full_numeric": torch.from_numpy(full_numeric),
    })
    torch.manual_seed(seed)
    model = SlateDeepFM(core_x.shape[1], core_numeric.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    refit_history = []
    for epoch in range(1, int(selected_epoch) + 1):
        model.train()
        losses = []
        for indices, group_ids in _group_batches(full_users, batch_rows, random):
            group_tensor = torch.from_numpy(group_ids)
            y = tensors["full_y"][indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                tensors["full_x"][indices], tensors["full_numeric"][indices], group_tensor,
            )
            group_counts = torch.bincount(group_tensor).to(y.dtype)
            user_weight = (len(group_ids) / max(len(group_counts), 1)) / group_counts[group_tensor]
            weights = (0.5 + 0.5 * user_weight) * torch.where(y > 0.5, positive_weight, 1.0)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights
            ).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        refit_history.append({"epoch": epoch, "loss": float(np.mean(losses))})
    valid_scores = predict("valid", valid_users)
    torch.save(model.state_dict(), output_dir / f"slate-deep-checkpoint-seed-{seed}.pt")
    return valid_scores, {
        "objective": "user_balanced_bce_with_deepsets_context",
        "seed": seed,
        "training_split": training_split,
        "selection_split": selection_split,
        "refit_split": f"independent temporal slates through {maximum_train_date}",
        "selected_epoch": int(selected_epoch),
        "confirmation_split": "2022-04-22..2022-04-28",
        "features": feature_names,
        "epochs": history,
        "refit_epochs": refit_history,
    }
