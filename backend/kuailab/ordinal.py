from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


ORDINAL_THRESHOLDS = (0.25, 0.5, 0.75)
PROFILE_FIELDS = (
    "user_active_degree", "is_lowactive_period", "is_live_streamer",
    "is_video_author", "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
    *[f"onehot_feat{index}" for index in range(18)],
)


def augment_with_user_profiles(
    splits: dict, encoded: dict, dimension: int,
) -> tuple[dict, int, dict]:
    """Append train-vocabulary static user-profile fields to encoded rows."""
    profile_path = (
        Path(__file__).resolve().parents[2]
        / "external" / "KuaiRand-Pure" / "data" / "user_features_pure.csv"
    )
    profiles: dict[str, tuple[str, ...]] = {}
    with profile_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            profiles[str(row["user_id"])] = tuple(
                str(row.get(field) or "UNK") for field in PROFILE_FIELDS
            )
    train_users = [str(row[1]) for row in splits["train"]]
    vocabularies: list[dict[str, int]] = [dict() for _ in PROFILE_FIELDS]
    for user in set(train_users):
        values = profiles.get(user, tuple("UNK" for _ in PROFILE_FIELDS))
        for field, value in enumerate(values):
            vocabulary = vocabularies[field]
            if value not in vocabulary:
                vocabulary[value] = len(vocabulary)
    offsets = np.cumsum(
        [dimension] + [len(vocabulary) + 1 for vocabulary in vocabularies[:-1]],
        dtype=np.int64,
    )
    augmented = {}
    for split_name, (x, y, users) in encoded.items():
        profile_x = np.empty((len(users), len(PROFILE_FIELDS)), dtype=np.int32)
        for row_index, user in enumerate(users):
            values = profiles.get(str(user), tuple("UNK" for _ in PROFILE_FIELDS))
            for field, value in enumerate(values):
                vocabulary = vocabularies[field]
                profile_x[row_index, field] = (
                    vocabulary.get(value, len(vocabulary)) + int(offsets[field])
                )
        augmented[split_name] = (
            np.concatenate([x, profile_x], axis=1), y, users,
        )
    new_dimension = dimension + sum(len(vocabulary) + 1 for vocabulary in vocabularies)
    return augmented, new_dimension, {
        "fields": list(PROFILE_FIELDS),
        "field_cardinalities": [len(vocabulary) + 1 for vocabulary in vocabularies],
        "profile_rows": len(profiles),
    }


def build_ordinal_watch_labels(
    rows: Sequence[tuple], thresholds: tuple[float, ...] = ORDINAL_THRESHOLDS,
) -> tuple[np.ndarray, dict]:
    """Encode nested progress-to-long-view thresholds from training outcomes."""
    if not rows or len(rows[0]) <= 9:
        raise ValueError("Ordinal watch labels require duration and play_time_ms")
    duration = np.asarray([max(float(row[5]), 1.0) for row in rows], dtype=np.float32)
    play_time = np.asarray([max(float(row[9]), 0.0) for row in rows], dtype=np.float32)
    # KuaiRand long_view is mechanically capped around 18 seconds. Intermediate
    # nested targets give near-threshold watches useful gradients without being
    # used at inference.
    progress = play_time / np.minimum(duration, 18_000.0)
    labels = np.stack([progress >= threshold for threshold in thresholds], axis=1).astype(
        np.float32
    )
    return labels, {
        "thresholds": list(thresholds),
        "positive_rates": [float(value) for value in labels.mean(axis=0)],
        "progress_mean": float(progress.mean()),
    }


def train_ordinal_deepfm(
    splits: dict,
    encoded: dict,
    dimension: int,
    output_dir: Path,
    parameters: dict,
    evaluator: Callable,
) -> tuple[np.ndarray, dict]:
    """Train a pointwise DeepFM with nested watch-progress auxiliary heads."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("ordinal_watch_deepfm requires PyTorch 2.1 or newer") from error

    class OrdinalDeepFM(nn.Module):
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
            self.ordinal_head = nn.Linear(shared_width, len(ORDINAL_THRESHOLDS))
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (summed.square() - embedding.square().sum(dim=1)).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            shared = self.shared(embedding.flatten(start_dim=1))
            long_view = linear + fm + self.long_view_head(shared).squeeze(-1)
            return long_view, self.ordinal_head(shared)

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
    core_ordinal, core_stats = build_ordinal_watch_labels(core_rows)
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
    auxiliary_weight = max(
        0.0, min(float(parameters.get("ordinal_aux_weight", 0.15)), 1.0)
    )
    torch.set_num_threads(max(1, min(int(parameters.get("deep_threads", 6)), 16)))

    def tensors(x, y, ordinal=None):
        values = [
            torch.from_numpy(x.astype(np.int64, copy=False)),
            torch.from_numpy(y.astype(np.float32, copy=False)),
        ]
        if ordinal is not None:
            values.append(torch.from_numpy(ordinal.astype(np.float32, copy=False)))
        return values

    core_x_tensor, core_y_tensor, core_ordinal_tensor = tensors(
        core_x, core_y, core_ordinal
    )
    dev_x_tensor, _ = tensors(dev_x, dev_y)
    valid_x_tensor = torch.from_numpy(valid_x.astype(np.int64, copy=False))

    def predict(model, x_tensor):
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                long_view, _ = model(x_tensor[start:start + batch_size * 4])
                predictions.append(long_view.numpy())
        return np.concatenate(predictions).astype(np.float32)

    def train_epoch(model, optimizer, x_tensor, y_tensor, ordinal_tensor, order):
        model.train()
        losses, long_losses, ordinal_losses = [], [], []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            x, y, ordinal_target = (
                x_tensor[indices], y_tensor[indices], ordinal_tensor[indices]
            )
            optimizer.zero_grad(set_to_none=True)
            long_view, ordinal = model(x)
            weights = torch.where(y > 0.5, positive_weight, 1.0)
            long_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    long_view, y, reduction="none"
                ) * weights
            ).sum() / weights.sum()
            ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
                ordinal, ordinal_target
            )
            loss = long_loss + auxiliary_weight * ordinal_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            long_losses.append(float(long_loss.detach()))
            ordinal_losses.append(float(ordinal_loss.detach()))
        return float(np.mean(losses)), float(np.mean(long_losses)), float(
            np.mean(ordinal_losses)
        )

    torch.manual_seed(seed)
    model = OrdinalDeepFM(core_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    random = np.random.default_rng(seed)
    best_score, best_epoch, bad = -1.0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        losses = train_epoch(
            model, optimizer, core_x_tensor, core_y_tensor, core_ordinal_tensor,
            random.permutation(len(core_y)),
        )
        scores = predict(model, dev_x_tensor)
        metrics = evaluator(dev_users, dev_y, scores)
        history.append({
            "epoch": epoch,
            "loss": losses[0],
            "long_view_loss": losses[1],
            "ordinal_loss": losses[2],
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
        raise RuntimeError("Ordinal DeepFM temporal selection produced no checkpoint")

    full_ordinal, full_stats = build_ordinal_watch_labels(train_rows)
    full_x_tensor, full_y_tensor, full_ordinal_tensor = tensors(
        train_x, train_y, full_ordinal
    )
    torch.manual_seed(seed)
    model = OrdinalDeepFM(train_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    refit_random = np.random.default_rng(seed + 8_129)
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        losses = train_epoch(
            model, optimizer, full_x_tensor, full_y_tensor, full_ordinal_tensor,
            refit_random.permutation(len(train_y)),
        )
        refit_history.append({"epoch": epoch, "loss": losses[0]})

    valid_scores = predict(model, valid_x_tensor)
    torch.save(model.state_dict(), output_dir / f"ordinal-deep-checkpoint-seed-{seed}.pt")
    maximum_date_text = str(maximum_train_date)
    maximum_date_label = (
        f"{maximum_date_text[:4]}-{maximum_date_text[4:6]}-{maximum_date_text[6:]}"
    )
    return valid_scores, {
        "objective": "long_view_bce_plus_nested_watch_progress_bce",
        "seed": seed,
        "ordinal_aux_weight": auxiliary_weight,
        "training_split": training_split,
        "selection_split": selection_split,
        "refit_split": f"2022-04-08..{maximum_date_label}",
        "confirmation_split": (
            "2022-04-15..2022-04-21"
            if maximum_train_date <= 20220414 else "2022-04-22..2022-04-28"
        ),
        "selected_epoch": int(best_epoch),
        "core_ordinal": core_stats,
        "full_ordinal": full_stats,
        "epochs": history,
        "refit_epochs": refit_history,
    }


def train_profile_deepfm(
    splits: dict,
    encoded: dict,
    dimension: int,
    output_dir: Path,
    parameters: dict,
    evaluator: Callable,
) -> tuple[np.ndarray, dict]:
    """Train the temporally selected DeepFM with static user-profile embeddings."""
    profile_encoded, profile_dimension, profile_stats = augment_with_user_profiles(
        splits, encoded, dimension
    )
    profile_parameters = {**parameters, "ordinal_aux_weight": 0.0}
    scores, history = train_ordinal_deepfm(
        splits, profile_encoded, profile_dimension, output_dir,
        profile_parameters, evaluator,
    )
    history["objective"] = "profile_augmented_deepfm_weighted_bce"
    history["user_profile"] = profile_stats
    return scores, history
