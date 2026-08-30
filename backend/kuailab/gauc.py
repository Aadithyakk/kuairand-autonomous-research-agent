from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from backend.kuailab.pairwise import sample_pair_indices


def train_gauc_deepfm(
    splits: dict,
    encoded: dict,
    dimension: int,
    output_dir: Path,
    parameters: dict,
    evaluator: Callable,
) -> tuple[np.ndarray, dict]:
    """Train a pointwise-anchored DeepFM with full-user AUC comparisons.

    The pair loss samples a logged negative from the same user for each usable
    positive. Epoch selection uses GAUC on a strictly later train-only window;
    the selected epoch count is then refit on every row available to the run.
    """
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("gauc_deepfm requires PyTorch 2.1 or newer") from error

    class GaucDeepFM(nn.Module):
        def __init__(self, fields: int, k: int, hidden: int, dropout: float):
            super().__init__()
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, k)
            self.deep = nn.Sequential(
                nn.Linear(fields * k, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, max(hidden // 2, 8)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(hidden // 2, 8), 1),
            )
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (
                summed.square() - embedding.square().sum(dim=1)
            ).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            return linear + fm + self.deep(embedding.flatten(start_dim=1)).squeeze(-1)

    train_rows = splits["train"]
    train_x, train_y, train_users = encoded["train"]
    valid_x, _, _ = encoded["valid"]
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
            index
            for index, row in enumerate(train_rows)
            if dev_start <= int(row[0]) <= maximum_train_date
        ],
        dtype=np.int64,
    )
    core_x, core_y = train_x[core_indices], train_y[core_indices]
    core_users = [train_users[index] for index in core_indices]
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
    pair_weight = max(0.0, min(float(parameters.get("gauc_pair_weight", 0.05)), 0.5))
    torch.set_num_threads(max(1, min(int(parameters.get("deep_threads", 6)), 16)))

    def tensors(x, y):
        return (
            torch.from_numpy(x.astype(np.int64, copy=False)),
            torch.from_numpy(y.astype(np.float32, copy=False)),
        )

    core_x_tensor, core_y_tensor = tensors(core_x, core_y)
    dev_x_tensor, _ = tensors(dev_x, dev_y)
    full_x_tensor, full_y_tensor = tensors(train_x, train_y)
    valid_x_tensor = torch.from_numpy(valid_x.astype(np.int64, copy=False))

    def predict(model, x_tensor):
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), batch_size * 4):
                predictions.append(
                    model(x_tensor[start:start + batch_size * 4]).numpy()
                )
        return np.concatenate(predictions).astype(np.float32)

    def train_epoch(
        model, optimizer, x_tensor, y_tensor, users, row_random, pair_random,
    ):
        model.train()
        order = row_random.permutation(len(y_tensor))
        positive_indices = negative_indices = np.empty(0, dtype=np.int64)
        if pair_weight > 0:
            positive_indices, negative_indices = sample_pair_indices(
                users, y_tensor.numpy(), pair_random
            )
            if not len(positive_indices):
                raise RuntimeError("GAUC pair loss found no within-user pairs")
        losses, pointwise_losses, pair_losses = [], [], []
        pair_batch_size = max(1024, batch_size // 2)
        for batch_number, start in enumerate(range(0, len(order), batch_size)):
            indices = torch.from_numpy(order[start:start + batch_size])
            x, y = x_tensor[indices], y_tensor[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            weights = torch.where(y > 0.5, positive_weight, 1.0)
            pointwise_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits, y, reduction="none"
                ) * weights
            ).sum() / weights.sum()
            pair_loss = logits.new_zeros(())
            if pair_weight > 0:
                pair_offsets = (
                    np.arange(pair_batch_size, dtype=np.int64)
                    + batch_number * pair_batch_size
                ) % len(positive_indices)
                positive = torch.from_numpy(positive_indices[pair_offsets])
                negative = torch.from_numpy(negative_indices[pair_offsets])
                pair_loss = nn.functional.softplus(
                    -(model(x_tensor[positive]) - model(x_tensor[negative]))
                ).mean()
            loss = pointwise_loss + pair_weight * pair_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            pointwise_losses.append(float(pointwise_loss.detach()))
            pair_losses.append(float(pair_loss.detach()))
        return {
            "loss": float(np.mean(losses)),
            "pointwise_loss": float(np.mean(pointwise_losses)),
            "pair_loss": float(np.mean(pair_losses)),
            "pairs": int(len(positive_indices)),
        }

    torch.manual_seed(seed)
    model = GaucDeepFM(core_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    row_random = np.random.default_rng(seed)
    pair_random = np.random.default_rng(seed + 4_099)
    best_gauc, best_primary, best_epoch, bad = -1.0, -1.0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        losses = train_epoch(
            model, optimizer, core_x_tensor, core_y_tensor, core_users,
            row_random, pair_random,
        )
        scores = predict(model, dev_x_tensor)
        metrics = evaluator(dev_users, dev_y, scores)
        history.append({
            "epoch": epoch,
            **losses,
            "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]),
        })
        gauc, primary = float(metrics["GAUC"]), float(metrics["primary"])
        if gauc > best_gauc + 1e-5 or (
            abs(gauc - best_gauc) <= 1e-5 and primary > best_primary
        ):
            best_gauc, best_primary, best_epoch, bad = gauc, primary, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if not best_epoch:
        raise RuntimeError("GAUC DeepFM temporal selection produced no checkpoint")

    torch.manual_seed(seed)
    model = GaucDeepFM(train_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    refit_row_random = np.random.default_rng(seed + 9_731)
    refit_pair_random = np.random.default_rng(seed + 13_831)
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        losses = train_epoch(
            model, optimizer, full_x_tensor, full_y_tensor, train_users,
            refit_row_random, refit_pair_random,
        )
        refit_history.append({"epoch": epoch, **losses})

    valid_scores = predict(model, valid_x_tensor)
    torch.save(model.state_dict(), output_dir / f"gauc-deep-checkpoint-seed-{seed}.pt")
    maximum_date_text = str(maximum_train_date)
    maximum_date_label = (
        f"{maximum_date_text[:4]}-{maximum_date_text[4:6]}-{maximum_date_text[6:]}"
    )
    return valid_scores, {
        "objective": "long_view_bce_plus_full_user_softplus_pair_loss",
        "selection_metric": "GAUC",
        "seed": seed,
        "gauc_pair_weight": pair_weight,
        "training_split": training_split,
        "selection_split": selection_split,
        "refit_split": f"2022-04-08..{maximum_date_label}",
        "confirmation_split": (
            "2022-04-15..2022-04-21"
            if maximum_train_date <= 20220414 else "2022-04-22..2022-04-28"
        ),
        "selected_epoch": int(best_epoch),
        "epochs": history,
        "refit_epochs": refit_history,
    }
