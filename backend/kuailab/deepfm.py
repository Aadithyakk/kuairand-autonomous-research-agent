from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


def train_deepfm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_users: list[str],
    dimension: int,
    output_dir: Path,
    parameters: dict,
    evaluator: Callable,
) -> tuple[np.ndarray, dict]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("fm_deep_blend requires PyTorch 2.1 or newer") from error

    class DeepFM(nn.Module):
        def __init__(self, fields: int, k: int, hidden: int, dropout: float):
            super().__init__()
            self.linear = nn.Embedding(dimension, 1)
            self.embedding = nn.Embedding(dimension, k)
            self.deep = nn.Sequential(
                nn.Linear(fields * k, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            nn.init.normal_(self.embedding.weight, std=0.01)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            embedding = self.embedding(x)
            summed = embedding.sum(dim=1)
            fm = 0.5 * (summed.square() - embedding.square().sum(dim=1)).sum(dim=1)
            linear = self.linear(x).sum(dim=1).squeeze(-1)
            deep = self.deep(embedding.flatten(start_dim=1)).squeeze(-1)
            return linear + fm + deep

    seed = int(parameters.get("deep_seed", parameters.get("seed", 0)))
    k = max(4, min(int(parameters.get("k", 16)), 32))
    hidden = max(16, min(int(parameters.get("deep_hidden", 64)), 256))
    dropout = max(0.0, min(float(parameters.get("deep_dropout", 0.05)), 0.5))
    learning_rate = max(0.00005, min(float(parameters.get("deep_lr", 0.001)), 0.01))
    epochs = max(3, min(int(parameters.get("deep_epochs", 15)), 40))
    patience = max(2, min(int(parameters.get("deep_patience", 4)), 8))
    batch_size = max(2048, min(int(parameters.get("batch_size", 8192)), 32768))
    positive_weight = max(1.0, min(float(parameters.get("positive_weight", 1.0)), 10.0))
    torch.set_num_threads(max(1, min(int(parameters.get("deep_threads", 6)), 16)))
    torch.manual_seed(seed)
    train_x_tensor = torch.from_numpy(train_x.astype(np.int64, copy=False))
    train_y_tensor = torch.from_numpy(train_y)
    valid_x_tensor = torch.from_numpy(valid_x.astype(np.int64, copy=False))
    model = DeepFM(train_x.shape[1], k, hidden, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    random = np.random.default_rng(seed)
    best_score, best_state, bad = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = random.permutation(len(train_y))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size])
            x = train_x_tensor[indices]
            y = train_y_tensor[indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            weights = torch.where(y > 0.5, positive_weight, 1.0)
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * weights
            ).sum() / weights.sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(valid_x_tensor), batch_size * 4):
                predictions.append(model(valid_x_tensor[start:start + batch_size * 4]).numpy())
        scores = np.concatenate(predictions).astype(np.float32)
        metrics = evaluator(valid_users, valid_y, scores)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "GAUC": float(metrics["GAUC"]),
            "nDCG@5": float(metrics["nDCG@5"]),
            "primary": float(metrics["primary"]),
            "users": int(metrics["users"]),
            "rows": int(metrics["rows"]),
        })
        if metrics["primary"] > best_score + 1e-5:
            best_score, bad = float(metrics["primary"]), 0
            best_state = (
                {name: value.detach().clone() for name, value in model.state_dict().items()},
                scores.copy(),
            )
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("DeepFM training produced no checkpoint")
    model.load_state_dict(best_state[0])
    torch.save(model.state_dict(), output_dir / f"deep-checkpoint-seed-{seed}.pt")
    return best_state[1], {"objective": "deepfm_weighted_bce", "seed": seed, "epochs": history}
