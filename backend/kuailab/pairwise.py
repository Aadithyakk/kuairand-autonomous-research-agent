from __future__ import annotations

from collections import defaultdict

import numpy as np


def sample_pair_indices(
    users: list[str], labels: np.ndarray, random: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one logged negative from the same user for every usable positive."""
    negatives: dict[str, list[int]] = defaultdict(list)
    positives: list[int] = []
    for index, (user, label) in enumerate(zip(users, labels)):
        if label > 0.5:
            positives.append(index)
        else:
            negatives[user].append(index)
    positive_indices = np.asarray(
        [index for index in positives if negatives.get(users[index])], dtype=np.int64
    )
    negative_indices = np.fromiter(
        (
            negatives[users[index]][random.integers(len(negatives[users[index]]))]
            for index in positive_indices
        ),
        dtype=np.int64,
        count=len(positive_indices),
    )
    order = random.permutation(len(positive_indices))
    return positive_indices[order], negative_indices[order]
