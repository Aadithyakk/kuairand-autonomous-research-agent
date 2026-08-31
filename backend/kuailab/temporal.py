from __future__ import annotations

from datetime import datetime

import numpy as np


def encode_clock_context(splits: dict) -> tuple[dict, int]:
    """Encode base categorical fields plus label-free request-time context."""
    train = splits["train"]
    duration_edges = np.quantile([row[5] for row in train], np.linspace(0, 1, 11)[1:-1])
    weekday_by_date = {
        row[0]: datetime.strptime(str(row[0]), "%Y%m%d").weekday()
        for rows in splits.values()
        for row in rows
    }

    def raw(row: tuple) -> list[str]:
        hour = int(row[7])
        return [
            row[1],
            row[2],
            row[3],
            row[4],
            str(int(np.searchsorted(duration_edges, row[5]))),
            str(hour),
            str(hour // 4),
            str(weekday_by_date[row[0]]),
        ]

    vocabs: list[dict[str, int]] = [dict() for _ in range(8)]
    for row in train:
        for field, value in enumerate(raw(row)):
            if value not in vocabs[field]:
                vocabs[field][value] = len(vocabs[field])
    unknown = [len(vocab) for vocab in vocabs]
    dimensions = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
    encoded = {}
    for name, rows in splits.items():
        features = np.empty((len(rows), len(vocabs)), dtype=np.int32)
        labels = np.empty(len(rows), dtype=np.float32)
        users = []
        for index, row in enumerate(rows):
            for field, value in enumerate(raw(row)):
                features[index, field] = vocabs[field].get(value, unknown[field]) + offsets[field]
            labels[index] = row[6]
            users.append(row[1])
        encoded[name] = (features, labels, users)
    return encoded, int(sum(dimensions))
