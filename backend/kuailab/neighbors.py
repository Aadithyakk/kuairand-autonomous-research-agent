from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse


SUPPORTED_VIEWS = ("item", "author", "music", "tag", "video_type")
SUPPORTED_PROFILE_MODES = ("exposure", "positive", "signed")


def _metadata(data_dir: Path) -> dict[str, dict[str, str]]:
    fields = ("author_id", "music_id", "tag", "video_type")
    result: dict[str, dict[str, str]] = {}
    with (data_dir / "video_features_basic_pure.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[str(row["video_id"])] = {
                field: str(row.get(field) or "UNK") for field in fields
            }
    return result


def _ordered_codes(values: Iterable[str]) -> tuple[dict[str, int], np.ndarray]:
    mapping: dict[str, int] = {}
    codes: list[int] = []
    for raw_value in values:
        value = str(raw_value)
        code = mapping.get(value)
        if code is None:
            code = len(mapping)
            mapping[value] = code
        codes.append(code)
    return mapping, np.asarray(codes, dtype=np.int32)


def _view_value(row: tuple, view: str, metadata: dict[str, dict[str, str]]) -> str:
    if view == "item":
        return str(row[2])
    if view == "author":
        return str(row[3])
    field = {"music": "music_id", "tag": "tag", "video_type": "video_type"}[view]
    return metadata.get(str(row[2]), {}).get(field, "UNK")


def _profile_block(
    rows: list[tuple],
    user_codes: np.ndarray,
    labels: np.ndarray,
    user_count: int,
    view: str,
    metadata: dict[str, dict[str, str]],
    mode: str,
    idf_power: float,
) -> sparse.csr_matrix:
    _, entity_codes = _ordered_codes(_view_value(row, view, metadata) for row in rows)
    if mode == "exposure":
        values = np.ones(len(rows), dtype=np.float32)
    elif mode == "positive":
        values = labels.astype(np.float32, copy=True)
    else:
        values = (2.0 * labels - 1.0).astype(np.float32, copy=False)
    matrix = sparse.coo_matrix(
        (values, (user_codes, entity_codes)),
        shape=(user_count, int(entity_codes.max()) + 1),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    if mode in {"exposure", "positive"}:
        matrix.eliminate_zeros()
        matrix.data[:] = 1.0
    if idf_power > 0 and matrix.nnz:
        binary = matrix.copy()
        binary.data[:] = 1.0
        document_frequency = np.asarray(binary.sum(axis=0)).ravel()
        idf = np.power(
            np.log((1.0 + user_count) / (1.0 + document_frequency)) + 1.0,
            idf_power,
        ).astype(np.float32)
        matrix.data *= idf[matrix.indices]
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return (sparse.diags(1.0 / np.maximum(norms, 1e-8)) @ matrix).tocsr()


def _nearest_users(
    profile: sparse.csr_matrix,
    query_users: np.ndarray,
    neighbor_count: int,
    similarity_power: float,
    minimum_similarity: float,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors = np.full((len(query_users), neighbor_count), -1, dtype=np.int32)
    weights = np.zeros((len(query_users), neighbor_count), dtype=np.float32)
    for start in range(0, len(query_users), block_size):
        stop = min(start + block_size, len(query_users))
        similarities = (profile[query_users[start:stop]] @ profile.T).tocsr()
        for local_row, user in enumerate(query_users[start:stop]):
            begin, end = similarities.indptr[local_row:local_row + 2]
            indices = similarities.indices[begin:end]
            values = similarities.data[begin:end]
            keep = (indices != user) & (values >= minimum_similarity) & (values > 0)
            indices = indices[keep]
            values = values[keep]
            if not len(indices):
                continue
            order = np.lexsort((indices, -values))[:neighbor_count]
            selected_indices = indices[order]
            selected_values = np.power(values[order], similarity_power)
            count = len(selected_indices)
            neighbors[start + local_row, :count] = selected_indices
            weights[start + local_row, :count] = selected_values
    return neighbors, weights


def build_multiview_neighbor_scores(
    train_rows: list[tuple],
    valid_rows: list[tuple],
    data_dir: Path,
    parameters: dict[str, Any],
    *,
    video_metadata: dict[str, dict[str, str]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a frozen user-neighbour score without reading validation outcomes."""
    raw_views = parameters.get("neighbor_views", ["item"])
    views = [str(view) for view in raw_views if str(view) in SUPPORTED_VIEWS]
    if not views:
        raise ValueError(f"neighbor_views must contain at least one of {SUPPORTED_VIEWS}")
    if len(set(views)) != len(views) or len(views) != len(raw_views):
        raise ValueError("neighbor_views must contain distinct supported view names")
    raw_weights = parameters.get("neighbor_view_weights", [1.0] * len(views))
    if len(raw_weights) != len(views):
        raise ValueError("neighbor_view_weights must provide one value per neighbor view")
    view_weights = np.asarray(raw_weights, dtype=np.float32)
    if not np.all(np.isfinite(view_weights)) or np.any(view_weights < 0) or not np.any(view_weights > 0):
        raise ValueError("neighbor_view_weights must be finite, nonnegative, and include a positive value")
    view_weights /= float(view_weights.sum())

    profile_mode = str(parameters.get("neighbor_profile_mode", "exposure"))
    if profile_mode not in SUPPORTED_PROFILE_MODES:
        raise ValueError(f"Unsupported neighbor_profile_mode {profile_mode!r}")
    neighbor_count = max(10, min(int(parameters.get("neighbor_count", 60)), 120))
    similarity_power = max(0.5, min(float(parameters.get("neighbor_similarity_power", 2.0)), 4.0))
    idf_power = max(0.0, min(float(parameters.get("neighbor_idf_power", 0.0)), 1.0))
    minimum_similarity = max(0.0, min(float(parameters.get("neighbor_min_similarity", 0.0)), 0.95))
    neighbor_smoothing = max(0.1, min(float(parameters.get("neighbor_smoothing", 8.0)), 100.0))
    item_smoothing = max(0.1, min(float(parameters.get("neighbor_item_smoothing", 20.0)), 200.0))

    metadata = video_metadata if video_metadata is not None else _metadata(data_dir)
    all_rows = [*train_rows, *valid_rows]
    user_mapping, all_user_codes = _ordered_codes(str(row[1]) for row in all_rows)
    item_mapping, all_item_codes = _ordered_codes(str(row[2]) for row in all_rows)
    train_size = len(train_rows)
    train_users = all_user_codes[:train_size]
    valid_users = all_user_codes[train_size:]
    train_items = all_item_codes[:train_size]
    valid_items = all_item_codes[train_size:]
    train_labels = np.asarray([float(row[6]) for row in train_rows], dtype=np.float32)
    user_count = len(user_mapping)
    item_count = len(item_mapping)

    blocks: list[sparse.csr_matrix] = []
    block_nnz: dict[str, int] = {}
    for view, weight in zip(views, view_weights):
        block = _profile_block(
            train_rows, train_users, train_labels, user_count, view,
            metadata, profile_mode, idf_power,
        )
        block = block.multiply(np.sqrt(float(weight))).tocsr()
        blocks.append(block)
        block_nnz[view] = int(block.nnz)
    profile = sparse.hstack(blocks, format="csr")
    profile_norms = np.sqrt(np.asarray(profile.multiply(profile).sum(axis=1)).ravel())
    profile = (sparse.diags(1.0 / np.maximum(profile_norms, 1e-8)) @ profile).tocsr()

    unique_valid_users = np.unique(valid_users)
    neighbors, weights = _nearest_users(
        profile, unique_valid_users, neighbor_count, similarity_power,
        minimum_similarity,
    )
    neighbor_lookup = np.full(user_count, -1, dtype=np.int32)
    neighbor_lookup[unique_valid_users] = np.arange(len(unique_valid_users), dtype=np.int32)

    exposure = sparse.coo_matrix(
        (np.ones(train_size, dtype=np.float32), (train_users, train_items)),
        shape=(user_count, item_count),
    ).tocsr()
    positive = sparse.coo_matrix(
        (train_labels, (train_users, train_items)),
        shape=(user_count, item_count),
    ).tocsr()
    global_prior = float(train_labels.mean())
    item_exposure = np.asarray(exposure.sum(axis=0)).ravel().astype(np.float64)
    item_positive = np.asarray(positive.sum(axis=0)).ravel().astype(np.float64)
    item_prior = (item_positive + item_smoothing * global_prior) / (item_exposure + item_smoothing)

    scores = item_prior[valid_items].astype(np.float32, copy=True)
    support = np.zeros(len(valid_rows), dtype=np.float32)
    order = np.argsort(valid_users, kind="stable")
    ordered_users = valid_users[order]
    boundaries = np.flatnonzero(np.r_[True, ordered_users[1:] != ordered_users[:-1], True])
    for boundary in range(len(boundaries) - 1):
        rows = order[boundaries[boundary]:boundaries[boundary + 1]]
        user = int(valid_users[rows[0]])
        neighbor_row = int(neighbor_lookup[user])
        local_neighbors = neighbors[neighbor_row]
        local_weights = weights[neighbor_row].astype(np.float64)
        keep = local_neighbors >= 0
        local_neighbors = local_neighbors[keep]
        local_weights = local_weights[keep]
        if not len(local_neighbors) or not np.any(local_weights > 0):
            continue
        items = valid_items[rows]
        local_exposure = exposure[local_neighbors][:, items].toarray().astype(np.float64)
        local_positive = positive[local_neighbors][:, items].toarray().astype(np.float64)
        denominator = local_weights @ local_exposure
        numerator = local_weights @ local_positive
        support[rows] = denominator.astype(np.float32)
        scores[rows] = (
            (numerator + neighbor_smoothing * item_prior[items])
            / (denominator + neighbor_smoothing)
        ).astype(np.float32)

    diagnostics = {
        "kind": "train_only_multiview_user_neighbor",
        "views": views,
        "view_weights": [float(value) for value in view_weights],
        "profile_mode": profile_mode,
        "neighbor_count": neighbor_count,
        "similarity_power": similarity_power,
        "idf_power": idf_power,
        "minimum_similarity": minimum_similarity,
        "neighbor_smoothing": neighbor_smoothing,
        "item_smoothing": item_smoothing,
        "training_rows": len(train_rows),
        "prediction_rows": len(valid_rows),
        "users": user_count,
        "profile_nnz": int(profile.nnz),
        "view_nnz": block_nnz,
        "supported_prediction_fraction": float(np.mean(support > 0)),
        "support_median": float(np.median(support)),
        "support_p90": float(np.quantile(support, 0.9)),
        "validation_outcomes_accessed": False,
        "hidden_test_accessed": False,
    }
    return scores, diagnostics
