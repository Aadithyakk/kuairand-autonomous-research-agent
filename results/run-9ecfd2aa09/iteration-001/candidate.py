import numpy as np

CONFIG = {
    "k": 16,
    "lr": 0.001,
    "epochs": 40,
    "batch_size": 8192,
    "patience": 4,
    "seed": 0,
    "ensemble_seeds": [0, 1, 2],
    "positive_weight": 2.0,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _fm_logits(feature_ids, bias, linear, factors):
    vb = factors[feature_ids]
    summed = vb.sum(axis=1)
    interaction = 0.5 * ((summed * summed) - (vb * vb).sum(axis=1)).sum(axis=1)
    return bias + linear[feature_ids].sum(axis=1) + interaction


def _average_ranks(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _validation_metrics(labels, logits, user_ids):
    order = np.argsort(user_ids, kind="mergesort")
    users = user_ids[order]
    labels = labels[order]
    logits = logits[order]
    cuts = np.flatnonzero(users[1:] != users[:-1]) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, users.size]

    auc_sum = 0.0
    auc_weight = 0
    ndcg_sum = 0.0
    ndcg_users = 0
    discounts = 1.0 / np.log2(np.arange(2, 7, dtype=np.float64))

    for start, stop in zip(starts, stops):
        y = labels[start:stop]
        s = logits[start:stop]
        positives = int(y.sum())
        negatives = y.size - positives

        if positives > 0 and negatives > 0:
            ranks = _average_ranks(s)
            auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
            auc_sum += y.size * auc
            auc_weight += y.size

        top = min(5, y.size)
        ranked = np.argsort(-s, kind="mergesort")[:top]
        dcg = np.sum(y[ranked] * discounts[:top])
        ideal_top = min(5, positives)
        ideal = discounts[:ideal_top].sum()
        ndcg_sum += 0.0 if ideal == 0.0 else dcg / ideal
        ndcg_users += 1

    gauc = np.nan if auc_weight == 0 else auc_sum / auc_weight
    ndcg5 = np.nan if ndcg_users == 0 else ndcg_sum / ndcg_users
    return gauc, ndcg5, 0.5 * (gauc + ndcg5)


def train_fm_positive_weight(train_feature_ids, train_labels,
                             val_feature_ids, val_labels, val_user_ids,
                             n_features):
    train_feature_ids = np.asarray(train_feature_ids, dtype=np.int64)
    val_feature_ids = np.asarray(val_feature_ids, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.float64)
    val_labels = np.asarray(val_labels, dtype=np.float64)
    val_user_ids = np.asarray(val_user_ids)

    if train_feature_ids.ndim != 2 or val_feature_ids.ndim != 2:
        raise ValueError("feature IDs must have shape [examples, active_fields]")
    if np.any(train_feature_ids < 0) or np.any(train_feature_ids >= n_features):
        raise ValueError("training feature ID outside n_features")
    if np.any(val_feature_ids < 0) or np.any(val_feature_ids >= n_features):
        raise ValueError("validation feature ID outside n_features")

    rng = np.random.default_rng(CONFIG["seed"])
    linear = np.zeros(n_features, dtype=np.float64)
    factors = rng.normal(0.0, 0.01, size=(n_features, CONFIG["k"]))
    bias = 0.0
    best_primary = -np.inf
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        permutation = rng.permutation(train_labels.size)
        for left in range(0, train_labels.size, CONFIG["batch_size"]):
            batch_index = permutation[left:left + CONFIG["batch_size"]]
            ids = train_feature_ids[batch_index]
            y = train_labels[batch_index]
            logits = _fm_logits(ids, bias, linear, factors)
            probabilities = _sigmoid(logits)

            # Isolated typed change: positive examples receive weight 2.0.
            example_weight = np.where(y == 1.0, CONFIG["positive_weight"], 1.0)
            dlogit = example_weight * (probabilities - y) / y.size

            batch_factors = factors[ids]
            factor_sum = batch_factors.sum(axis=1)
            flat_ids = ids.reshape(-1)
            unique_ids, inverse = np.unique(flat_ids, return_inverse=True)

            linear_occurrence_grad = np.repeat(dlogit, ids.shape[1])
            linear_grad = np.zeros(unique_ids.size, dtype=np.float64)
            np.add.at(linear_grad, inverse, linear_occurrence_grad)

            factor_occurrence_grad = (
                dlogit[:, None, None] *
                (factor_sum[:, None, :] - batch_factors)
            ).reshape(-1, CONFIG["k"])
            factor_grad = np.zeros((unique_ids.size, CONFIG["k"]), dtype=np.float64)
            np.add.at(factor_grad, inverse, factor_occurrence_grad)

            bias -= CONFIG["lr"] * dlogit.sum()
            linear[unique_ids] -= CONFIG["lr"] * linear_grad
            factors[unique_ids] -= CONFIG["lr"] * factor_grad

        val_logits = _fm_logits(val_feature_ids, bias, linear, factors)
        gauc, ndcg5, primary = _validation_metrics(val_labels, val_logits, val_user_ids)
        if not np.all(np.isfinite([gauc, ndcg5, primary, bias])):
            raise FloatingPointError("non-finite training state or validation metric")
        history.append({"epoch": epoch + 1, "gauc": gauc, "ndcg5": ndcg5, "primary": primary})

        if primary > best_primary:
            best_primary = primary
            best_state = (bias, linear.copy(), factors.copy())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= CONFIG["patience"]:
                break

    bias, linear, factors = best_state
    return {
        "validation_logits": _fm_logits(val_feature_ids, bias, linear, factors),
        "history": history,
        "best_primary": best_primary,
    }
