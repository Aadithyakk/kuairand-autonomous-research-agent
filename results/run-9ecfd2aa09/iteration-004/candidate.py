import numpy as np

PARAMETERS = {
    "k": 16,
    "lr": 0.001,
    "epochs": 40,
    "batch_size": 8192,
    "patience": 4,
    "seed": 0,
    "ensemble_seeds": [0, 1, 2],
    "positive_weight": 2.75,
}

def fm_logits(feature_ids, bias, linear, factors):
    """FM logits for integer feature IDs, one active feature per field."""
    ids = np.asarray(feature_ids, dtype=np.int64)
    selected = factors[ids]
    summed = selected.sum(axis=1)
    interaction = 0.5 * (
        np.square(summed).sum(axis=1)
        - np.square(selected).sum(axis=(1, 2))
    )
    return float(bias) + linear[ids].sum(axis=1) + interaction

def positive_weighted_bce_with_logit_grad(logits, labels):
    """Mean weighted logistic loss and its exact logit gradient."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if z.shape != y.shape or z.ndim != 1:
        raise ValueError("logits and labels must be equal-length vectors")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("labels must be binary")

    weight = 1.0 + (PARAMETERS["positive_weight"] - 1.0) * y
    per_example = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    loss = np.mean(weight * per_example)

    probability = np.empty_like(z)
    nonnegative = z >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    exp_z = np.exp(z[~nonnegative])
    probability[~nonnegative] = exp_z / (1.0 + exp_z)
    logit_grad = weight * (probability - y) / y.size
    return float(loss), logit_grad

def fm_batch_loss_and_grads(feature_ids, labels, bias, linear, factors):
    """Concrete NumPy FM objective used by the typed positive-weight run."""
    ids = np.asarray(feature_ids, dtype=np.int64)
    selected = factors[ids]
    summed = selected.sum(axis=1)
    logits = fm_logits(ids, bias, linear, factors)
    loss, dz = positive_weighted_bce_with_logit_grad(logits, labels)

    grad_linear = np.zeros_like(linear)
    np.add.at(grad_linear, ids.reshape(-1), np.repeat(dz, ids.shape[1]))

    occurrence_grad = dz[:, None, None] * (summed[:, None, :] - selected)
    grad_factors = np.zeros_like(factors)
    np.add.at(
        grad_factors,
        ids.reshape(-1),
        occurrence_grad.reshape(-1, factors.shape[1]),
    )
    grad_bias = float(dz.sum())
    return loss, grad_bias, grad_linear, grad_factors
