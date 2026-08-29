import numpy as np

CONFIG = {
    "experiment_type": "fm_positive_weight",
    "k": 16,
    "lr": 0.001,
    "epochs": 40,
    "batch_size": 8192,
    "patience": 4,
    "seed": 0,
    "ensemble_seeds": [0, 1, 2],  # Typed but unused for a single-FM experiment.
    "positive_weight": 2.5,
}

def weighted_logistic_loss_and_grad(logits, labels, positive_weight=2.5):
    """NumPy implementation of the selected FM logit loss and its gradient."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if z.shape != y.shape:
        raise ValueError("logits and labels must have identical shapes")
    if z.size == 0:
        raise ValueError("batch must not be empty")
    if not (1.0 <= positive_weight <= 10.0):
        raise ValueError("positive_weight must be in [1, 10]")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("labels must be binary")

    example_weight = np.where(y == 1.0, positive_weight, 1.0)
    per_example_bce = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    loss = np.mean(example_weight * per_example_bce)

    probability = np.empty_like(z)
    nonnegative = z >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-z[nonnegative]))
    exp_z = np.exp(z[~nonnegative])
    probability[~nonnegative] = exp_z / (1.0 + exp_z)
    grad_logits = example_weight * (probability - y) / z.size
    return float(loss), grad_logits

# The trusted executor uses this loss for the official FM while leaving its
# architecture, data split, optimizer settings, early stopping, and seed fixed.
assert CONFIG["positive_weight"] == 2.5
