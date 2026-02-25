from typing import Dict, List, Optional

import numpy as np
import sklearn.metrics as sklearn_metrics


def regression_metrics_fn(
    x: np.ndarray,
    x_rec: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Computes metrics for regression.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - kl_divergence: KL divergence
        - mse: mean squared error
        - mae: mean absolute error
    If no metrics are specified, kl_divergence, mse, and mae are computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        x: True target data sample.
        x_rec: reconstructed data sample.
        metrics: List of metrics to compute. Default is ["kl_divergence", "mse", "mae"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import regression_metrics_fn
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> x_rec = np.array([1.2, 1.8, 2.9])
        >>> regression_metrics_fn(x, x_rec, metrics=["mse"])
        {'mse': 0.03}
    """
    if metrics is None:
        metrics = ["kl_divergence", "mse", "mae"]

    x = x.flatten()
    x_rec = x_rec.flatten()
    
    if x.shape != x_rec.shape:
        raise ValueError("x and x_rec must have the same shape.")

    output = {}
    for metric in metrics:
        if metric == "kl_divergence":
            x_safe = np.maximum(x, 1e-6)
            x_rec_safe = np.maximum(x_rec, 1e-6)
            x_dist = x_safe / np.sum(x_safe)
            x_rec_dist = x_rec_safe / np.sum(x_rec_safe)
            kl_divergence = np.sum(x_rec_dist * np.log(x_rec_dist / x_dist))
            output["kl_divergence"] = kl_divergence
        elif metric == "mse":
            mse = sklearn_metrics.mean_squared_error(x, x_rec)
            output["mse"] = mse
        elif metric == "mae":
            mae = sklearn_metrics.mean_absolute_error(x, x_rec)
            output["mae"] = mae
        else:
            raise ValueError(f"Unknown metric for regression task: {metric}")
    return output


if __name__ == "__main__":
    x = np.random.random(size=10000)
    x_rec = np.random.random(size=10000)
    print(regression_metrics_fn(x, x_rec))
