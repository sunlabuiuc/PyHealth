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
    If no metrics are specified, kd_div, mse, mae are computed by default.

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
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
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
            x[x < 1e-6] = 1e-6
            x_rec[x_rec < 1e-6] = 1e-6
            x = x / np.sum(x)
            x_rec = x_rec / np.sum(x_rec)
            kl_divergence = np.sum(x_rec * np.log(x_rec / x))
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
