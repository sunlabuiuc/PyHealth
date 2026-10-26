import numpy as np
import sklearn.metrics as sklearn_metrics


def regression_metrics_fn(
    x: np.ndarray,
    x_rec: np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Computes metrics for regression.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - kl_divergence: KL divergence KL(x_rec || x), computed after clipping
          values below 1e-6 and normalizing each array to sum to 1
        - mse: mean squared error
        - mae: mean absolute error
    If no metrics are specified, kl_divergence, mse and mae are computed by
    default.

    The mse and mae metrics call sklearn.metrics functions. For more
    information on them, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        x: True target data sample.
        x_rec: Reconstructed data sample. Must have the same shape as x.
        metrics: List of metrics to compute. Default is
            ["kl_divergence", "mse", "mae"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values as Python floats.

    Raises:
        ValueError: If x and x_rec do not have the same shape, or if an
            unknown metric name is given.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics import regression_metrics_fn
        >>> x = np.array([0.1, 0.2, 0.3, 0.4])
        >>> x_rec = np.array([0.1, 0.25, 0.25, 0.4])
        >>> result = regression_metrics_fn(x, x_rec, metrics=["mse", "mae"])
        >>> {k: round(v, 5) for k, v in result.items()}
        {'mse': 0.00125, 'mae': 0.025}
    """
    if metrics is None:
        metrics = ["kl_divergence", "mse", "mae"]

    # Check shapes before flattening: flattening first would let arrays of
    # different shapes but equal size, e.g. (2, 3) and (3, 2), pass silently.
    if x.shape != x_rec.shape:
        raise ValueError(
            f"x and x_rec must have the same shape, got {x.shape} and {x_rec.shape}."
        )

    x = x.flatten()
    x_rec = x_rec.flatten()

    output = {}
    for metric in metrics:
        if metric == "kl_divergence":
            # Work on copies to avoid mutating x/x_rec for subsequent metrics
            x_kl = x.copy()
            x_rec_kl = x_rec.copy()
            x_kl[x_kl < 1e-6] = 1e-6
            x_rec_kl[x_rec_kl < 1e-6] = 1e-6
            x_kl = x_kl / np.sum(x_kl)
            x_rec_kl = x_rec_kl / np.sum(x_rec_kl)
            kl_divergence = np.sum(x_rec_kl * np.log(x_rec_kl / x_kl))
            output["kl_divergence"] = float(kl_divergence)
        elif metric == "mse":
            mse = sklearn_metrics.mean_squared_error(x, x_rec)
            output["mse"] = float(mse)
        elif metric == "mae":
            mae = sklearn_metrics.mean_absolute_error(x, x_rec)
            output["mae"] = float(mae)
        else:
            raise ValueError(f"Unknown metric for regression task: {metric}")
    return output


if __name__ == "__main__":
    x = np.random.random(size=10000)
    x_rec = np.random.random(size=10000)
    print(regression_metrics_fn(x, x_rec))
