from typing import Dict, List, Optional

import numpy as np
import sklearn.metrics as sklearn_metrics

def calculate_ccc(y_true, y_pred):
    """
    This function calculates the concordance correlation coefficient (CCC) between two vectors
    :param y_true: real data
    :param y_pred: estimated data
    :return: CCC
    :rtype: float
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator

def regression_metrics_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Computes metrics for regression.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - mse: mean squared error
        - mae: mean absolute error
        - mape: mean absolute percentage error
        - rmse: root mean squared error
        - ccc: concordance correlation coefficient
        - r2: R^2 score
    If no metrics are specified, mse, mae and r2 are computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_pred: Predicted values of shape (n_samples,).
        metrics: List of metrics to compute. Default is ["mse", "mae", "r2"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_true = np.array([1, 3, 2, 4])
        >>> y_pred = np.array([1.1, 2.4, 1.35, 2.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["mse", "mae", "r2"])
        {'mse': 0.1475, 'mae': 0.275, 'r2': 0.6923076923076923}
    """
    if metrics is None:
        metrics = ["mse", "mae", "r2"]

    output = {}
    for metric in metrics:
        if metric == "mse":
            mse = sklearn_metrics.mean_squared_error(y_true, y_pred)
            output["mse"] = mse
        elif metric == "mae":
            mae = sklearn_metrics.mean_absolute_error(y_true, y_pred)
            output["mae"] = mae
        elif metric == "mape":
            mape = sklearn_metrics.mean_absolute_percentage_error(y_true, y_pred)
            output["mape"] = mape
        elif metric == "rmse":
            rmse = np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred))
            output["rmse"] = rmse
        elif metric == "ccc":
            ccc = calculate_ccc(y_true, y_pred)
            output["ccc"] = ccc
        elif metric == "r2":
            r2 = sklearn_metrics.r2_score(y_true, y_pred)
            output["r2"] = r2
        else:
            raise ValueError(f"Unknown metric for regression: {metric}")
    return output


if __name__ == "__main__":
    all_metrics = [
        "mse",
        "mae",
        "r2",
        "ccc",
        "rmse",
        "mape"
    ]
    y_true = np.random.randint(0,10, size=100000)
    y_pred = np.random.random(size=100000)*10
    print(regression_metrics_fn(y_true, y_pred, metrics=all_metrics))
