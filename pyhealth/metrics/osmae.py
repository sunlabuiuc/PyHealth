
import numpy as np
from typing import Dict, Optional


def calculate_outcome_prediction_result(outcome_pred, outcome_true):
    outcome_pred = 1 if outcome_pred > 0.5 else 0
    return "true" if outcome_pred == outcome_true else "false"


def calculate_epsilon(los_true, threshold, large_los):
    """
    epsilon is the decay term
    """
    if los_true <= threshold:
        return 1
    else:
        return max(0, (los_true - large_los) / (threshold - large_los))


def calculate_osmae(los_pred, los_true, large_los, threshold, case="true"):
    if case == "true":
        epsilon = calculate_epsilon(los_true, threshold, large_los)
        return epsilon * np.abs(los_pred - los_true)
    elif case == "false":
        epsilon = calculate_epsilon(los_true, threshold, large_los)
        return epsilon * (max(0, large_los - los_pred) + max(0, large_los - los_true))
    else:
        raise ValueError("case must be 'true' or 'false'")


def osmae_score(
    y_true_outcome: np.ndarray,
    y_true_los: np.ndarray,
    y_prob_outcome: np.ndarray,
    y_pred_los: np.ndarray,
    large_los: Optional[float] = None,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """Computes outcome-specific mean absolute error for length-of-stay prediction.
    
    Paper: Junyi Gao, et al. A Comprehensive Benchmark for COVID-19 Predictive Modeling
    Using Electronic Health Records in Intensive Care: Choosing the Best Model for
    COVID-19 Prognosis. arXiv preprint arXiv:2209.07805, 2023.

    Args:
        y_true_outcome: True target outcome of shape (n_samples,).
        y_true_los: Time to true target outcome of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        y_prob_outcome: Predicted outcome probabilities of shape (n_samples,).
        y_pred_los: Predicted length-of-stay of shape (n_samples,).
        large_los: Largest length-of-stay E, default is 95% percentile of maximum of total
        length-of-stay.
        threshold: Threshold gamma for late prediction penalties. Default is 0.5 *
        mean(y_true_los).
        
    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import osmae_score
        >>> y_true_outcome = np.array([0, 0, 1, 1])
        >>> y_true_los = np.array([5, 3, 8, 1])
        >>> y_prob_outcome = np.array([0.1, 0.4, 0.7, 0.2])
        >>> y_pred_los = np.array([10, 5, 7, 3])
        >>> osmae_score(y_true_outcome, y_true_los, y_prob_outcome, y_pred_los)
        {'osmae': 4.0638297872340425, 'large_los': 8, 'threshold': 2.125}
    """
    if large_los is None:
        large_los = np.sort(y_true_los)[int(0.95 * len(y_true_los))]
        
    if threshold is None:
        threshold = 0.5 * np.mean(y_true_los)
    
    metric = []
    num_records = len(y_prob_outcome)
    for i in range(num_records):
        cur_outcome_pred = y_prob_outcome[i]
        cur_los_pred = y_pred_los[i]
        cur_outcome_true = y_true_outcome[i]
        cur_los_true = y_true_los[i]
        prediction_result = calculate_outcome_prediction_result(
            cur_outcome_pred, cur_outcome_true
        )
        metric.append(
            calculate_osmae(
                cur_los_pred,
                cur_los_true,
                large_los,
                threshold,
                case=prediction_result,
            )
        )
    result = np.array(metric)
    return {"osmae": result.mean(axis=0).item(), "large_los": large_los, "threshold": threshold}

if __name__ == "__main__":
    y_true_outcome = np.array([0, 0, 1, 1])
    y_true_los = np.array([5, 3, 8, 1])
    y_prob_outcome = np.array([0.1, 0.4, 0.7, 0.2])
    y_pred_los = np.array([10, 5, 7, 3])
    print(osmae_score(y_true_outcome, y_true_los, y_prob_outcome, y_pred_los))