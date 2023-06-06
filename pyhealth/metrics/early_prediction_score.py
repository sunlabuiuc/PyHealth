import numpy as np
from typing import Dict, Optional


def calculate_confusion_matrix_value_result(outcome_pred, outcome_true):
    outcome_pred = 1 if outcome_pred > 0.5 else 0
    if outcome_pred == 1 and outcome_true == 1:
        return "tp"
    elif outcome_pred == 0 and outcome_true == 0:
        return "tn"
    elif outcome_pred == 1 and outcome_true == 0:
        return "fp"
    elif outcome_pred == 0 and outcome_true == 1:
        return "fn"
    else:
        raise ValueError("Unknown value occurred")

def calculate_es(los_true, threshold, penalty, case="tp"):
    metric = 0.0
    if case == "tp":
        if los_true >= threshold:  # predict correct in early stage
            metric = 1
        else:
            metric = los_true / threshold
    elif case == "fn":
        if los_true >= threshold:  # predict wrong in early stage
            metric = 0
        else:
            metric = los_true / threshold - 1
    elif case == "tn":
        metric = 0.0
    elif case == "fp":
        metric = penalty # penalty term
    return metric


def early_prediction_score(
    y_true_outcome: np.ndarray,
    y_true_los: np.ndarray,
    y_prob: np.ndarray,
    late_threshold: Optional[float] = None,
    fp_penalty: Optional[float] = -0.1
) -> Dict[str, float]:
    """Computes early prediction score for binary classification.
    
    Paper: Junyi Gao, et al. A Comprehensive Benchmark for COVID-19 Predictive Modeling
    Using Electronic Health Records in Intensive Care: Choosing the Best Model for
    COVID-19 Prognosis. arXiv preprint arXiv:2209.07805, 2023.

    Args:
        y_true_outcome: True target outcome of shape (n_samples,).
        y_true_los: Time to true target outcome of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples,).
        late_threshold: Threshold gamma for late prediction penalties. Default is 0.5 *
        mean(y_true_los).
        fp_penalty: Penalty term for false positive predictions. Default is -0.1.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import early_prediction_score
        >>> y_true_outcome = np.array([0, 0, 1, 1])
        >>> y_true_los = np.array([5, 3, 8, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.7, 0.8])
        >>> early_prediction_score(y_true_outcome, y_true_los, y_prob)
        {'score': 0.5952380952380952, 'late_threshold': 2.125, 'fp_penalty': 0.1}
    """
    metric = []
    metric_optimal = []
    num_records = len(y_prob)
    
    if late_threshold is None:
        late_threshold = 0.5 * np.mean(y_true_los)
        
    for i in range(num_records):
        cur_outcome_pred = y_prob[i]
        cur_outcome_true = y_true_outcome[i]
        cur_los_true = y_true_los[i]
        prediction_result = calculate_confusion_matrix_value_result(cur_outcome_pred, cur_outcome_true)
        prediction_result_optimal = calculate_confusion_matrix_value_result(cur_outcome_true, cur_outcome_true)
        metric.append(
            calculate_es(
                cur_los_true,
                late_threshold,
                penalty=fp_penalty,
                case=prediction_result,
            )
        )
        metric_optimal.append(
            calculate_es(
                cur_los_true,
                late_threshold,
                penalty=fp_penalty,
                case=prediction_result_optimal,
            )
        )
    metric = np.array(metric)
    metric_optimal = np.array(metric_optimal)
    result = 0.0
    if metric_optimal.sum() > 0.0:
        result = metric.sum() / metric_optimal.sum()
    result = max(result, -1.0)
    if isinstance(result, np.float64):
        result = result.item()
    return {"score": result, 'late_threshold': late_threshold, 'fp_penalty': fp_penalty}

if __name__ == "__main__":
    y_true_outcome = np.array([0, 1, 1, 1])
    y_true_los = np.array([5, 3, 8, 1])
    y_prob = np.array([0.1, 0.4, 0.7, 0.8])
    print(early_prediction_score(y_true_outcome, y_true_los, y_prob))