import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import List, Union

def radiographic_agreement(y_true: List[Union[int, List[int]]], y_pred: List[Union[int, List[int]]]) -> Dict[str, float]:
    """Calculates inter-rater agreement metrics for radiographic findings.

    This function computes Cohen's Kappa and percentage agreement between true and predicted
    radiographic labels (e.g., presence of pneumonia, edema). It supports both single-label
    and multi-label cases.

    Args:
        y_true (List[Union[int, List[int]]]): Ground truth labels (0 or 1 for single-label,
                                              list of 0/1 for multi-label per finding).
        y_pred (List[Union[int, List[int]]]): Predicted labels matching y_true format.

    Returns:
        Dict[str, float]: Dictionary containing:
            - "kappa": Cohen's Kappa score (range [-1, 1], 1 = perfect agreement).
            - "percent_agreement": Percentage of matching labels (range [0, 100]).

    Raises:
        ValueError: If y_true and y_pred lengths or formats mismatch.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length of y_true and y_pred must match.")
    
    # Flatten multi-label lists if present
    y_true_flat = [item if isinstance(item, int) else int(np.any(np.array(item))) for item in y_true]
    y_pred_flat = [item if isinstance(item, int) else int(np.any(np.array(item))) for item in y_pred]
    
    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(y_true_flat, y_pred_flat)
    
    # Compute percentage agreement
    agreement = sum(1 for t, p in zip(y_true_flat, y_pred_flat) if t == p) / len(y_true_flat) * 100
    
    return {
        "kappa": kappa,
        "percent_agreement": agreement
    }

if __name__ == "__main__":
    # Test with single-label data
    y_true_single = [1, 0, 1, 0]
    y_pred_single = [1, 0, 0, 1]
    result_single = radiographic_agreement(y_true_single, y_pred_single)
    print("Single-label results:", result_single)

    # Test with multi-label data (e.g., multiple findings)
    y_true_multi = [[1, 0], [0, 1], [1, 1], [0, 0]]  # [pneumonia, edema]
    y_pred_multi = [[1, 0], [0, 0], [1, 0], [0, 1]]
    result_multi = radiographic_agreement(y_true_multi, y_pred_multi)
    print("Multi-label results:", result_multi)