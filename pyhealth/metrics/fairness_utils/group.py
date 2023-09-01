import numpy as np

"""
Notation:
    - Protected group: P
    - Unprotected group: U
"""

def disparate_impact(sensitive_attributes: np.ndarray, y_pred: np.ndarray, favorable_outcome: int = 1, allow_zero_division = False, epsilon: float = 1e-8) -> float:
    """
    Computes the disparate impact between the the protected and unprotected group.

    disparate_impact = P(y_pred = favorable_outcome | P) / P(y_pred = favorable_outcome | U)
    
    Args:
        sensitive_attributes: Sensitive attributes of shape (n_samples,) where 1 is the protected group and 0 is the unprotected group.
        y_pred: Predicted target values of shape (n_samples,).
        favorable_outcome: Label value which is considered favorable (i.e. "positive").
        allow_zero_division: If True, use epsilon instead of 0 in the denominator if the denominator is 0. Otherwise, raise a ValueError.
    
    Returns:
        The disparate impact between the protected and unprotected group.
    """
    
    p_fav_unpr = np.sum(y_pred[sensitive_attributes == 0] == favorable_outcome) / len(y_pred[sensitive_attributes == 0])
    p_fav_prot = np.sum(y_pred[sensitive_attributes == 1] == favorable_outcome) / len(y_pred[sensitive_attributes == 1])

    if p_fav_unpr == 0:
        if allow_zero_division:
            p_fav_unpr = epsilon
        else:
            raise ValueError("Unprotected group has no instances with a favorable outcome. Disparate impact is undefined.")

    disparate_impact_value = p_fav_prot / p_fav_unpr

    return disparate_impact_value

def statistical_parity_difference(sensitive_attributes: np.ndarray, y_pred: np.ndarray, favorable_outcome: int = 1) -> float:
    """
    Computes the statistical parity difference between the the protected and unprotected group.

    statistical_parity_difference = P(y_pred = favorable_outcome | P) - P(y_pred = favorable_outcome | U)
    Args:
        sensitive_attributes: Sensitive attributes of shape (n_samples,) where 1 is the protected group and 0 is the unprotected group.
        y_pred: Predicted target values of shape (n_samples,).
        favorable_outcome: Label value which is considered favorable (i.e. "positive").
    Returns:
        The statistical parity difference between the protected and unprotected group.
    """

    p_fav_unpr = np.sum(y_pred[sensitive_attributes == 0] == favorable_outcome) / len(y_pred[sensitive_attributes == 0])
    p_fav_prot = np.sum(y_pred[sensitive_attributes == 1] == favorable_outcome) / len(y_pred[sensitive_attributes == 1])
    
    statistical_parity_difference_value = p_fav_prot - p_fav_unpr

    return statistical_parity_difference_value



    