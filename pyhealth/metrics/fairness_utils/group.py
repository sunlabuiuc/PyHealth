import numpy as np

"""
Notation:
    - Protected group: P
    - Unprotected group: U
"""


def _favorable_outcome_rate(
    group_y_pred: np.ndarray, favorable_outcome: int, group_name: str
) -> float:
    """Computes P(y_pred = favorable_outcome) for a single group's predictions.

    Args:
        group_y_pred: Predicted target values for one group (already
            filtered by the sensitive attribute), of shape (n_group,).
        favorable_outcome: Label value which is considered favorable.
        group_name: Human-readable group name, used only in the error
            message if the group is empty.

    Returns:
        The favorable-outcome rate for this group. This is a genuine
        float, never NaN: an empty group raises ValueError instead of
        silently producing a 0/0 NaN that downstream code (or a naive
        ``== 0`` check) can't detect, since NaN never equals 0.

    Raises:
        ValueError: If the group has no instances (0 samples). The rate
            is undefined in that case -- this is different from a
            non-empty group whose favorable-outcome rate happens to be
            exactly 0, which is a legitimate value.
    """
    n = len(group_y_pred)
    if n == 0:
        raise ValueError(
            f"The {group_name} group has no instances (0 samples); "
            "the favorable-outcome rate is undefined."
        )
    return float(np.sum(group_y_pred == favorable_outcome) / n)


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

    Raises:
        ValueError: If either group has no instances at all (this is
            always an error, regardless of allow_zero_division -- there
            is no meaningful epsilon substitute for a group we have zero
            information about), or if the unprotected group's
            favorable-outcome rate is exactly 0 and allow_zero_division
            is False.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.fairness_utils import disparate_impact
        >>> sensitive_attributes = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([1, 0, 1, 1, 0])
        >>> disparate_impact(sensitive_attributes, y_pred)
        1.3333333333333333
    """
    p_fav_unpr = _favorable_outcome_rate(
        y_pred[sensitive_attributes == 0], favorable_outcome, "unprotected"
    )
    p_fav_prot = _favorable_outcome_rate(
        y_pred[sensitive_attributes == 1], favorable_outcome, "protected"
    )

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

    Raises:
        ValueError: If either group has no instances at all. Unlike
            disparate_impact, a favorable-outcome rate of exactly 0 for
            a non-empty group is not an error here (it's a legitimate
            value for a difference, e.g. 0 - 0.3 = -0.3).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.fairness_utils import statistical_parity_difference
        >>> sensitive_attributes = np.array([0, 0, 1, 1, 1])
        >>> y_pred = np.array([1, 0, 1, 1, 0])
        >>> statistical_parity_difference(sensitive_attributes, y_pred)
        0.16666666666666663
    """
    p_fav_unpr = _favorable_outcome_rate(
        y_pred[sensitive_attributes == 0], favorable_outcome, "unprotected"
    )
    p_fav_prot = _favorable_outcome_rate(
        y_pred[sensitive_attributes == 1], favorable_outcome, "protected"
    )

    statistical_parity_difference_value = p_fav_prot - p_fav_unpr

    return statistical_parity_difference_value

    