import numpy as np


def size(y_pred:np.ndarray):
    """Average size of the prediction set.
    """
    return np.mean(y_pred.sum(1))

def rejection_rate(y_pred:np.ndarray):
    """Rejection rate, defined as the proportion of samples with prediction set size != 1
    """
    return np.mean(y_pred.sum(1) != 1)

def _missrate(y_pred:np.ndarray, y_true:np.ndarray, ignore_rejected=False):
    """Computes the class-wise mis-coverage rate (or risk).

    Args:
        y_pred (np.ndarray): prediction scores.
        y_true (np.ndarray): true labels.
        ignore_rejected (bool, optional): If True, we compute the miscoverage rate
            without rejection  (that is, condition on the unrejected samples). Defaults to False.

    Returns:
        np.ndarray: miss-coverage rates for each class.
    """
    # currently handles multilabel and multiclass
    K = y_pred.shape[1]
    if len(y_true.shape) == 1:
        y_true, _ = np.zeros((len(y_true),K), dtype=bool), y_true
        y_true[np.arange(len(y_true)), _] = 1
    y_true = y_true.astype(bool)

    keep_msk = (y_pred.sum(1) == 1) if ignore_rejected else np.ones(len(y_true), dtype=bool)
    missed = []
    for k in range(K):
        missed.append(1-np.mean(y_pred[keep_msk & y_true[:, k], k]))

    return np.asarray(missed)



def miscoverage_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for all samples (similar to recall).

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0],[0, 1, 0]])
        >>> y_true = np.asarray([1,0,1,2])
        >>> error_ps(y_pred, y_true)
        array([0. , 0.5, 1. ])


    Explanation:
    For class 0, the 1-th prediction set ({0}) contains the label, so the miss-coverage is 0/1=0.
    For class 1, the 0-th prediction set ({0}) does not contain the label, the 2-th prediction
    set ({0,1}) contains the label. Thus, the miss-coverage is 1/2=0.5.
    For class 2, the last prediction set is {1} and the label is 2, so the miss-coverage is 1/1=1.
    """
    return _missrate(y_pred, y_true, False)

def error_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for unrejected samples, where rejection is defined to be sets with size !=1).

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0],[0, 1, 0]])
        >>> y_true = np.asarray([1,0,1,2])
        >>> error_ps(y_pred, y_true)
        array([0., 1., 1.])

    Explanation:
    For class 0, the 1-th sample is correct and not rejected, so the error is 0/1=0.
    For class 1, the 0-th sample is incorrerct and not rejected, the 2-th is rejected.
    Thus, the error is 1/1=1.
    For class 2, the last sample is not-rejected but the prediction set is {1}, so the error
    is 1/1=1.
    """
    return _missrate(y_pred, y_true, True)

def miscoverage_overall_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rate for the true label. Only for multiclass.

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0]])
        >>> y_true = np.asarray([1,0,1])
        >>> miscoverage_overall_ps(y_pred, y_true)
        0.333333

    Explanation:
    The 0-th prediction set is {0} and the label is 1 (not covered).
    The 1-th prediction set is {0} and the label is 0 (covered).
    The 2-th prediction set is {0,1} and the label is 1 (covered).
    Thus the miscoverage rate is 1/3.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]

    return 1 - np.mean(truth_pred)

def error_overall_ps(y_pred:np.ndarray, y_true:np.ndarray):
    """Overall error rate for the un-rejected samples.

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0]])
        >>> y_true = np.asarray([1,0,1])
        >>> error_overall_ps(y_pred, y_true)
        0.5

    Explanation:
    The 0-th prediction set is {0} and the label is 1, so it is an error (no rejection
    as its prediction set has only one class).
    The 1-th sample is not rejected and incurs on error.
    The 2-th sample is rejected, thus excluded from the computation.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]
    truth_pred = truth_pred[y_pred.sum(1) == 1]
    return 1 - np.mean(truth_pred)
