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



def missrate(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for all samples (similar to recall)."""
    return _missrate(y_pred, y_true, False)

def missrate_certain(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for unrejected samples,
        where rejection is defined to be sets with size !=1).
    """
    return _missrate(y_pred, y_true, True)

def missrate_overall(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rate for the true label. Only for multiclass"""
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]

    return 1 - np.mean(truth_pred)

def missrate_certain_overall(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rate for the true label for unrejected samples. Only for multiclass"""
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]
    truth_pred = truth_pred[y_pred.sum(1) == 1]
    return 1 - np.mean(truth_pred)
