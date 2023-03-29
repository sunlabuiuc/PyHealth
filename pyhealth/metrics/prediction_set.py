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
    """Miscoverage rates for all samples (similar to recall).

    For example, if our prediction sets are {1}, {1}, {0,2} and the labels
    are 0, 1, 1. Then, the micoverage rate for class 1 is 0.5 (two samples
    belong to class 1, but one of the prediction sets does not contain 1).
    Similarly, the miscoverage rate for class 0 is 0.
    """
    return _missrate(y_pred, y_true, False)

def missrate_certain(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rates for unrejected samples,
        where rejection is defined to be sets with size !=1).

    For example, if our prediction sets are {1}, {1}, {0,2} and the labels
    are 0, 1, 1. Then, the micoverage rate for unrejected samples for class 1
    is 0: only the second sample is unrejected, and belongs to class 1, and its
    prediction set contains class 1. Similarly, this rate for class 0 is 1.
    """
    return _missrate(y_pred, y_true, True)

def missrate_overall(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rate for the true label. Only for multiclass.


    For example, if a prediction set is {0,1} and the label is 2, then this is an error.
    If a prediction set is {0,1} and the label is 1, this is not an error.
    Miscoverage is the average of errors. The overall miscoverage rate for these two samples
    is 0.5.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]

    return 1 - np.mean(truth_pred)

def missrate_certain_overall(y_pred:np.ndarray, y_true:np.ndarray):
    """Miscoverage rate for the true label for unrejected samples. Only for multiclass.

    For example, if a prediction set is {0} and the label is 1, it is an error.
    {0} and 0 incurs no error. {0,1} and 1 is *ignored*.
    If we compute miscoverage rates for unrejected samples for these three
    samples, we get 0.5. Note this is similar to accuracy on the unrejected samples.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]
    truth_pred = truth_pred[y_pred.sum(1) == 1]
    return 1 - np.mean(truth_pred)
