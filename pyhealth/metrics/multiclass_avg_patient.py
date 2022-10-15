import numpy as np
from .multiclass import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    jaccard_score,
    cohen_kappa_score,
    r2_score,
)


def accuracy_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """Accuracy classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged accuracy over patients
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * accuracy_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())


def precision_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """Precision classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged precision over patients
    """
    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * precision_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())


def recall_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """Recall classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged recall over patients
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * recall_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())


def f1_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """F1 classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged F1 over patients
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * f1_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())


def roc_auc_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """ROC AUC classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged ROC AUC over patients
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * roc_auc_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())


def pr_auc_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """PR AUC classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged PR AUC over patients
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * average_precision_score(
            y_true[idx], y_pred[idx], **kwargs
        )
    return result / sum(pat_weight.values())


def jaccard_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """Jaccard classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged Jaccard over patients
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * jaccard_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())


def cohen_kappa_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """Cohen Kappa classification averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged Cohen Kappa over patients
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * cohen_kappa_score(
            y_true[idx], y_pred[idx], **kwargs
        )
    return result / sum(pat_weight.values())


def r2_score_avg_patient(y_true, y_pred, pat_id, pat_weight, **kwargs):
    """R2 regression averaged over patients.
    Arguments
        - y_true: a list of true labels
        - y_pred: a list of predicted labels
        - pat_id: a list of patient identities
        - pat_weight: a dictionary of <pat_id, weight>
    Returns
        - result: weighted averaged R2 over patients
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = 0
    for pat in pat_weight:
        idx = np.where(np.array(pat_id) == pat)[0]
        result += pat_weight[pat] * r2_score(y_true[idx], y_pred[idx], **kwargs)
    return result / sum(pat_weight.values())
