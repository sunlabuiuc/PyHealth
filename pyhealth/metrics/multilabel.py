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


def accuracy_multilabel(y_true, y_pred, label_weight, **kwargs):
    """Accuracy classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted accuracy score over labels
    """
    result = 0
    for w in label_weight:
        result += w * accuracy_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def precision_multilabel(y_true, y_pred, label_weight, **kwargs):
    """Precision classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted precision score over labels
    """
    result = 0
    for w in label_weight:
        result += w * precision_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def recall_multilabel(y_true, y_pred, label_weight, **kwargs):
    """Recall classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted recall score over labels
    """
    result = 0
    for w in label_weight:
        result += w * recall_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def f1_multilabel(y_true, y_pred, label_weight, **kwargs):
    """F1 classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted F1 score over labels
    """
    result = 0
    for w in label_weight:
        result += w * f1_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def roc_auc_multilabel(y_true, y_pred, label_weight, **kwargs):
    """ROC AUC classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted ROC AUC score over labels
    """
    result = 0
    for w in label_weight:
        result += w * roc_auc_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def pr_auc_multilabel(y_true, y_pred, label_weight, **kwargs):
    """PR AUC classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted PR AUC score over labels
    """
    result = 0
    for w in label_weight:
        result += w * average_precision_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def jaccard_multilabel(y_true, y_pred, label_weight, **kwargs):
    """Jaccard classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted Jaccard score over labels
    """
    result = 0
    for w in label_weight:
        result += w * jaccard_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def cohen_kappa_multilabel(y_true, y_pred, label_weight, **kwargs):
    """Cohen Kappa classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted Cohen Kappa score over labels
    """
    result = 0
    for w in label_weight:
        result += w * cohen_kappa_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def r2_score_multilabel(y_true, y_pred, label_weight, **kwargs):
    """R2 score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted R2 score over labels
    """
    result = 0
    for w in label_weight:
        result += w * r2_score(y_true, y_pred, **kwargs)
    return result / sum(label_weight)


def ddi_rate_score(predicted, ddi_matrix, threshold=0.4):
    """DDI rate score.
    INPUTS:
        - predicted: float based array-like of shape (n_samples, n_classes)
        - ddi_matrix: array-like of shape (n_classes, n_classes)
        - threshold: float
    OUTPUTS:
        - result: DDI rate score
    """
    all_cnt = 0
    dd_cnt = 0
    for visit in predicted:
        cur_med = np.where(visit >= threshold)[0]
        for i, med_i in enumerate(cur_med):
            for j, med_j in enumerate(cur_med):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt
