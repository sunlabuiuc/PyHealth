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


def accuracy_multilabel(y_true, y_pred, **kwargs):
    """Accuracy classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted accuracy score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(accuracy_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def precision_multilabel(y_true, y_pred, **kwargs):
    """Precision classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted precision score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(precision_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def recall_multilabel(y_true, y_pred, **kwargs):
    """Recall classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted recall score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(recall_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def f1_multilabel(y_true, y_pred, **kwargs):
    """F1 classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted f1 score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(f1_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def roc_auc_multilabel(y_true, y_pred, **kwargs):
    """ROC AUC classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted roc auc score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(roc_auc_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def pr_auc_multilabel(y_true, y_pred, **kwargs):
    """PR AUC classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted pr auc score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(average_precision_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def jaccard_multilabel(y_true, y_pred, **kwargs):
    """Jaccard classification score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted jaccard score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(jaccard_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def r2_score_multilabel(y_true, y_pred, **kwargs):
    """R2 regression score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted r2 score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(r2_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def cohen_kappa_multilabel(y_true, y_pred, **kwargs):
    """Cohen Kappa regression score.
    INPUTS:
        - y_true: array-like of shape (n_samples, n_classes)
        - y_pred: array-like of shape (n_samples, n_classes)
        - label_weight: list of weights for each label (n_classes,)
    OUTPUTS:
        - result: weighted cohen kappa score over labels
    """
    result = []
    for i in range(y_true.shape[0]):
        result.append(cohen_kappa_score(y_true[i], y_pred[i], **kwargs))
    return np.mean(result)


def ddi_rate_score(medications, ddi_matrix):
    """DDI rate score.
    INPUTS:
        - medications: array-like of shape (n_samples, n_classes)
        - ddi_matrix: array-like of shape (n_classes, n_classes)
    OUTPUTS:
        - result: DDI rate score
    """
    ddi_matrix = np.array(ddi_matrix)

    all_cnt = 0
    dd_cnt = 0
    for sample in medications:
        for i, med_i in enumerate(sample):
            for j, med_j in enumerate(sample):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt
