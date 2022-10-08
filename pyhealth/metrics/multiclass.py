import numpy as np
import sklearn.metrics as metrics


def accuracy_score(y_true, y_pred, **kwargs):
    """Accuracy classification score.
    Wrapper for sklearn.metrics.accuracy_score
    """
    return metrics.accuracy_score(y_true, y_pred, **kwargs)


def precision_score(y_true, y_pred, **kwargs):
    """Precision classification score.
    Wrapper for sklearn.metrics.precision_score
    """
    return metrics.precision_score(y_true, y_pred, **kwargs)


def recall_score(y_true, y_pred, **kwargs):
    """Recall classification score.
    Wrapper for sklearn.metrics.recall_score
    """
    return metrics.recall_score(y_true, y_pred, **kwargs)


def f1_score(y_true, y_pred, **kwargs):
    """F1 classification score.
    Wrapper for sklearn.metrics.f1_score
    """
    return metrics.f1_score(y_true, y_pred, **kwargs)


def roc_auc_score(y_true, y_pred, **kwargs):
    """ROC AUC classification score.
    Wrapper for sklearn.metrics.roc_auc_score
    """
    return metrics.roc_auc_score(y_true, y_pred, **kwargs)


def average_precision_score(y_true, y_pred, **kwargs):
    """PR AUC classification score.
    Wrapper for sklearn.metrics.average_precision_score
    """
    return metrics.average_precision_score(y_true, y_pred, **kwargs)


def jaccard_score(y_true, y_pred, **kwargs):
    """Jaccard similarity score.
    Wrapper for sklearn.metrics.jaccard_score
    """
    return metrics.jaccard_score(y_true, y_pred, **kwargs)


def confusion_matrix(y_true, y_pred, **kwargs):
    """Confusion matrix.
    Wrapper for sklearn.metrics.confusion_matrix
    """
    return metrics.confusion_matrix(y_true, y_pred, **kwargs)


def cohen_kappa_score(y_true, y_pred, **kwargs):
    """Cohen's kappa score.
    Wrapper for sklearn.metrics.cohen_kappa_score
    """
    return metrics.cohen_kappa_score(y_true, y_pred, **kwargs)


def r2_score(y_true, y_pred, **kwargs):
    """R2 score.
    Wrapper for sklearn.metrics.r2_score
    """
    return metrics.r2_score(y_true, y_pred, **kwargs)
