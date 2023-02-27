from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics


def multiclass_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Computes metrics for multiclass classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - roc_auc_macro_ovo: area under the receiver operating characteristic curve,
            macro averaged over one-vs-one multiclass classification
        - roc_auc_macro_ovr: area under the receiver operating characteristic curve,
            macro averaged over one-vs-rest multiclass classification
        - roc_auc_weighted_ovo: area under the receiver operating characteristic curve,
            weighted averaged over one-vs-one multiclass classification
        - roc_auc_weighted_ovr: area under the receiver operating characteristic curve,
            weighted averaged over one-vs-rest multiclass classification
        - accuracy: accuracy score
        - balanced_accuracy: balanced accuracy score (usually used for imbalanced
            datasets)
        - f1_micro: f1 score, micro averaged
        - f1_macro: f1 score, macro averaged
        - f1_weighted: f1 score, weighted averaged
        - jaccard_micro: Jaccard similarity coefficient score, micro averaged
        - jaccard_macro: Jaccard similarity coefficient score, macro averaged
        - jaccard_weighted: Jaccard similarity coefficient score, weighted averaged
        - cohen_kappa: Cohen's kappa score
    If no metrics are specified, accuracy, f1_macro, and f1_micro are computed
    by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples, n_classes).
        metrics: List of metrics to compute. Default is ["accuracy", "f1_macro",
            "f1_micro"].

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import multiclass_metrics_fn
        >>> y_true = np.array([0, 1, 2, 2])
        >>> y_prob = np.array([[0.9,  0.05, 0.05],
        ...                    [0.05, 0.9,  0.05],
        ...                    [0.05, 0.05, 0.9],
        ...                    [0.6,  0.2,  0.2]])
        >>> multiclass_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_micro"]
        metrics += ['brier_top1', 'ECE', 'ECE_adapt', 'cwECEt', 'cwECEt_adapt']

    y_pred = np.argmax(y_prob, axis=-1)

    output = {}
    for metric in metrics:
        if metric == "roc_auc_macro_ovo":
            roc_auc_macro_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovo"
            )
            output["roc_auc_macro_ovo"] = roc_auc_macro_ovo
        elif metric == "roc_auc_macro_ovr":
            roc_auc_macro_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            )
            output["roc_auc_macro_ovr"] = roc_auc_macro_ovr
        elif metric == "roc_auc_weighted_ovo":
            roc_auc_weighted_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovo"
            )
            output["roc_auc_weighted_ovo"] = roc_auc_weighted_ovo
        elif metric == "roc_auc_weighted_ovr":
            roc_auc_weighted_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovr"
            )
            output["roc_auc_weighted_ovr"] = roc_auc_weighted_ovr
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro")
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "jaccard_micro":
            jacard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jacard_micro
        elif metric == "jaccard_macro":
            jacard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jacard_macro
        elif metric == "jaccard_weighted":
            jacard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jacard_weighted
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == 'brier_top1':
            _,_,output[metric] = CalibrationEval.ECE_confidence(y_prob, y_true, bins=20)
        elif metric in {'ECE', 'ECE_adapt'}:
            _,output[metric],_ = CalibrationEval.ECE_confidence(y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt"))
        elif metric in {'cwECEt', 'cwECEt_adapt'}:
            thres = min(0.01, 1./y_prob.shape[1])
            _, classECE_by_class = CalibrationEval.ECE_class(y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt"), threshold=thres)
            output[metric] = classECE_by_class['avg']
        else:
            raise ValueError(f"Unknown metric for multiclass classification: {metric}")

    return output


class CalibrationEval:
    
    @classmethod
    def get_bins(cls, bins):
        if isinstance(bins, int):
            bins = list(np.arange(bins+1) / bins)
        return bins
    
    @classmethod
    def assign_bin(cls, sorted_ser, bins, adaptive=False):
        ret = pd.DataFrame(sorted_ser)
        if adaptive:
            assert isinstance(bins, int)
            step = len(sorted_ser) // bins
            nvals = [step for _ in range(bins)]
            for _ in range(len(sorted_ser) % bins): nvals[-_-1] += 1
            ret['bin'] = [ith for ith, val in enumerate(nvals) for _ in range(val)]
            nvals = list(np.asarray(nvals).cumsum())
            bins = [ret.iloc[0]['conf']]
            for iloc in nvals:
                bins.append(ret.iloc[iloc-1]['conf'])
                if iloc != nvals[-1]:
                    bins[-1] = 0.5 * bins[-1] + 0.5 *ret.iloc[iloc]['conf']
        else:
            bins = cls.get_bins(bins)
            import bisect

            bin_assign = pd.Series(0, index=sorted_ser.index)
            locs = [bisect.bisect(sorted_ser, b) for b in bins]
            locs[0], locs[-1] = 0, len(ret)
            for i, loc in enumerate(locs[:-1]):
                bin_assign.iloc[loc:locs[i+1]] = i
            ret['bin'] = bin_assign
        return ret['bin'], bins

    @classmethod
    def _ECE_loss(cls, summ):
        w = summ['cnt'] / summ['cnt'].sum()
        loss = np.average((summ['conf'] - summ['acc']).abs(), weights=w)
        return loss

    @classmethod
    def ECE_confidence(cls, preds, label, bins=20, adaptive=False, return_bins=False):
        df = pd.DataFrame({"conf": preds.max(1), 'truth': label, 'pred': np.argmax(preds, 1)}).sort_values(['conf']).reset_index()
        df['acc'] = (df['truth'] == df['pred']).astype(int)
        df['bin'], bin_boundary = cls.assign_bin(df['conf'], bins, adaptive=adaptive)
        summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())#.fillna(0.)
        summ['cnt'] = df.groupby('bin').size()
        summ = summ.reset_index()
        if return_bins: return summ, cls._ECE_loss(summ), np.mean(np.square(df['conf'].values - df['acc'].values)), bin_boundary
        return summ, cls._ECE_loss(summ), np.mean(np.square(df['conf'].values - df['acc'].values))
    @classmethod
    def ECE_class(cls, preds, label, bins=15, threshold=0., adaptive=False, return_bins=False):
        K = preds.shape[1]
        summs = []
        class_losses = {}
        bin_boundaries = {}
        for k in range(K):
            msk = preds[:, k] >= threshold
            if msk.sum() == 0: continue
            df = pd.DataFrame({"conf": preds[msk, k], 'truth': label[msk]}).sort_values(['conf']).reset_index()
            df['acc'] = (df['truth'] == k).astype(int)
            df['bin'], bin_boundaries[k] = cls.assign_bin(df['conf'], bins, adaptive=adaptive)
            summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())
            summ['cnt'] = df.groupby('bin').size()
            summ['k'] = k
            summs.append(summ.reset_index())
            class_losses[k] = cls._ECE_loss(summs[-1])
        class_losses = pd.Series(class_losses)
        class_losses['avg'], class_losses['sum'] = class_losses.mean(), class_losses.sum()
        summs = pd.concat(summs, ignore_index=True)
        if return_bins: return summs, class_losses, bin_boundaries
        return summs, class_losses
    
if __name__ == "__main__":
    all_metrics = [
        "roc_auc_macro_ovo",
        "roc_auc_macro_ovr",
        "roc_auc_weighted_ovo",
        "roc_auc_weighted_ovr",
        "accuracy",
        "balanced_accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "jaccard_micro",
        "jaccard_macro",
        "jaccard_weighted",
        "cohen_kappa",
    ]
    all_metrics += ['brier_top1', 'ECE', 'ECE_adapt', 'cwECEt', 'cwECEt_adapt']
    y_true = np.random.randint(4, size=100000)
    y_prob = np.random.randn(100000, 4)
    y_prob = np.exp(y_prob) / np.sum(np.exp(y_prob), axis=-1, keepdims=True)
    print(multiclass_metrics_fn(y_true, y_prob, metrics=all_metrics))
