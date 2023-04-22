"""Metrics that meature model calibration.

Reference Papers:

    [1] Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun.
    "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration
    for Deep Neural Networks."
    ICLR 2023.

    [2] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran.
    "Measuring Calibration in Deep Learning."
    In CVPR workshops, vol. 2, no. 7. 2019.

    [3] Patel, Kanil, William Beluch, Bin Yang, Michael Pfeiffer, and Dan Zhang.
    "Multi-class uncertainty calibration via mutual information maximization-based binning."
    ICLR 2021.

    [4] Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger.
    "On calibration of modern neural networks."
    ICML 2017.

    [5] Kull, Meelis, Miquel Perello Nieto, Markus Kängsepp, Telmo Silva Filho, Hao Song, and Peter Flach.
    "Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration."
    Advances in neural information processing systems 32 (2019).

    [6] Brier, Glenn W.
    "Verification of forecasts expressed in terms of probability."
    Monthly weather review 78, no. 1 (1950): 1-3.

"""
import bisect

import numpy as np
import pandas as pd


def _get_bins(bins):
    if isinstance(bins, int):
        bins = list(np.arange(bins+1) / bins)
    return bins

def assign_bin(sorted_ser: pd.Series, bins:int, adaptive:bool=False):
    ret = pd.DataFrame(sorted_ser)
    if adaptive:
        assert isinstance(bins, int)
        step = len(sorted_ser) // bins
        nvals = [step for _ in range(bins)]
        for _ in range(len(sorted_ser) % bins):
            nvals[-_-1] += 1
        ret['bin'] = [ith for ith, val in enumerate(nvals) for _ in range(val)]
        nvals = list(np.asarray(nvals).cumsum())
        bins = [ret.iloc[0]['conf']]
        for iloc in nvals:
            bins.append(ret.iloc[iloc-1]['conf'])
            if iloc != nvals[-1]:
                bins[-1] = 0.5 * bins[-1] + 0.5 *ret.iloc[iloc]['conf']
    else:
        bins = _get_bins(bins)
        bin_assign = pd.Series(0, index=sorted_ser.index)
        locs = [bisect.bisect(sorted_ser.values, b) for b in bins]
        locs[0], locs[-1] = 0, len(ret)
        for i, loc in enumerate(locs[:-1]):
            bin_assign.iloc[loc:locs[i+1]] = i
        ret['bin'] = bin_assign
    return ret['bin'], bins

def _ECE_loss(summ):
    w = summ['cnt'] / summ['cnt'].sum()
    loss = np.average((summ['conf'] - summ['acc']).abs(), weights=w)
    return loss

def _ECE_confidence(df, bins=20, adaptive=False):
    # df should have columns: conf, acc
    df = df.sort_values(['conf']).reset_index().drop('index', axis=1)
    df['bin'], _ = assign_bin(df['conf'], bins, adaptive=adaptive)
    summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())#.fillna(0.)
    summ['cnt'] = df.groupby('bin').size()
    summ = summ.reset_index()
    return summ, _ECE_loss(summ)

def _ECE_classwise(prob:np.ndarray, label_onehot:np.ndarray, bins=20, threshold=0., adaptive=False):
    summs = []
    class_losses = {}
    for k in range(prob.shape[1]):
        msk = prob[:, k] >= threshold
        if msk.sum() == 0:
            continue
        df = pd.DataFrame({"conf": prob[msk, k], 'acc': label_onehot[msk, k]})
        df = df.sort_values(['conf']).reset_index()
        df['bin'], _ = assign_bin(df['conf'], bins, adaptive=adaptive)
        summ = pd.DataFrame(df.groupby('bin')[['acc', 'conf']].mean())
        summ['cnt'] = df.groupby('bin').size()
        summ['k'] = k
        summs.append(summ.reset_index())
        class_losses[k] = _ECE_loss(summs[-1])
    class_losses = pd.Series(class_losses)
    class_losses['avg'], class_losses['sum'] = class_losses.mean(), class_losses.sum()
    summs = pd.concat(summs, ignore_index=True)
    return summs, class_losses

def ece_confidence_multiclass(prob:np.ndarray, label:np.ndarray, bins=20, adaptive=False):
    """Expected Calibration Error (ECE).

    We group samples into 'bins' basing on the top-class prediction.
    Then, we compute the absolute difference between the average top-class prediction and
    the frequency of top-class being correct (i.e. accuracy) for each bin.
    ECE is the average (weighed by number of points in each bin) of these absolute differences.
    It could be expressed by the following formula, with :math:`B_m` denoting the m-th bin:

    .. math::
        ECE = \\sum_{m=1}^M \\frac{|B_m|}{N} |acc(B_m) - conf(B_m)|

    Example:
        >>> pred = np.asarray([[0.2, 0.2, 0.6], [0.2, 0.31, 0.49], [0.1, 0.1, 0.8]])
        >>> label = np.asarray([2,1,2])
        >>> ECE_confidence_multiclass(pred, label, bins=2)
        0.36333333333333334

    Explanation of the example: The bins are [0, 0.5] and (0.5, 1].
    In the first bin, we have one sample with top-class prediction of 0.49, and its
    accuracy is 0. In the second bin, we have average confidence of 0.7 and average
    accuracy of 1. Thus, the ECE is :math:`\\frac{1}{3} \cdot 0.49 + \\frac{2}{3}\cdot 0.3=0.3633`.

    Args:
        prob (np.ndarray): (N, C)
        label (np.ndarray): (N,)
        bins (int, optional): Number of bins. Defaults to 20.
        adaptive (bool, optional): If False, bins are equal width ([0, 0.05, 0.1, ..., 1])
            If True, bin widths are adaptive such that each bin contains the same number
            of points. Defaults to False.
    """
    df = pd.DataFrame({'acc': label == np.argmax(prob, 1), 'conf': prob.max(1)})
    return _ECE_confidence(df, bins, adaptive)[1]

def ece_confidence_binary(prob:np.ndarray, label:np.ndarray, bins=20, adaptive=False):
    """Expected Calibration Error (ECE) for binary classification.

    Similar to :func:`ece_confidence_multiclass`, but on class 1 instead of the top-prediction.


    Args:
        prob (np.ndarray): (N, C)
        label (np.ndarray): (N,)
        bins (int, optional): Number of bins. Defaults to 20.
        adaptive (bool, optional): If False, bins are equal width ([0, 0.05, 0.1, ..., 1])
            If True, bin widths are adaptive such that each bin contains the same number
            of points. Defaults to False.
    """

    df = pd.DataFrame({'acc': label[:,0], 'conf': prob[:,0]})
    return _ECE_confidence(df, bins, adaptive)[1]

def ece_classwise(prob, label, bins=20, threshold=0., adaptive=False):
    """Classwise Expected Calibration Error (ECE).

    This is equivalent to applying :func:`ece_confidence_binary` to each class and take the average.

    Args:
        prob (np.ndarray): (N, C)
        label (np.ndarray): (N,)
        bins (int, optional): Number of bins. Defaults to 20.
        threshold (float): threshold to filter out samples.
            If the number of classes C is very large, many classes receive close to 0
            prediction. Any prediction below threshold is considered noise and ignored.
            In recent papers, this is typically set to a small number (such as 1/C).
        adaptive (bool, optional): If False, bins are equal width ([0, 0.05, 0.1, ..., 1])
            If True, bin widths are adaptive such that each bin contains the same number
            of points. Defaults to False.
    """

    K = prob.shape[1]
    if len(label.shape) == 1:
        # make it one-hot
        label, _ = np.zeros((len(label),K)), label
        label[np.arange(len(label)), _] = 1
    return _ECE_classwise(prob, label, bins, threshold, adaptive)[1]['avg']

def brier_top1(prob:np.ndarray, label:np.ndarray):
    """Brier score (i.e. mean squared error between prediction and 0-1 label) of the top prediction.
    """
    conf = prob.max(1)
    acc = (label == np.argmax(prob, 1)).astype(int)
    return np.mean(np.square(conf - acc))
