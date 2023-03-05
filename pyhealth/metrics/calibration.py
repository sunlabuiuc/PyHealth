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

def ECE_classwise(prob, label, bins=20, threshold=0., adaptive=False):
    K = prob.shape[1]
    if len(label.shape) == 1:
        # make it one-hot
        label, _ = np.zeros((len(label),K)), label
        label[np.arange(len(label)), _] = 1
    return _ECE_classwise(prob, label, bins, threshold, adaptive)[1]['avg']

def ECE_confidence_multiclass(prob:np.ndarray, label:np.ndarray, bins=20, adaptive=False):
    df = pd.DataFrame({'acc': label == np.argmax(prob, 1), 'conf': prob.max(1)})
    return _ECE_confidence(df, bins, adaptive)[1]

def ECE_confidence_binary(prob:np.ndarray, label:np.ndarray, bins=20, adaptive=False):
    df = pd.DataFrame({'acc': label[:,0], 'conf': prob[:,0]})
    return _ECE_confidence(df, bins, adaptive)[1]

def brier_top1(prob:np.ndarray, label:np.ndarray):
    conf = prob.max(1)
    acc = (label == np.argmax(prob, 1)).astype(int)
    return np.mean(np.square(conf - acc))