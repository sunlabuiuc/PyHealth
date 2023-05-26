"""Wrapper for the coordinate descent ("QuickSearch")"""
import numpy as np

from pyhealth.calib.utils import one_hot_np

_CYTHON_ENABLED = False
try:
    import pyximport
    pyximport.install()
    from . import quicksearch_cython as cdc
    _CYTHON_ENABLED = True
except:
    print("This is a warning of potentially slow compute. You could uncomment this line and use the Python implementation instead of Cython.")

__all__ = ['loss_overall', 'loss_classspecific',
           'coord_desc_overall', 'coord_desc_classspecific']

def _thresholding_py(ts, output):
    pred = np.asarray(output > ts, dtype=np.int32)
    return pred

def __loss_overall_helper(total_err, total_sure, alpha, N, la, lc, lcs):
    ambiguity_loss = (1. - total_sure / float(N)) ** 2
    risk = total_err / float(max(total_sure, 1))
    tempf = risk - alpha
    coverage_loss = np.power(max(tempf, 0), 2)
    coverage_loss_sim = np.power(tempf, 2)
    loss = ambiguity_loss * la + coverage_loss * lc + coverage_loss_sim * lcs
    return loss

def __loss_class_specific_complete_helper(err, sure, rs, weights, total_sure, N, la, lc, lcs):
    a_loss = 1- total_sure / N
    if sure.min() == 0: return np.inf
    tempf = err / sure
    if rs is not None:
        tempf -= rs
    if weights is None:
        c_loss = np.power(tempf.clip(0, 1), 2).sum()
        cs_loss = np.power(tempf, 2).sum()
    else:
        c_loss = np.dot(np.power(tempf.clip(0, 1), 2), weights)
        cs_loss = np.dot(np.power(tempf, 2), weights)
    return a_loss * la + c_loss * lc + cs_loss * lcs

def __update_cnts(pred, y, err, sure, inc=1):
    if pred != y:
        err[y] += inc
    sure[y] += inc
    return inc

def loss_overall_py(preds, labels, max_classes, risk, *, lk=1e4, fill_max=False):
    cnt = preds.sum(1)
    cnt1 = np.expand_dims(cnt == 1, 1)
    total_sure = sum(cnt1.squeeze(1))
    risk_indicator = labels * (1 - preds)
    total_err = (risk_indicator * cnt1).sum()
    if fill_max:
        cnt0 = cnt == 0
        total_sure += sum(cnt0)
        new_risk = max_classes != np.argmax(labels, 1)
        total_err += sum(cnt0 & new_risk)
    return __loss_overall_helper(total_err, total_sure, risk, len(labels), la=1, lc=lk, lcs=0)

def loss_class_specific_py(preds, labels, max_classes, risk, *, class_weights=False, lk=1e4, fill_max=False):
    N,K = preds.shape
    cnt = preds.sum(1)
    sure_msk = cnt == 1
    total_sure = np.sum(sure_msk)
    correct_msk = np.sum(preds * labels, 1) == 1
    sure = np.sum(labels[sure_msk], 0)
    err = np.sum(labels[sure_msk & ~correct_msk], 0)
    if fill_max:
        sure_msk = cnt == 0
        total_sure += np.sum(sure_msk)
        correct_msk = max_classes == np.argmax(labels, 1)
        sure += np.sum(labels[sure_msk], 0)
        err += np.sum(labels[sure_msk & ~correct_msk], 0)
    if isinstance(class_weights, bool):
        if class_weights:
            class_weights = np.asarray(
                np.unique(labels, return_counts=True)[1], dtype=np.float64
                ) * K / float(N)
        else:
            class_weights = None
    elif class_weights is not None:
        class_weights = np.asarray(class_weights)
    return __loss_class_specific_complete_helper(
        err, sure, risk, class_weights, total_sure, N, la=1, lc=lk, lcs=0)


def search_full_class_specific_py(mo, rnkscores_or_maxclasses, scores_idx, labels, alphas, class_weights, ps, d, *,
                                  lk=1e4, fill_max=False):
    # d is dimesion to search/descend
    N,K = mo.shape
    ps = ps.copy()
    if len(rnkscores_or_maxclasses.shape) == 2:
        ts = [(rnkscores_or_maxclasses[ps[ki], ki]) for ki in range(K)]
        max_classes = np.argmax(mo, 1)
    else:
        ts = [ps[ki] for ki in range(K)]
        max_classes = rnkscores_or_maxclasses

    preds = _thresholding_py(ts, mo)
    preds[:, d] = 1
    cnt = preds.sum(1)
    labels_onehot = one_hot_np(labels, K)
    cnt1_msk = cnt == 1
    correct_msk = cnt1_msk & (np.sum(preds * labels_onehot, 1) == 1)
    total_sure = np.sum(cnt1_msk)
    sure = np.sum(labels_onehot[cnt1_msk], 0)
    err = np.sum(labels_onehot[cnt1_msk & ~correct_msk], 0)
    if fill_max:
        cnt0_msk = cnt == 0
        cnt0_correct_msk = max_classes == labels
        total_sure += np.sum(cnt0_msk)
        sure += np.sum(labels_onehot[cnt0_msk], 0)
        err += np.sum(labels_onehot[cnt0_msk & ~cnt0_correct_msk], 0)
    _preds = np.ones(N, dtype=int) * K
    _preds[cnt1_msk] = np.argmax(preds[cnt1_msk], 1)
    cnt2_msk = np.asarray((cnt == 2) & (preds[:, d]), dtype=bool) #second mask is trivial
    preds[:, d] = 0 #this order cannot chance
    _preds[cnt2_msk] = np.argmax(preds[cnt2_msk], 1)

    best_i, best_loss = -1, np.inf
    for ijjj in range(N-1):
        i = scores_idx[ijjj, d]
        yi = labels[i]
        tint = _preds[i]
        if cnt[i] == 1 or cnt[i] == 2:
            total_sure += __update_cnts(tint, yi, err, sure, 1 if cnt[i] == 2 else -1)
            if fill_max and cnt[i] == 1:
                total_sure += __update_cnts(max_classes[i], yi, err, sure, 1)
        cnt[i] -= 1
        curr_loss = __loss_class_specific_complete_helper(
            err, sure, alphas, class_weights, total_sure, N, la=1, lc=lk, lcs=0)
        if curr_loss < best_loss:
            best_i, best_loss = ijjj, curr_loss
    return best_i, best_loss


def search_full_overall_py(mo, rnkscores_or_maxclasses, scores_idx, labels, alpha, ps, d, *,
                           lk=1e4, fill_max=False):
    N,K = mo.shape
    ps = ps.copy()
    if len(rnkscores_or_maxclasses.shape) == 2:
        ts = [(rnkscores_or_maxclasses[ps[ki], ki]) for ki in range(K)]
        max_classes = np.argmax(mo, 1)
    else:
        ts = [ps[ki] for ki in range(K)]
        max_classes = rnkscores_or_maxclasses
    preds = _thresholding_py(ts, mo)
    preds[:, d] = 1
    cnt = preds.sum(1)

    #initialize mems for ijjj=0
    total_err, total_sure = 0, 0
    for ii in range(N):
        yii = labels[ii]
        if cnt[ii] == 1:
            total_sure += 1
            total_err += 1 - preds[ii, yii]
        if cnt[ii] == 0 and fill_max:
            total_sure += 1
            if max_classes[ii] != yii:
                total_err += 1
    best_i, best_loss = -1, np.inf

    for ijjj in range(N-1):
        i = scores_idx[ijjj, d]
        yi = labels[i]
        #preds[i, d] will change from 1 to 0
        if cnt[i] == 2:#unsure -> sure
            if d == yi:#we miss this item in this case, so error increases by 1
                total_err += 1
            elif mo[i, yi] <= ts[yi]: #we miss the item
                total_err += 1
            total_sure += 1
        elif cnt[i] == 1: #Sure -> unsure. Also, this cnt has to be class d
            if d != yi:
                total_err -= 1
            total_sure -= 1
            if fill_max:
                total_sure += 1
                if max_classes[i] != yi:
                    total_err += 1
        else: #Do nothing as we have an unsure -> unsure case
            pass
        cnt[i] -= 1

        curr_loss = __loss_overall_helper(total_err, total_sure, alpha, N, la=1, lc=lk, lcs=0)

        if curr_loss < best_loss:
            best_i, best_loss = ijjj, curr_loss
    return best_i, best_loss

def coord_desc_classspecific_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps, alphas, *,
                                          class_weights=False, lk=1e4, fill_max=False):
    N, K = mo.shape
    best_loss, ps = np.inf, ps.copy()

    keep_going = True
    if isinstance(class_weights, bool):
        if class_weights:
            weights = np.sum(one_hot_np(labels, K), 0) / float(N) * K
        else:
            weights = None
    else:
        weights = class_weights

    while keep_going:
        curr_ps = np.zeros(K)
        best_ki, curr_loss = None, best_loss
        for ki in range(K):
            curr_ps[ki], temp_loss = search_full_class_specific_py(
                mo, rnkscores_or_maxclasses, scores_idx, labels, alphas, weights, ps, ki,
                lk=lk, fill_max=fill_max)
            if temp_loss < curr_loss:
                best_ki, curr_loss = ki, temp_loss
        if curr_loss < best_loss:
            ps[best_ki] = curr_ps[best_ki]
            best_loss = curr_loss
        else:
            keep_going = False
    return best_loss, ps, None


def coord_desc_overall_py(mo, rnkscores_or_maxclasses, scores_idx, labels, ps, alpha, *,
                                   lk=1e4, fill_max=False):
    N, K = mo.shape
    best_loss, ps = np.inf, ps.copy()

    keep_going = True

    while keep_going:
        curr_ps = np.zeros(K)
        best_ki, curr_loss = None, best_loss
        for ki in range(K):
            curr_ps[ki], temp_loss = search_full_overall_py(
                mo, rnkscores_or_maxclasses, scores_idx, labels, alpha, ps, ki,
                lk=lk, fill_max=fill_max)
            if temp_loss < curr_loss:
                best_ki, curr_loss = ki, temp_loss
        if curr_loss < best_loss:
            ps[best_ki] = curr_ps[best_ki]
            best_loss = curr_loss
        else:
            keep_going = False
    return best_loss, ps, None

def loss_overall(idx2rnk, rnk2idx, labels, max_classes, ps, r, *,
                 lk=1e4, fill_max=False):
    if not _CYTHON_ENABLED:
        preds = np.asarray(idx2rnk > ps, np.int32)
        return loss_overall_py(
            preds, one_hot_np(labels, idx2rnk.shape[1]), max_classes, r,
            lk=lk, fill_max=fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int32)
    rnk2idx = np.asarray(rnk2idx, np.int32)
    labels = np.asarray(labels, np.int32)
    max_classes = np.asarray(max_classes, np.int32)
    ps = np.asarray(ps, np.int32)
    return cdc.loss_overall_c(
        idx2rnk, rnk2idx, labels, max_classes, ps, r,
        la=1, lc=lk, lcs=0, fill_max=fill_max)

def loss_classspecific(idx2rnk, rnk2idx, labels, max_classes, ps, rks, *,
                        class_weights=None, lk=1e4, fill_max=False):
    if not _CYTHON_ENABLED:
        preds = np.asarray(idx2rnk > ps, np.int32)
        return loss_class_specific_py(
            preds, one_hot_np(labels, idx2rnk.shape[1]), max_classes, rks,
            class_weights=class_weights, lk=lk, fill_max=fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int32)
    rnk2idx = np.asarray(rnk2idx, np.int32)
    labels = np.asarray(labels, np.int32)
    max_classes = np.asarray(max_classes, np.int32)
    ps = np.asarray(ps, np.int32)
    if class_weights is not None:
        class_weights = np.asarray(class_weights, np.float64)
    if rks is not None:
        rks = np.asarray(rks, np.float64)
    return cdc.loss_class_specific_c(
        idx2rnk, rnk2idx, labels, max_classes, ps, rks,
        class_weights, la=1, lc=lk, lcs=0, fill_max=fill_max)

def coord_desc_overall(idx2rnk, rnk2idx, labels, max_classes, init_ps, r, *,
                               max_step_size=None, lk=1e4, fill_max=False):
    if not _CYTHON_ENABLED:
        assert max_step_size is None
        return coord_desc_overall_py(
            idx2rnk, max_classes, rnk2idx, labels, init_ps, r,
            lk=lk, fill_max=fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int32)
    rnk2idx = np.asarray(rnk2idx, np.int32)
    labels = np.asarray(labels, np.int32)
    max_classes = np.asarray(max_classes, np.int32)
    init_ps = np.asarray(init_ps, np.int32)
    return cdc.coord_desc_overall_c(
        idx2rnk, rnk2idx, labels, max_classes, init_ps, r,
        max_step_size, la=1, lc=lk, lcs=0, fill_max=fill_max)


def coord_desc_classspecific(idx2rnk, rnk2idx, labels, max_classes, init_ps, rks, *,
                                      class_weights=None,
                                      max_step_size=None, lk=1e4, fill_max=False):
    if not _CYTHON_ENABLED:
        assert max_step_size is None
        return coord_desc_classspecific_py(
            idx2rnk, max_classes, rnk2idx, labels, init_ps, rks,
            class_weights=class_weights, lk=lk, fill_max=fill_max)
    idx2rnk = np.asarray(idx2rnk, np.int32)
    rnk2idx = np.asarray(rnk2idx, np.int32)
    labels = np.asarray(labels, np.int32)
    max_classes = np.asarray(max_classes, np.int32)
    init_ps = np.asarray(init_ps, np.int32)
    if class_weights is not None:
        class_weights = np.asarray(class_weights, np.float64)
    if rks is not None:
        rks = np.asarray(rks, np.float64)
    return cdc.coord_desc_classspecific_c(
        idx2rnk, rnk2idx, labels, max_classes, init_ps, rks,
        class_weights, max_step_size, la=1, lc=lk, lcs=0, fill_max=fill_max)
