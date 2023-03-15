#https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html
#https://cython-docs2.readthedocs.io/en/latest/src/tutorial/numpy.html
from libc.stdlib cimport calloc, free

import numpy as np
cimport numpy as np
DTYPE = np.float64
DTYPE_INT = np.int32
_FLAG_FILLMAX = 2 ** 5

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE_int_t

cdef _parse_mode(int mode):
    cdef int fill_max = <int> ((mode & _FLAG_FILLMAX) > 0)
    mode = mode & 0xf
    return fill_max,mode

cdef int _update_counts(int pred, int truth, int* err, int* sure, int inc):
    if pred != truth:
        err[truth] += inc
    sure[truth] += inc
    return inc
#============================Compute

cdef double loss_overall_helper__(int total_err, int total_sure, double alpha, int N,
                                 double la, double lc, double lcs):
    #if total_sure == 0: return np.inf
    cdef double a_loss = (1. - total_sure / <double> N) ** 2
    cdef double tempf = total_err / <double> max(total_sure,1) - alpha
    cdef double c_loss = max(tempf, 0.) ** 2
    cdef double cs_loss = tempf ** 2
    return a_loss * la + c_loss * lc + cs_loss * lcs

cdef double loss_class_specific_complete_helper__(int K, int* err, int* sure,
                                                 double* rs, #targets
                                                 double* weights,
                                                 int total_sure, int N,
                                                 double la, double lc, double lcs):
    #If rs is NULL, simply penalize (it's a weighted risk loss)
    #If rs is not NULL, they are the targets..
    cdef double precision_, recall_, f1_, tf, weight_k=1., c_loss=0., cs_loss = 0.
    cdef double a_loss = (1. - total_sure / <double>N) #** 2
    for ki in range(K):
        if sure[ki] == 0: return np.inf #Ignored the case of (sure[ki] == err[ki])
        tf = err[ki] / <double> sure[ki]
        if rs != NULL: tf -= rs[ki]
        if weights != NULL: weight_k = weights[ki]
        c_loss += weight_k * (max(tf, 0.) ** 2)
        cs_loss += weight_k * (tf ** 2)
    return a_loss * la + c_loss * lc + cs_loss * lcs

#=============================Searches
cdef int __full_eval(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk, np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,
                     np.ndarray[DTYPE_int_t, ndim=1] labels, np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                     np.ndarray[DTYPE_int_t, ndim=1] ps,
                     int* err, int* sure,
                     int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, cnt, total_sure = 0, temp_last_pred
    for ni in range(N):
        yi = labels[ni]
        cnt = temp_last_pred = 0
        for ki in range(K):
            if idx2rnk[ni, ki] > ps[ki]:
                cnt += 1
                pred_i = ki
        if cnt == 1:
            total_sure += _update_counts(pred_i, yi, err, sure, 1)
        if cnt == 0 and fill_max:
            total_sure += _update_counts(max_classes[ni], yi, err, sure, 1)
    return total_sure

cdef (int, int, int*) __full_eval_overall(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk, np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,
                                    np.ndarray[DTYPE_int_t, ndim=1] labels, np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                    np.ndarray[DTYPE_int_t, ndim=1] ps,
                                    int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, cnt_i
    cdef int total_err = 0, total_sure = 0
    cdef int* cnt = <int*> calloc(N, sizeof(int))
    for ni in range(N):
        yi = labels[ni]
        cnt_i = 0
        for ki in range(K):
            if idx2rnk[ni, ki] > ps[ki]:
                cnt_i += 1
        if cnt_i == 1:
            total_sure += 1
            if idx2rnk[ni, yi] <= ps[yi]:
                total_err += 1
        elif cnt_i == 0 and fill_max:
            total_sure += 1
            if max_classes[ni] != yi:
                total_err += 1
        cnt[ni] = cnt_i
    return total_err, total_sure, cnt

cdef double loss_class_specific_q__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                    np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                    np.ndarray[DTYPE_int_t, ndim=1] labels,
                                    np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                   #Start of memory stuff
                                   np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                                   #Start of loss function args
                                   double* rs, double* weights,
                                   double la, double lc, double lcs, int fill_max):

    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, cnt, temp_last_pred
    cdef int* err = <int*> calloc(K, sizeof(int))
    cdef int* sure = <int*> calloc(K, sizeof(int))
    cdef int total_sure = __full_eval(idx2rnk, rnk2idx, labels, max_classes, ps, err, sure, fill_max)
    cdef double loss = loss_class_specific_complete_helper__(K, err, sure, rs, weights, total_sure, N, la, lc, lcs)
    free(err); free(sure);
    return loss


cdef double loss_overall_q__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk, #input data args
                            np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,
                            np.ndarray[DTYPE_int_t, ndim=1] labels,
                            np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                            #Start of memory stuff
                            np.ndarray[DTYPE_int_t, ndim=1] ps, #positions
                            #Start of loss function args
                            double alpha,
                            double la, double lc, double lcs, int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int total_err = 0, total_sure = 0
    cdef int* cnt = NULL
    total_err, total_sure, cnt = __full_eval_overall(idx2rnk, rnk2idx, labels, max_classes, ps, fill_max)
    cdef double loss_ = loss_overall_helper__(total_err, total_sure, alpha, N, la, lc, lcs)
    free(cnt)
    return loss_

cdef (int, double) search_full_class_specific_complete__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                                         np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                                         np.ndarray[DTYPE_int_t, ndim=1] labels,
                                                         np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                                         #Start of memory stuff
                                                         np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                                                         int dim, int start_pos, int end_pos,
                                                         #Start of loss function args
                                                         double* rs, double* weights,
                                                         double la, double lc, double lcs, int fill_max):

    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, tempint, tempint2, total_sure = 0, orig_pd = ps[dim]
    if end_pos == -1: end_pos = N - 1 #Update  the actual end_pos. [start_pos, end_pos)
    ps[dim] = start_pos - 1 #inclusive of start_pos, so we first eval the loss for start_pos-1
    cdef int* err = <int*> calloc(K, sizeof(int))
    cdef int* sure = <int*> calloc(K, sizeof(int))
    cdef int* cnt = <int*> calloc(N, sizeof(int))
    cdef int* alt = <int*> calloc(N, sizeof(int))
    #cnt[i] == 1: alt[i] = k means the sure prediction is k
    #cnt[i] == 2: alt[i] = k means this is the prediction other than k=dim, >=K means both are not..
    #otherwise, alt[i] does not mean anything

    for ni in range(N):
        tempint = 0 #tempint counts how many k!=dim in prediction. alt[ni] is the sum of all of them
        yi = labels[ni]
        for ki in range(K):
            if idx2rnk[ni, ki] > ps[ki]:
                cnt[ni] += 1
                if ki != dim:
                    tempint += 1
                    alt[ni] += ki
        if cnt[ni] == 1:
            if tempint == 0: alt[ni] = dim #alt[ni] is now the prediction, yi is truth
            total_sure += _update_counts(alt[ni], yi, err, sure, 1)
        elif cnt[ni] == 2:
            if tempint == 2: alt[ni] = K #Both are not dim, so it stays uncertain regardless of the threshold of dim
        elif cnt[ni] == 0 and fill_max: #Don't worry about this case later, as cnt will only decrease as we search
            total_sure += _update_counts(max_classes[ni], yi, err, sure, 1)

    #iterate through all quantiles
    cdef double curr_loss, best_loss = np.inf
    cdef int best_nj = -1, nj
    for nj in range(start_pos, end_pos):#At nj, we are computing the loss for the case >nj
        ni = rnk2idx[nj, dim]
        yi = labels[ni]
        tempint = alt[ni] #prediction (when we do use this)
        if cnt[ni] == 2: #unsure->sure
            #assert tempint != K and tempint != dim
            total_sure += _update_counts(tempint, yi, err, sure, 1)
        elif cnt[ni] == 1: #sure -> unsure
            total_sure += _update_counts(tempint, yi, err, sure, -1)
            if fill_max:#unsure -> sure, but uses the alternative prediction in max_classes..
                total_sure += _update_counts(max_classes[ni], yi, err, sure, 1)
                #cnt[ni] += 1 #We will use cnt only to count the number of triggered thresholds for clarity
        cnt[ni] -= 1
        curr_loss = loss_class_specific_complete_helper__(K, err, sure, rs, weights, total_sure, N, la, lc, lcs)
        if curr_loss < best_loss:
            best_nj, best_loss = nj, curr_loss
    free(err); free(sure);
    free(cnt); free(alt);
    ps[dim] = orig_pd
    return best_nj, best_loss

cdef (int, double) search_full_overall__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                         np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                         np.ndarray[DTYPE_int_t, ndim=1] labels,
                                         np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                         #Start of memory stuff
                                         np.ndarray[DTYPE_int_t, ndim=1] ps,  #positions
                                         int dim, int start_pos, int end_pos,
                                         #Start of loss function args
                                         double r,
                                         double la, double lc, double lcs, int fill_max):

    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef int ni, ki, yi, tempint, orig_pd = ps[dim]
    if end_pos == -1: end_pos = N - 1 #Update  the actual end_post. [start_pos, end_pos)
    ps[dim] = start_pos - 1 #inclusive of start_pos, so we first eval the loss for start_pos-1
    cdef int* cnt = NULL
    cdef int total_err = 0, total_sure = 0
    total_err, total_sure, cnt = __full_eval_overall(idx2rnk, rnk2idx, labels, max_classes, ps, fill_max)

    cdef double curr_loss, best_loss = np.inf
    cdef int best_nj = -1, nj
    for nj in range(start_pos, end_pos): #At nj, we are computing the loss for the case >nj
        ni = rnk2idx[nj, dim]
        yi = labels[ni]
        if cnt[ni] == 2: #unsure->sure
            if dim == yi: #We miss this class dim item
                total_err += 1
            elif idx2rnk[ni, yi] <= ps[yi]: #We originally missed this but now this becomes sure
                total_err += 1
            total_sure += 1
        elif cnt[ni] == 1: #sure -> unsure
            if dim != yi: total_err -= 1 #only in this case was there an error before this change
            total_sure -= 1
            if fill_max:
                total_sure += 1
                if max_classes[ni] != yi: total_err += 1
        cnt[ni] -= 1
        curr_loss = loss_overall_helper__(total_err, total_sure, r, N, la, lc, lcs)
        if curr_loss < best_loss:
            best_nj, best_loss = nj, curr_loss

    free(cnt)
    ps[dim] = orig_pd
    return best_nj, best_loss

cdef (int, double) main_coord_descent_class_specific__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                                np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                                np.ndarray[DTYPE_int_t, ndim=1] labels,
                                                np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                                np.ndarray[DTYPE_int_t, ndim=1] init_ps, #initial positions
                                                int* mod_ps, #Store ps here
                                                double* best_loss_ptr,
                                                double* rs, double* class_weights, int max_step,
                                                double la, double lc, double lcs, int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef double best_loss = np.inf, curr_loss, tempf
    cdef int keep_going = 1, ki, pi, curr_best_ki, start_pos, end_pos, max_step_temp=-1, n_searches = 0
    cdef np.ndarray[DTYPE_int_t, ndim=1] ps = init_ps.copy()
    cdef np.ndarray[DTYPE_int_t, ndim=1] curr_p = np.empty([K], dtype=DTYPE_INT)
    while keep_going == 1:
        n_searches += 1
        curr_loss = best_loss
        for ki in range(K):
            if max_step > 0:
                start_pos, end_pos = max(ps[ki]-max_step, 0), min(ps[ki]+max_step, N-1)
            else:
                start_pos, end_pos = 0, -1
            #print(ps[0], ps[1], ps[2]," | ", ki, start_pos, end_pos, " | ", rs[0], rs[1], rs[2])
            curr_p[ki], tempf = search_full_class_specific_complete__(idx2rnk, rnk2idx, labels, max_classes, ps, ki, start_pos, end_pos, rs, class_weights, la, lc, lcs, fill_max)
            if tempf < curr_loss:
                curr_loss, curr_best_ki = tempf, ki
        if curr_loss < best_loss:
            ps[curr_best_ki] = curr_p[curr_best_ki]
            best_loss = curr_loss
            if max_step_temp > 0: #Switch back as we used this to move further in this round
                max_step, max_step_temp = max_step_temp, -1
        else:
            if max_step_temp == -1 and max_step > 0: #We could try moving further
                max_step_temp, max_step = max_step, -1
            else:
                keep_going = 0

    for ki in range(K):
        mod_ps[ki] = ps[ki]
    best_loss_ptr[0] = best_loss
    return (n_searches, best_loss)

cdef (int, double) main_coord_descent_overall__(np.ndarray[DTYPE_int_t, ndim=2] idx2rnk,  #ranks[ni, ki] is the ranking of ni-th data's score ki in all (small is low)
                                         np.ndarray[DTYPE_int_t, ndim=2] rnk2idx,  #idx2ranks[ranks2idx[j,k], k] = j
                                         np.ndarray[DTYPE_int_t, ndim=1] labels,
                                         np.ndarray[DTYPE_int_t, ndim=1] max_classes,
                                         np.ndarray[DTYPE_int_t, ndim=1] init_ps,  #initial positions
                                         int* mod_ps,  #Store ps here
                                         double* best_loss_ptr,
                                         double r, int max_step,
                                         double la, double lc, double lcs, int fill_max):
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    cdef double best_loss = np.inf, curr_loss, tempf
    cdef int keep_going = 1, ki, pi, curr_best_ki, start_pos, end_pos, max_step_temp=-1, n_searches = 0
    cdef np.ndarray[DTYPE_int_t, ndim=1] ps = init_ps.copy()
    cdef np.ndarray[DTYPE_int_t, ndim=1] curr_p = np.empty([K], dtype=DTYPE_INT)
    while keep_going == 1:
        n_searches += 1
        curr_loss = best_loss
        for ki in range(K):
            if max_step > 0:
                start_pos, end_pos = max(ps[ki]-max_step, 0), min(ps[ki]+max_step, N-1)
            else:
                start_pos, end_pos = 0, -1
            curr_p[ki], tempf = search_full_overall__(idx2rnk, rnk2idx, labels, max_classes, ps, ki, start_pos, end_pos, r, la, lc, lcs, fill_max)
            if tempf < curr_loss:
                curr_loss, curr_best_ki = tempf, ki
        if curr_loss < best_loss:
            ps[curr_best_ki] = curr_p[curr_best_ki]
            best_loss = curr_loss
            if max_step_temp > 0: #Switch back as we used this to move further in this round
                max_step, max_step_temp = max_step_temp, -1
        else:
            if max_step_temp == -1 and max_step > 0: #We could try moving further
                max_step_temp, max_step = max_step, -1
            else:
                keep_going = 0
    for ki in range(K):
        mod_ps[ki] = ps[ki]
    best_loss_ptr[0] = best_loss
    return (n_searches, best_loss)

#=============================Python Interfaces
#https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python

cpdef double loss_overall_c(idx2rnk, rnk2idx, labels, max_classes, ps, alpha, la=1, lc=1e4, lcs=0, fill_max=False):
    #Can't do np.asarray as it creates memory leak on Windows (not on Ubuntu for some reason...)
    cdef loss_ = loss_overall_q__(idx2rnk, rnk2idx, labels, max_classes, ps,
                                  float(alpha), float(la), float(lc), float(lcs), int(fill_max))
    return loss_

cpdef double loss_class_specific_c(idx2rnk, rnk2idx, labels, max_classes, ps, rks, class_weights=None,
                             la=1, lc=1e4, lcs=0, fill_max=False):
    #Checks
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    assert rks is None or idx2rnk.shape[1] == rks.shape[0], "rks issue"
    assert idx2rnk.shape[0] == labels.shape[0], "label"
    assert not isinstance(class_weights, bool), "class_weights cannot be bool now."


    #clean classes
    cdef double[::1] weights_
    cdef double* weights_ptr = NULL
    if class_weights is not None:
        weights_ = class_weights
        weights_ptr = &weights_[0]

    #clean risks
    cdef double[::1] rks_
    cdef double* rks_ptr = NULL
    if rks is not None:
        rks_ = rks
        rks_ptr = &rks_[0]

    cdef double loss_ = loss_class_specific_q__(idx2rnk, rnk2idx, labels, max_classes, ps,
                                                rks_ptr, weights_ptr, float(la), float(lc), float(lcs), int(fill_max))
    return loss_


def search_full_class_specific(idx2rnk, rnk2idx, labels, max_classes, ps, k, rks, la=1, lc=1e4, lcs=0, fill_max=False):
    #clean risks
    cdef double[::1] rks_
    cdef double* rks_ptr = NULL
    if rks is not None:
        rks_ = rks
        rks_ptr = &rks_[0]

    return search_full_class_specific_complete__(idx2rnk, rnk2idx, labels, max_classes, ps,
                                                 k, 0, -1, rks_ptr, NULL, la, lc, lcs, int(fill_max))


def search_full_overall(idx2rnk, rnk2idx, labels, max_classes, ps, k, r, la=1, lc=1e4, lcs=0, fill_max=False):
    return search_full_overall__(idx2rnk, rnk2idx, labels, max_classes, ps,
                                 k, 0, -1, r, la, lc, lcs, int(fill_max))


cpdef coord_desc_classspecific_c(idx2rnk, rnk2idx, labels, max_classes, init_ps, rks,
                                         class_weights=None, max_step=None, la=1, lc=1e4, lcs=0,
                                         fill_max=False):
    #Checks
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    assert rks is None or idx2rnk.shape[1] == rks.shape[0]
    assert N == rnk2idx.shape[0] == labels.shape[0] == max_classes.shape[0]
    assert len(init_ps.shape) == 1, "init_ps"
    assert not isinstance(class_weights, bool), "class_weights cannot be bool now."
    cdef int[::1] new_ps = np.asarray(init_ps.copy(), np.int32)

    #clean classes
    cdef double[::1] weights_
    cdef double* weights_ptr = NULL
    if class_weights is not None:
        weights_ = class_weights
        weights_ptr = &weights_[0]

    #clean risks
    cdef double[::1] rks_
    cdef double* rks_ptr = NULL
    if rks is not None:
        rks_ = rks
        rks_ptr = &rks_[0]

    cdef double best_loss
    cdef int n_searches
    n_searches , best_loss = main_coord_descent_class_specific__(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                                                &new_ps[0], &best_loss, rks_ptr, weights_ptr, max_step or -1,
                                                                float(la), float(lc), float(lcs), int(fill_max))
    return best_loss, np.asarray(new_ps, dtype=np.int32), n_searches



cpdef coord_desc_overall_c(idx2rnk, rnk2idx, labels, max_classes, init_ps, r,
                                  max_step=None, la=1, lc=1e4, lcs=0,
                                  fill_max=False):
    #Checks
    cdef int N = idx2rnk.shape[0], K = idx2rnk.shape[1]
    assert N == rnk2idx.shape[0] == labels.shape[0]
    assert K == rnk2idx.shape[1] == init_ps.shape[0]
    cdef int[::1] new_ps = np.asarray(init_ps.copy(), np.int32)

    cdef double best_loss
    cdef int n_searches
    n_searches , best_loss = main_coord_descent_overall__(idx2rnk, rnk2idx, labels, max_classes, init_ps,
                                                          &new_ps[0], &best_loss, float(r), max_step or -1,
                                                          float(la), float(lc), float(lcs), int(fill_max))
    return best_loss, np.asarray(new_ps, dtype=np.int32), n_searches


