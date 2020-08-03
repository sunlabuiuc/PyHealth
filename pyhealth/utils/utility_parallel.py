# -*- coding: utf-8 -*-
"""A set of utility functions to support parallel computation.
"""

# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause
import numpy as np
from joblib import effective_n_jobs

import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# This elegent solution is from:
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution


def partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=np.int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    xdiff = [starts[n] - starts[n - 1] for n in range(1, len(starts))]

    # print("Split among workers default:", starts, xdiff)
    print('Split tasks into', n_jobs, 'cores...')
    return n_estimators_per_job.tolist(), [0] + starts.tolist(), n_jobs


def unfold_parallel(lists, n_jobs):
    """Internal function to unfold the results returned from the parallization

    Parameters
    ----------
    lists : list
        The results from the parallelization operations.

    n_jobs : optional (default=1)
        The number of jobs to run in parallel for both `fit` and
        `predict`. If -1, then the number of jobs is set to the
        number of cores.

    Returns
    -------
    result_list : list
        The list of unfolded result.
    """
    full_list = []
    for i in range(n_jobs):
        full_list.extend(lists[i])
    return full_list
