"""
SCRIB: Set-classifier with Class-specific Risk Bounds

From:
    Lin, Zhen, Lucas Glass, M. Brandon Westover, Cao Xiao, and Jimeng Sun. 
    "SCRIB: Set-classifier with Class-specific Risk Bounds for Blackbox Models." 
    AAAI 2022.

Implementation based on https://github.com/zlin7/scrib

"""

import time
from typing import Dict

import ipdb
import numpy as np
import pandas as pd
import torch
import tqdm

from pyhealth.uq.base_classes import SetPredictor

from . import quicksearch as qs

OVERALL_LOSSFUNC = 'overall'
CLASSPECIFIC_LOSSFUNC = 'classSpec'

def parse_loss_func(loss_func:str):
    loss_kwargs = {}
    loss_name = loss_func.split()[0]
    if len(loss_func.split()) > 1:
        for s in loss_func.split()[1:]:
            loss_kwargs[s.split('=')[0]] = float(s.split('=')[1])
    return loss_name, loss_kwargs

__all__ = ['SCRIB']



class _CoordDescent():
    def __init__(self, model_output, labels, rks,
                 loss_func=OVERALL_LOSSFUNC, loss_kwargs=None,
                 restart_n=1000, restart_range=0.1,
                 init_range=None,
                 fill_max=False, verbose=False):
        self.N, self.K = model_output.shape
        
        # quantities useful for loss eval
        self.loss_name = loss_func
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_kwargs = loss_kwargs
        if self.loss_name == OVERALL_LOSSFUNC:
            assert isinstance(rks, float)
        elif rks is not None:
            rks = np.asarray(rks)
        self.loss_kwargs.update({"fill_max": fill_max})
        self.idx2rnk = np.asarray(pd.DataFrame(model_output).rank(ascending=True), np.int32)
        if np.min(self.idx2rnk) == 1:
            self.idx2rnk -= 1
        self.rnk2idx = np.asarray(np.argsort(model_output, axis=0), np.int32)
        if len(labels.shape) == 2:
            # one-hot -> class indices
            labels = np.argmax(labels, 1)
        self.labels = np.asarray(labels, np.int32)
        self.max_classes = np.argmax(model_output, 1)
        self.rks = rks

        self.model_output = model_output
        self.restart_n = restart_n
        self.restart_range = restart_range
        self.init_range = init_range or (int(np.ceil(self.N / 2)), self.N - 1)
        self.verbose = verbose

    def _search(self, ps):
        _search_fn = {
            CLASSPECIFIC_LOSSFUNC: qs.main_coord_descent_class_specific,
            OVERALL_LOSSFUNC: qs.main_coord_descent_overall
            }[self.loss_name]
        return _search_fn(self.idx2rnk, self.rnk2idx, self.labels, self.max_classes,
                          ps, self.rks, **self.loss_kwargs)
    
    def _loss_eval(self, ps):
        _loss_fn = {
            CLASSPECIFIC_LOSSFUNC: qs.loss_class_specific,
            OVERALL_LOSSFUNC: qs.loss_overall
            }[self.loss_name]
        return _loss_fn(self.idx2rnk, self.rnk2idx, self.labels, self.max_classes,
                        ps, self.rks, **self.loss_kwargs)
        
    def _p2t(self, p):
        # Translate ranks to thresholds
        return [self.model_output[self.rnk2idx[p[k], k], k] for k in range(self.K)]

    def sample_new_loc(self, old_p, restart_range=0.1):
        diff = np.random.uniform(-restart_range, restart_range, self.K)
        new_p = old_p.copy()
        for k in range(self.K):
            new_p[k] = max(min(int(new_p[k] + diff[k] * self.N), self.N-1), 0)
        return new_p

    def search(self, seed=7, get_complexity=False):
        np.random.seed(seed)
        total_searches = 0
        best_ps = np.random.randint(*self.init_range, self.K)

        st = time.time()
        best_loss, best_ps, n_searches = self._search(best_ps)
        total_searches += n_searches
        ed1 = time.time()
        if self.restart_n > 0:
            keep_going = True
            while keep_going:
                keep_going = False
                curr_restart_best_loss, curr_restart_best_ps = np.inf, None

                for _ in range(self.restart_n):
                    # Restart in neighborhood
                    new_ps_ = self.sample_new_loc(best_ps, self.restart_range)
                    loss_ = self._loss_eval(new_ps_)
                    if loss_ < best_loss:
                        if self.verbose: print("Neighborhood has a better loc with loss={} < {} ".format(loss_, best_loss))
                        best_loss, best_ps, n_searches = self._search(new_ps_)
                        total_searches += n_searches
                        keep_going = True
                        break
                    elif loss_ < curr_restart_best_loss:
                        curr_restart_best_loss, curr_restart_best_ps = loss_, new_ps_
                if not keep_going:
                    if self.verbose: print(f"Tried {curr_restart_best_ps} vs {best_ps}, loss:{curr_restart_best_loss} > {best_loss}")
        ed2 = time.time()
        if self.verbose: print(f"{ed1-st:.3f} + {ed2-ed1:.3f} seconds")
        if get_complexity: return self._p2t(best_ps), best_loss, total_searches
        return self._p2t(best_ps), best_loss
    
    @classmethod
    def search_multiple_times(cls, prob:np.ndarray, label:np.ndarray, 
                            rks, B=10, **kwargs):
        # label is not one-hot
        best_loss, best_ts = np.inf, None
        searcher = cls(prob, label, rks, **kwargs)
        for seed in range(B):
            np.random.seed(seed)
            ts, _l = searcher.search(seed+1)
            print(f"{seed}: loss={_l}")
            if _l < best_loss:
                best_loss,best_ts = _l, ts
        return best_ts, best_loss
    




class SCRIB(SetPredictor):
    def __init__(self, model, rks, **kwargs) -> None:
        super().__init__(model, **kwargs)
        raise NotImplementedError()


    def calibrate(self, cal_dataset):
        raise NotImplementedError()
        return super().calibrate(cal_dataset)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
        old_pred = self.model(**kwargs)
        return {
            #'y_prob': pred,
            'y_true': old_pred['y_true'],
            #'loss': self.proj.criterion.log_loss(pred, old_pred['y_true'])
        }
