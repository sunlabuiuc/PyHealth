from collections import deque
from importlib import reload

import numpy as np
import pandas as pd
import tqdm
#from data_utils import INTEGER_SAFE_DELTA
from scipy.special import expit

from pyhealth.calib.predictionset.favmac import quantiletree


class FavMac:
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None, C_max=1.) -> None:
        self.target_cost = target_cost
        self.delta = delta

        self.cost_fn = cost_fn # (S, Y)
        self.util_fn = util_fn # (S, Y=None, pred=None)
        self.proxy_fn = proxy_fn # (S, pred)

        #Threshold
        self.t = None

        # reserved parameters to avoid repeating queries
        self._thresholds_mem = {}
        self._cnt = 0

        self.quantiletree = quantiletree.QuantileTree()
        self._queue = deque()
        self.C_max = C_max

    def _add_sample(self, predset, extra_info):
        costs, proxies = extra_info
        _sidx = np.argsort(proxies)
        costs = np.asarray(costs)[_sidx]
        proxies = np.asarray(proxies)[_sidx]
        assert max(costs) <= 1
        if self.delta is not None:
            costs = pd.Series(costs).cummax().values
            _valid_ts = [score for cost, score in zip(costs, proxies) if cost > self.target_cost] # more like invalid ts
            t_k_i = min(_valid_ts) if len(_valid_ts) > 0 else np.inf
            self.quantiletree.insert(t_k_i, 1)
            self._queue.append(t_k_i)
        else:
            curr_cost = 0
            for cost, score in zip(costs, proxies):
                if cost > curr_cost:
                    self.quantiletree.insert(score, cost - curr_cost)
                    curr_cost = cost
            self._queue.append((costs, proxies))

    def _query_threshold(self):
        n = len(self._queue)
        if self.delta is None:
            cutoff = self.target_cost * (n+1) - self.C_max
            return self.quantiletree.query_cumu_weight(cutoff, prev=False)
        else:
            cutoff = self.delta * (n+1) - 1# We should assume a violation for the next point? Should we minus 1??
            return self.quantiletree.query_cumu_weight(cutoff, prev=False)

    def _greedy_sequence(self, pred:np.ndarray):
        raise NotImplementedError()

    def _forward(self, logit, label=None):
        # return predset, (costs, cost_proxies)
        # costs[j] or cost_proxies[j] is for S_j
        # (S_0 \subset S_1 \ldots S_K)
        pred = expit(logit)
        Ss, proxies = self._greedy_sequence(pred)
        costs, predset = None, None
        if label is not None:
            costs = [self.cost_fn(S, label) for S in Ss]
        if self.t is not None:
            candidates = [S for S,v in zip(Ss, proxies) if v < self.t]
            predset = Ss[0] if len(candidates) == 0 else candidates[-1]
        return predset, (costs, proxies)


    def query_threshold(self, target_cost=None):
        if target_cost is not None:
            old_target_cost = self.target_cost
            self.target_cost = target_cost
            t = self._query_threshold()
            self.target_cost = old_target_cost
            return t
        if self._cnt not in self._thresholds_mem:
            self._thresholds_mem[self._cnt] = self._query_threshold()
        return self._thresholds_mem[self._cnt]

    def update(self, logit, label):
        predset, extra_info = self._forward(logit, label)
        self._add_sample(predset, extra_info)
        self._cnt += 1
        self.t = self.query_threshold()
        return predset, extra_info

    def init_calibrate(self, logits, labels):
        n = len(logits)
        for i, (_logit, _y) in tqdm.tqdm(enumerate(zip(logits, labels)), desc='initial calibration...', total=n):
            predset, extra_info = self._forward(_logit, _y)
            self._add_sample(predset, extra_info)
            self._cnt += 1
        self.t = self.query_threshold()

    def __call__(self, logit, label=None, update=True):
        if update and label is not None:
            return self.update(logit, label)
        return self._forward(logit, label)

class FavMac_GreedyRatio(FavMac):
    #In each step, maximize dValue/dProxy.
    def _greedy_sequence(self, pred:np.ndarray):
        proxy_fn = lambda _S: self.proxy_fn(_S, pred=pred, target_cost = None if self.delta is None else self.target_cost)
        try:
            if self.proxy_fn.is_additive():
                Ss, _ = self.util_fn.greedy_maximize_seq(pred=pred, d_proxy = self.proxy_fn.values * (1-pred))
                return Ss, list(map(proxy_fn, Ss))
        except:
            pass

        Ss = [np.zeros(len(pred), dtype=int)]
        proxies = [proxy_fn(Ss[0])]
        while Ss[-1].min() == 0:
            S = Ss[-1].copy()
            curr_d_proxies = [np.nan] * len(S)
            for k in range(len(S)):
                if S[k] == 1: continue
                S[k] = 1
                curr_d_proxies[k] = proxy_fn(S) - proxies[-1]
                S[k] = 0
            k, du_div_dp = self.util_fn.greedy_maximize(S, pred=pred, d_proxy = np.asarray(curr_d_proxies))
            if k is None: break
            S[k] = 1
            Ss.append(S)
            proxies.append(curr_d_proxies[k] + proxies[-1])
        return Ss, proxies