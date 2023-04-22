"""Bandwidth selection."""
import numpy as np
import torch
import tqdm
from sklearn.model_selection import GroupKFold, KFold

from pyhealth.calib.utils import LogLoss

from .kde import KDE_classification, RBFKernelMean


class GoldenSectionBoundedSearch():
    """An implementation of https://en.wikipedia.org/wiki/Golden-section_search
    """
    gr = (1 + (5 ** 0.5)) * 0.5
    def __init__(self, func, lb, ub, tol=1e-4):
        self.func = func
        self.lb = lb
        self.ub = ub
        self.mem = {}
        self.hist = []
        self.tol = tol

        self.round_digit = -int(np.floor(np.log10(tol/2)))
        self._search(lb, ub)


    def eval(self, x):
        """Cached evaluation of the target function.
        """
        assert x >= self.lb and x <= self.ub
        x = np.round(x, self.round_digit)
        v = self.mem.get(x, None)
        if v is None:
            v = self.func(x)
            self.mem[x] = v
            self.hist.append(x)
        return v

    def _search(self, a, b):
        c = b - (b - a) / self.gr
        d = a + (b - a) / self.gr
        steps = int(np.ceil(np.log((b-a)/self.tol)/np.log(self.gr)))
        with tqdm.tqdm(total=steps) as pbar:
            while abs(b - a) > self.tol:
                lc = self.eval(c)
                ld = self.eval(d)
                if lc < ld:
                    b = d
                    ss_repr = f'h={c:5f} Loss:{lc:3f}'
                else:
                    a = c
                    ss_repr = f'h={d:5f} Loss:{ld:3f}'
                c = b - (b - a) / self.gr
                d = a + (b - a) / self.gr
                pbar.update(1)
                pbar.set_description(ss_repr)
        return (b + a) / 2.

    @classmethod
    def search(cls, func, lb, ub, tol=1e-4):
        """Wrapper of the main search function.
        """
        o = GoldenSectionBoundedSearch(func, lb, ub, tol)
        return o._search(lb, ub), o


def fit_bandwidth(X:torch.Tensor, Y:torch.Tensor, group=None, *,
                 kern:RBFKernelMean=None,
                 num_fold=np.inf, lb:float=1e-1, ub:float=1e1, seed=42) -> float:
    """Use k-fold cross-validation to find the best bandwidth.

    Args:
        X (torch.Tensor): embeddings
        Y (torch.Tensor): one-hot target
        groups (List[str,int,...], optional): groups to observe during the cross-validation.
            Defaults to None.
        kern (RBFKernelMean, optional): kernel object.
            Defaults to None.
        num_fold (int, optional): number of folds for cross-validation.
            Defaults to np.inf, which means leave-one-out cross-validation.
        lb (float, optional): lower-bound of the search range. Defaults to 1e-1.
        ub (float, optional): upper-bound of the search range. Defaults to 1e1.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        h (float): the optimal bandwidth found.
    """
    loss_func = LogLoss(reduction='mean')
    if kern is None:
        kern = RBFKernelMean()
    base_h = kern.get_bandwidth()

    num_fold = min(num_fold, len(Y))
    if group is not None:
        num_fold = min(num_fold, len(set(group)))
    print(f"Using {num_fold}-folds")

    def eval_loss(h):
        kern.set_bandwidth(h)
        if group is None:
            kf = KFold(n_splits=num_fold, random_state=seed, shuffle=True)
        else:
            kf = GroupKFold(n_splits=num_fold)
        preds, y_true = [], []
        for _supp, _pred in kf.split(Y, Y, group):
            preds.append(KDE_classification(X[_supp], Y[_supp], kern, X[_pred]))
            y_true.append(Y[_pred].argmax(1))
        return loss_func(torch.cat(preds, 0), torch.cat(y_true, 0)).item()
    h, _ = GoldenSectionBoundedSearch.search(eval_loss, lb * base_h, ub * base_h)
    kern.set_bandwidth(base_h)
    return h
