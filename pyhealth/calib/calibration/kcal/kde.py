from typing import Union

import torch

from pyhealth.calib.utils import LogLoss

_MAX_KERNEL_VALUE = 1.


class RBFKernelMean(torch.nn.Module):
    def __init__(self, h=1.):
        super().__init__()
        self.h = h

    def forward(self, x, x1=None):
        #if x1 is None: return 1
        _, dim = x.shape
        if x1 is None:
            x1 = x
        d = torch.pow(torch.cdist(x,x1,2), 2) / (-dim * self.h)
        return torch.exp(d)
    def set_bandwidth(self, h: Union[int, float]):
        """Set the bandwidth"""
        self.h = h
    def get_bandwidth(self):
        return self.h

def KDE_classification(X: torch.Tensor, Y: torch.Tensor, kern:RBFKernelMean=None, X_pred: torch.Tensor=None,
                       weights: Union[torch.Tensor, float, int]=1., min_eps:float=1e-10, drop_max:bool=False):
    """KDE classifier.
    We will be using X and Y to estimate the density for each of the K classes.
    kern is the kernel. X_pred is the data to predict.

    Args:
        X (torch.Tensor): Data of shape (N, d)
        Y (torch.Tensor): One-hot label of shape (N, K)
        kern (RBFKernelMean, optional): kernel.
            Defaults to None (a RBFKernelMean of bandwidth=1).
        X_pred (torch.Tensor, optional): Data to predict.
            Defaults to None, which means we will perform leave-one-out prediction.
        weights (Union[torch.Tensor, float, int], optional): Weights on each data in X.
            Defaults to 1.
        drop_max (bool, optional): Whether to ignore x where K(x_0, x) = 1.
            This typically means x is just x_0.
            Defaults to False.
            If you know there are overlap between X and X_pred, you could
            set this to True for convenience.
            Note that if X_pred=None (LOO) this is automatically set to True.

    Returns:
        torch.Tensor: Probability predictions for X_pred.
    """
    if kern is None:
        kern = RBFKernelMean(h=1.)
    #Y is a one-hot representation
    if X_pred is None:
        Kijs = kern(X, X)
        drop_max = True
    else:
        if len(X_pred.shape) == 1:
            X_pred = X_pred.unsqueeze(0)
        Kijs = kern(X_pred, X)
    if drop_max:
        Kijs = torch.where(
                Kijs < _MAX_KERNEL_VALUE-min_eps,
                Kijs,
                torch.zeros((), device=Kijs.device, dtype=Kijs.dtype)
                )
    Kijs = Kijs * weights #Kijs[:, j] *= weights[j]
    Kijs = Kijs / torch.sum(Kijs, 1, keepdim=True).clip(min_eps) #K[i,j] = K(x[i], self.X[j])
    pred = torch.matmul(Kijs, Y)
    return pred

def batched_KDE_classification(X: torch.Tensor, Y: torch.Tensor, kern=None, X_pred: torch.Tensor=None,
                       weights: Union[torch.Tensor, float, int]=1., min_eps=1e-10):
    pred = []
    batch_size=32
    drop_max = False
    if X_pred is None:
        drop_max = True
        X_pred = X
    with torch.no_grad():
        for st in range(0, len(X_pred), batch_size):
            ed = min(len(X_pred), st+batch_size)
            pred.append(KDE_classification(
                X, Y, kern, X_pred[st:ed], weights, min_eps=min_eps, drop_max=drop_max))
        return torch.concat(pred)

class KDECrossEntropyLoss(torch.nn.Module):
    reduction: str
    def __init__(self, ignore_index: int = -100,
                 reduction: str = 'mean', h: float = 1.0, nclass=None) -> None:
        super(KDECrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.kern = RBFKernelMean(h=h)
        self.log_loss = LogLoss(reduction=reduction)
        self.nclass = nclass

    def forward(self, input: dict, target: torch.Tensor, eval_only=False) -> torch.Tensor:
        weights = input['weights'] if 'weights' in input else 1.
        nclass = self.nclass or max(input['supp_target'].max(), target.max()).item() + 1
        supp_Y = torch.nn.functional.one_hot(input['supp_target'], nclass).float()
        if eval_only:
            pred = batched_KDE_classification(
                input['supp_embed'], supp_Y, self.kern, weights=weights, X_pred=input['pred_embed'])
        else:
            pred = KDE_classification(
                input['supp_embed'], supp_Y, self.kern, weights=weights, X_pred=input['pred_embed'])
        ret = {}
        ret['loss'] = self.log_loss(pred, target)
        ret['extra_output'] = {"prediction": pred}
        return ret
