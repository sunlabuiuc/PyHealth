from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from torch import Tensor

from pyhealth.datasets import utils as datautils
import logging

def agg_loss(loss:torch.Tensor, reduction: str):
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def one_hot_np(labels, K):
    new_labels = np.zeros((len(labels), K))
    new_labels[np.arange(len(labels)), labels] = 1
    return new_labels

class LogLoss(torch.nn.Module):
    """Cross entropy, but takes in the probability instead of the logits"""
    reduction: str
    def __init__(self, weight: Optional[Tensor] = None,  ignore_index: int = -100, reduction: str = 'mean', clip=1e-10) -> None:
        super(LogLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.clip = clip

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimensions (got {dim})"
        input = input.clip(self.clip)#this weight should be trivial, so I won't normalize
        input = -torch.log(input)
        if self.weight is not None:
            input = input * self.weight.unsqueeze(0)
        loss = torch.gather(input, -1, target.unsqueeze(-1)).squeeze(-1)
        return agg_loss(loss, self.reduction)


def prepare_numpy_dataset(model, dataset, keys, forward_kwargs=None,
                         incl_data_keys=None, debug=False, batch_size=32):
    if forward_kwargs is None:
        forward_kwargs = {}
    if incl_data_keys is None:
        incl_data_keys = []
    loader = datautils.get_dataloader(dataset, batch_size, shuffle=False)

    ret = defaultdict(list)
    with torch.no_grad():
        for _i, data in tqdm.tqdm(enumerate(loader), desc=f"retrieving {keys}", total=len(loader)):
            if debug and _i % 10 != 0:
                continue
            data.update(forward_kwargs)
            res = model(**data)
            for key in keys:
                ret[key].append(res[key].detach().cpu().numpy())
            for key in incl_data_keys:
                ret[key].extend(data[key])
    for key in incl_data_keys:
        ret[key] = np.asarray(ret[key])
    for key in keys:
        ret[key] = np.concatenate(ret[key])
    return ret

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import LeaveOneOut


def get_distribution(dist_values):
    logging.info("Computing distributions .. ")
    params = {'bandwidth': 10 ** np.linspace(-1, 1, 5)}
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=LeaveOneOut())
    grid.fit(dist_values)
    kde = KernelDensity(kernel='gaussian', bandwidth=grid.best_params_['bandwidth'])
    print("bandwidth ", grid.best_params_['bandwidth'])
    kde.fit(dist_values)
    return kde

def compute_weights_for_codrug(
    calibrate_dist_values, test_dist_values, density_scheme=None
):
    calib_density = get_distribution(calibrate_dist_values)
    test_density = get_distribution(test_dist_values)
    test_probs = test_density.score_samples(calibrate_dist_values)
    calib_probs = calib_density.score_samples(calibrate_dist_values)
    weights = np.exp(test_probs)/np.exp(calib_probs)
    weights = weights/np.sum(weights)
    
    return weights
