"""
Dirichlet Calibration.

Implementation based on https://github.com/dirichletcal/dirichlet_python
"""

from typing import Dict

import ipdb
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Subset

from pyhealth.calib.base_classes import PostHocCalibrator
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel


def _get_identity_weights(n_classes, method='Full'):

    raw_weights = None

    if (method is None) or (method == 'Full'):
        raw_weights = torch.zeros((n_classes, n_classes + 1)) + \
                      torch.hstack([torch.eye(n_classes), torch.zeros((n_classes, 1))])
        raw_weights = raw_weights.ravel()
    return raw_weights.ravel()
    return torch.tensor(raw_weights.ravel(), requires_grad=True)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - torch.max(X, axis=1).values.reshape(-1, 1)
    exps = torch.exp(shiftx)
    return exps / torch.sum(exps, axis=1).reshape(-1, 1)

def _calculate_outputs(weights, X):
    #mul = torch.dot(X, weights.transpose())
    mul = torch.matmul(X, weights.permute(1,0))
    return _softmax(mul)

def _get_weights(params, k, ref_row, method):
        ''' Reshapes the given params (weights) into the full matrix including 0
        '''

        if method in ['Full', None]:
            raw_weights = params.reshape(-1, k+1)
        if ref_row:
            weights = raw_weights - torch.repeat_interleave(
                raw_weights[-1, :].reshape(1, -1), k, dim=0)
        else:
            weights = raw_weights
        return weights

def _forward(params, logits, y, k, method, reg_lambda, reg_mu, ref_row, reg_format):
    device = params.device

    logits = torch.hstack([logits, torch.zeros((logits.shape[0], 1), device=device)])
    weights = _get_weights(params, k, ref_row, method)
    new_logits = torch.matmul(logits, weights.permute(1,0))
    probs = _softmax(new_logits)
    #outputs = _calculate_outputs(weights, logits)
    #outputs = torch.softmax(torch.matmul(X, params),dim=1)
    loss = torch.mean(-torch.log(torch.sum(y * probs, dim=1)))
    if reg_mu is None:
        if reg_format == 'identity':
            reg = torch.hstack([torch.eye(k), torch.zeros((k, 1))])
        else:
            reg = torch.zeros((k, k+1))
        loss = loss + reg_lambda * torch.sum((weights - reg.to(device))**2)
    else:
        weights_hat = weights - torch.hstack([weights[:, :-1] * torch.eye(k, device=device),
                                           torch.zeros((k, 1), device=device)])
        loss = loss + reg_lambda * torch.sum(weights_hat[:, :-1] ** 2) + \
            reg_mu * torch.sum(weights_hat[:, -1] ** 2)

    return {'loss': loss, 'logits': new_logits}


class _DirCalModule(torch.nn.Module):
    def __init__(self, num_classes, device) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.weights = torch.tensor(_get_identity_weights(self.num_classes), device=device, requires_grad=True)


    def forward(self, logits):
        device = logits.device
        logits = torch.hstack([logits, torch.zeros((logits.shape[0], 1), device=device)])
        ...

class DirichletCalibration(PostHocCalibrator):
    def __init__(self, model:BaseModel, debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.mode = self.model.mode
        for param in model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.device = model.device
        self.debug = debug

        self.num_classes = None
        self.weights = None

    def _forward(self, logits):
        logits = torch.hstack([logits, torch.zeros((logits.shape[0], 1), device=self.device)])
        weights = _get_weights(self.weights_0_, k=self.num_classes, ref_row=True, method='Full')
        new_logits = torch.matmul(logits, weights.permute(1,0))
        return new_logits, weights


    def calibrate(self, cal_dataset:Subset, lr=0.01, max_iter=128, reg_lambda=1e-3):
        _cal_data = prepare_numpy_dataset(self.model, cal_dataset,
                                          ['y_true', 'logit'], debug=self.debug)
        if self.num_classes is None:
            self.num_classes = _cal_data['logit'].shape[1]

        self.weights_0_ = torch.tensor(_get_identity_weights(self.num_classes), device=self.device, requires_grad=True)
        #self.weights = torch.nn.Linear(self.num_classes, self.num_classes, bias=False).to(self.device)
        #self.weights = torch.eye(self.num_classes).to(self.device)
        optimizer = optim.LBFGS([self.weights_0_], lr=lr, max_iter=max_iter)
        logits = torch.tensor(_cal_data['logit'], dtype=torch.float, device=self.device)
        label = torch.tensor(F.one_hot(torch.tensor(_cal_data['y_true']), num_classes=self.num_classes),
                             dtype=torch.long if self.model.mode == 'multiclass' else torch.float32,
                             device=self.device)


        reg_mu = None
        reg_format = 'identity'
        def _eval():
            optimizer.zero_grad()
            new_logits, weights = self._forward(logits)
            probs = _softmax(new_logits)
            loss = torch.mean(-torch.log(torch.sum(label * probs, dim=1)))
            if reg_mu is None:
                if reg_format == 'identity':
                    reg = torch.hstack([torch.eye(self.num_classes), torch.zeros((self.num_classes, 1))])
                else:
                    reg = torch.zeros((self.num_classes, self.num_classes+1))
                loss = loss + reg_lambda * torch.sum((weights - reg.to(self.device))**2)
            else:
                weights_hat = weights - torch.hstack([weights[:, :-1] * torch.eye(self.num_classes, device=self.device),
                                                torch.zeros((self.num_classes, 1), device=self.device)])
                loss = loss + reg_lambda * torch.sum(weights_hat[:, :-1] ** 2) + \
                    reg_mu * torch.sum(weights_hat[:, -1] ** 2)

            loss.backward()
            return loss
            loss = _forward(self.weights_0_, logits, label,
                              self.num_classes, 'Full', reg_lambda, None, True, 'identity')
            loss['loss'].backward()
            return loss['loss']
        self.train()
        optimizer.step(_eval)
        self.eval()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        ret = self.model(**kwargs)
        ret['logit'] = self._forward(ret['logit'])[0]
        ret['y_prob'] = self.model.prepare_y_prob(ret['logit'])
        criterion = self.model.get_loss_function()
        ret['loss'] = criterion(ret['logit'], ret['y_true'])
        return ret