"""
Temperature Scaling 

From:
    Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. "On calibration of modern neural networks." In International conference on machine learning, pp. 1321-1330. PMLR, 2017

Implementation based on https://github.com/gpleiss/temperature_scaling

"""

from typing import Dict, List, Optional, Tuple

import ipdb
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader

from pyhealth.datasets import utils as datautils
from pyhealth.uq.base_classes import PostHocCalibrator

__all__ = ['TemperatureScaling']

def _prepare_logit_dataset(model, dataset, debug=False, batch_size=32):
    logits = []
    labels = []
    loader = datautils.get_dataloader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        for _i, data in tqdm.tqdm(enumerate(loader), desc="embedding...", total=len(loader)):
            if debug and _i % 10 != 0: continue
            res = model(**data)
            labels.append(res['y_true'].detach().cpu().numpy())
            logits.append(res['logit'].detach().cpu().numpy())
    return {'labels': np.concatenate(labels), 'logits': np.concatenate(logits)}


class TemperatureScaling(PostHocCalibrator):
    def __init__(self, model:torch.nn.Module, debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        if model.mode != 'multiclass':
            raise NotImplementedError()
        self.mode = self.model.mode # multiclass
        for param in model.parameters():
            param.requires_grad = False

        self.model.eval()
        # TODO: Try to get rid of "device"?
        self.device = model.device
        self.debug = debug

        self.num_classes = None
        
        self.temperature = torch.tensor(1.5, dtype=torch.float32, device=self.device, requires_grad=True)
        
    def fit(self, **kwargs):
        pass

    def calibrate(self, cal_dataset, lr=0.01, max_iter=50):
        _cal_data = _prepare_logit_dataset(self.model, cal_dataset, self.debug)
        if self.num_classes is None:
            self.num_classes = max(_cal_data['labels']) + 1
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = torch.nn.CrossEntropyLoss()
        logits = torch.tensor(_cal_data['logits'], dtype=torch.float, device=self.device)
        label = torch.tensor(_cal_data['labels'], dtype=torch.long, device=self.device)
        def eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, label)
            loss.backward()
            return loss
        self.train()
        optimizer.step(eval)
        self.eval()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # like the forward for all basemodels (take batches of data)
        old_pred = self.model(**kwargs)
        logits = old_pred['logit'] / self.temperature
        y_prob = torch.softmax(logits, 1)
        return {
            'y_prob': y_prob,
            'y_true': old_pred['y_true'],
            'loss': torch.nn.CrossEntropyLoss()(logits, old_pred['y_true'])
        }
