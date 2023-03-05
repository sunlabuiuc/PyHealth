"""
KCal: Kernel-based Calibration 

From:
    Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun. "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks." ICLR 2023.

Implementation based on https://github.com/zlin7/KCal

"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from pyhealth.datasets import utils as datautils
from pyhealth.uq.base_classes import PostHocCalibrator
from pyhealth.uq.utils import prepare_numpy_dataset

from .bw import fit_bandwidth
from .embed_data import _EmbedData
from .kde import KDE_classification, KDECrossEntropyLoss, RBFKernelMean

__all__ = ['KCal']

class ProjectionWrap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = KDECrossEntropyLoss()
        self.mode = 'multiclass' #TODO: use metric later

    def embed(self, x):
        raise NotImplementedError()

    def forward(self, data, target=None, device=None):
        device = device or self.fc.weight.device

        data['supp_embed'] = self.embed(data['supp_embed'].to(device))
        data['supp_target'] = data['supp_target'].to(device)
        if target is None:
            # no supp vs pred - LOO prediction (for eval)
            assert 'pred_embed' not in data
            data['pred_embed'] = None
            target = data['supp_target']
            assert not self.training
        else:
            # used for train
            data['pred_embed'] = self.embed(data['pred_embed'].to(device))
            if 'weights' in data and isinstance(data['weights'], torch.Tensor):
                data['weights'] = data['weights'].to(device)
        loss = self.criterion(data, target.to(device), eval_only=data['pred_embed'] is None)
        return {'loss': loss['loss'],
                'y_prob': loss['extra_output']['prediction'],
                'y_true': target}

class Identity(ProjectionWrap):
    def __init__(self):
        super().__init__()
    def embed(self, x):
        return x
    def forward(self, data, target=None):
        return super().forward(data, target, data['supp_embed'].device)

class SkipELU(ProjectionWrap):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(input_features)
        self.mid = torch.nn.Linear(input_features, output_features)
        self.bn2 = torch.nn.BatchNorm1d(output_features)
        self.fc = torch.nn.Linear(output_features, output_features, bias=False)
        self.act = torch.nn.ELU()

    def embed(self, x):
        x = self.mid(self.bn(x))
        ret = self.fc(self.act(x))
        return ret + x

    def forward(self, data, target=None):
        return super().forward(data, target, self.fc.weight.device)

def _prepare_embedding_dataset(model, dataset, record_id_name=None, debug=False, batch_size=32):

    ret = prepare_numpy_dataset(model, dataset, ['y_true', 'embed'],
                                incl_data_keys=['patient_id'] + ([] if record_id_name is None else [record_id_name]),
                                forward_kwargs={'embed': True}, debug=debug, batch_size=batch_size)
    return {"labels": pd.Series(ret['y_true'], ret.get(record_id_name, None)),
            'embed': ret['embed'], 'group': ret['patient_id']}

class KCal(PostHocCalibrator):
    def __init__(self, model:torch.nn.Module, debug=False, dim=32, **kwargs) -> None:
        super().__init__(model, **kwargs)
        if model.mode != 'multiclass':
            raise NotImplementedError()
        self.mode = self.model.mode # multiclass
        self.model.eval()
        
        self.device = model.device
        self.dim = dim
        self.debug = debug

        self.proj = Identity()
        self.kern = RBFKernelMean()
        self.record_id_name = None
        self.cal_data = {}
        self.num_classes = None
        
    def _fit(self, train_dataset, val_dataset=None, split_by_patient=True,
            bs_pred=64, bs_supp=20, niters_per_epoch=5000, epochs=10,
            load_best_model_at_last=False):
        from pyhealth.trainer import Trainer

        _train_data = _prepare_embedding_dataset(self.model, train_dataset, self.record_id_name, self.debug)
        
        self.num_classes = max(_train_data['labels'].values) + 1
        if not split_by_patient:
            # Allow using other samples from the same patient to make the prediction
            _train_data.pop('group')
        _train_data = _EmbedData(bs_pred=bs_pred, bs_supp=bs_supp, niters_per_epoch=niters_per_epoch, **_train_data)
        train_loader = DataLoader(_train_data, batch_size=1, collate_fn=_EmbedData._collate_func)

        val_loader = None
        if val_dataset is not None:
            _val_data = _prepare_embedding_dataset(self.model, val_dataset, self.record_id_name, self.debug)
            _val_data = _EmbedData(niters_per_epoch=1, **_val_data)
            val_loader = DataLoader(_val_data, batch_size=1, collate_fn=_EmbedData._collate_func)
            
        self.proj = SkipELU(len(_train_data.embed[0]), self.dim).to(self.device)
        trainer = Trainer(model=self.proj)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            monitor="loss",
            monitor_criterion='min',
            load_best_model_at_last=load_best_model_at_last,
        )
        self.proj.eval()
    
    def calibrate(self, cal_dataset, record_id_name='record_id',
                  train_dataset=None, train_split_by_patient=True,
                  load_best_model_at_last=False, **train_kwargs):
        self.record_id_name = record_id_name
        if train_dataset is not None:
            self._fit(train_dataset, val_dataset=cal_dataset,
                      split_by_patient=train_split_by_patient,
                      load_best_model_at_last=load_best_model_at_last,
                      **train_kwargs)
        else:
            print("No `train_dataset` found - using the raw embeddings from the base classifier.")

        _cal_data = _prepare_embedding_dataset(self.model, cal_dataset, self.record_id_name, self.debug)
        if self.num_classes is None:
            self.num_classes = max(_cal_data['labels'].values) + 1
        self.cal_data['Y'] = torch.tensor(_cal_data['labels'].values, dtype=torch.long, device=self.device)
        self.cal_data['Y'] = torch.nn.functional.one_hot(self.cal_data['Y'], self.num_classes).float()
        with torch.no_grad():
            self.cal_data['X'] = self.proj.embed(torch.tensor(_cal_data['embed'], dtype=torch.float, device=self.device))
        
        # Choose bandwidth
        self.kern.set_bandwidth(fit_bandwidth(group=_cal_data['group'], **self.cal_data))


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # like the forward for all basemodels (take batches of data)
        old_pred = self.model(embed=True, **kwargs)
        X_pred = self.proj.embed(old_pred['embed'])
        pred = KDE_classification(kern=self.kern, X_pred=X_pred, **self.cal_data)
        return {
            'y_prob': pred,
            'y_true': old_pred['y_true'],
            'loss': self.proj.criterion.log_loss(pred, old_pred['y_true'])
        }
