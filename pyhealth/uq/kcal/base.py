from importlib import reload
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from pyhealth.datasets import utils as datautils
from pyhealth.uq.base_classes import PostHocCalibrator

from .bw import fit_bandwidth
from .embed_data import _EmbedData
from .kde import KDE_classification, KDECrossEntropyLoss, RBFKernelMean


class SkipELU(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(input_features)
        self.mid = torch.nn.Linear(input_features, output_features)
        self.bn2 = torch.nn.BatchNorm1d(output_features)
        self.fc = torch.nn.Linear(output_features, output_features, bias=False)
        self.act = torch.nn.ELU()

        self.criterion = KDECrossEntropyLoss()
    
    def embed(self, x):
        x = self.mid(self.bn(x))
        ret = self.fc(self.act(x))
        return ret + x

    def forward(self, data, target=None):
        device = self.fc.weight.device

        data['supp_embed'] = self.embed(data['supp_embed'].to(device))
        data['supp_target'] = data['supp_target'].to(device)
        if target is None:
            # no supp vs pred - LOO prediction (for eval)
            assert 'pred_embed' not in data
            data['pred_embed'] = None
            target = data['supp_target']
        else:
            # used for train
            data['pred_embed'] = self.embed(data['pred_embed'].to(device))
            if 'weights' in data and isinstance(data['weights'], torch.Tensor):
                data['weights'] = data['weights'].to(device)
        loss = self.criterion(data, target.to(device))
        return {'loss': loss['loss'],
                'y_pred': loss['extra_output']['prediction'],
                'y_true': target}

def _prepare_embedding_dataset(model, dataset, record_id_name='record_id', debug=False):
    embeds = []
    indices = []
    labels = []
    loader = datautils.get_dataloader(dataset, 1, shuffle=False)
    with torch.no_grad():
        for _i, data in tqdm.tqdm(enumerate(loader), desc="embedding...", total=len(loader)):
            if debug and _i % 10 != 0: continue
            res = model(embed=True, **data)
            indices.append(data[record_id_name][0])
            labels.append(res['y_true'][0].item())
            embeds.append(res['embed'][0].detach().cpu().numpy())
    return {'labels': pd.Series(labels, indices), 'embed': np.stack(embeds)}

class KCal(PostHocCalibrator):
    def __init__(self, model:torch.nn.Module, debug=False, d=32, **kwargs) -> None:
        super().__init__(model, **kwargs)
        if model.mode != 'multiclass':
            raise NotImplementedError()
        self.model.eval()
        # TODO: Try to get rid of this?
        self.device = model.device
        self.d = d
        self.debug = debug

        self.proj = None
        self.kern = RBFKernelMean()
        self.record_id_name = None
        self.cal_data = {}
        self.num_classes = None

    def fit(self, train_dataset, val_dataset=None, record_id_name='record_id', bs_pred=64, bs_supp=20):
        import pyhealth.trainer as trr

        train_embed_dataset = _prepare_embedding_dataset(self.model, train_dataset, record_id_name, self.debug)
        self.num_classes = max(train_embed_dataset['labels'].values) + 1
        train_embed_dataset = _EmbedData(bs_pred=bs_pred, bs_supp=bs_supp, niters_per_epoch=100, **train_embed_dataset)
        train_loader = DataLoader(train_embed_dataset, batch_size=1, collate_fn=_EmbedData._collate_func)

        if val_dataset is not None:
            val_embed_dataset = _prepare_embedding_dataset(self.model, val_dataset, record_id_name, self.debug)
            val_embed_dataset = _EmbedData(niters_per_epoch=1, **val_embed_dataset)
            val_loader = DataLoader(val_embed_dataset, batch_size=1, collate_fn=_EmbedData._collate_func)
        else:
            val_loader = None

        self.proj = SkipELU(len(train_embed_dataset.embed[0]), self.d).to(self.device)
        trainer = trr.Trainer(model=self.proj)
        if not self.debug:
            trainer.train(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=2,
                monitor="loss",
            )
        self.proj.eval()

        # remember a few things
        self.record_id_name = record_id_name
    
    def calibrate(self, cal_dataset, group_sep='-'):
        cal_embed_dataset = _prepare_embedding_dataset(self.model, cal_dataset, self.record_id_name, self.debug)
        self.cal_data['Y'] = torch.tensor(cal_embed_dataset['labels'].values, dtype=torch.long, device=self.device)
        self.cal_data['Y'] = torch.nn.functional.one_hot(self.cal_data['Y'], self.num_classes).float()
        with torch.no_grad():
            self.cal_data['X'] = self.proj.embed(torch.tensor(cal_embed_dataset['embed'], dtype=torch.float, device=self.device))
        
        # Choose bandwidth
        groups = None
        if group_sep is not None:
            groups = cal_embed_dataset['labels'].index.map(lambda _: _.split(group_sep)[0])
        self.kern.set_bandwidth(fit_bandwidth(groups=groups, **self.cal_data))


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # like the forward for all basemodels (take batches of data)
        old_pred = self.model(embed=True, **kwargs)
        X_pred = self.proj.embed(old_pred['embed'])
        pred = KDE_classification(kern=self.kern, X_pred=X_pred, **self.cal_data)
        return {
            'y_prob': pred,
            'y_true': old_pred['y_true']
        }

        