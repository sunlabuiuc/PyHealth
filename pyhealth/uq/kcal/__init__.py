"""
KCal: Kernel-based Calibration 

From:
    Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun. 
    "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration for Deep Neural Networks." 
    ICLR 2023.

Implementation based on https://github.com/zlin7/KCal

"""
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from pyhealth.uq.base_classes import PostHocCalibrator
from pyhealth.uq.utils import prepare_numpy_dataset

from .bw import fit_bandwidth
from .embed_data import _EmbedData
from .kde import KDE_classification, KDECrossEntropyLoss, RBFKernelMean

__all__ = ['KCal']

class ProjectionWrap(torch.nn.Module):
    """Base class for reprojections.
    """
    def __init__(self) -> None:
        super().__init__()
        self.criterion = KDECrossEntropyLoss()
        self.mode = 'multiclass'

    def embed(self, x):
        """The actual projection"""
        raise NotImplementedError()

    def _forward(self, data, target=None, device=None):
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
    """The identity reprojection (no reprojection).
    """

    def embed(self, x):
        return x

    def forward(self, data, target=None):
        """Foward operations"""
        return self._forward(data, target, data['supp_embed'].device)

class SkipELU(ProjectionWrap):
    """The default reprojection module with 2 layers and a skip connection.
    """
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
        """Foward operations"""
        return self._forward(data, target, self.fc.weight.device)

def _embed_dataset(model, dataset, record_id_name=None, debug=False, batch_size=32):

    ret = prepare_numpy_dataset(
        model, dataset, ['y_true', 'embed'], 
        incl_data_keys=['patient_id'] + ([] if record_id_name is None else [record_id_name]),
        forward_kwargs={'embed': True}, debug=debug, batch_size=batch_size)
    return {"labels": pd.Series(ret['y_true'], ret.get(record_id_name, None)),
            'embed': ret['embed'], 'group': ret['patient_id']}

class KCal(PostHocCalibrator):
    """Kernel-based Calibration. 

    This is a *full* calibration method for *multiclass* classification. 
    It tries to calibrate the predicted probabilities for all classes, 
        by using KDE classifiers estimated from the calibration set. 

    Paper: Lin, Zhen, Shubhendu Trivedi, and Jimeng Sun. 
        "Taking a Step Back with KCal: Multi-Class Kernel-Based Calibration 
        for Deep Neural Networks." ICLR 2023.

    Args:
        model (BaseModel): A trained model.

    Examples:
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_staging_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate
        >>> cal_model = uq.KCal(model)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Alternatively, you could re-fit the reprojection:
        >>> # cal_model.calibrate(cal_dataset=val_data, train_dataset=train_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> print(Trainer(model=cal_model).evaluate(test_dl))
    """
    def __init__(self, model:torch.nn.Module, debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        if model.mode != 'multiclass':
            raise NotImplementedError()
        self.mode = self.model.mode # multiclass
        self.model.eval()

        self.device = model.device
        self.debug = debug

        self.proj = Identity()
        self.kern = RBFKernelMean()
        self.record_id_name = None
        self.cal_data = {}
        self.num_classes = None

    def _fit(self, train_dataset, val_dataset=None, split_by_patient=False,
            dim=32, bs_pred=64, bs_supp=20, epoch_len=5000, epochs=10,
            load_best_model_at_last=False):
        from pyhealth.trainer import Trainer

        _train_data = _embed_dataset(self.model, train_dataset, self.record_id_name, self.debug)

        self.num_classes = max(_train_data['labels'].values) + 1
        if not split_by_patient:
            # Allow using other samples from the same patient to make the prediction
            _train_data.pop('group')
        _train_data = _EmbedData(bs_pred=bs_pred, bs_supp=bs_supp, epoch_len=epoch_len, 
                                 **_train_data)
        train_loader = DataLoader(_train_data, batch_size=1, collate_fn=_EmbedData._collate_func)

        val_loader = None
        if val_dataset is not None:
            _val_data = _embed_dataset(self.model, val_dataset, self.record_id_name, self.debug)
            _val_data = _EmbedData(epoch_len=1, **_val_data)
            val_loader = DataLoader(_val_data, batch_size=1, collate_fn=_EmbedData._collate_func)

        self.proj = SkipELU(len(_train_data.embed[0]), dim).to(self.device)
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

    def calibrate(self, cal_dataset:Subset, record_id_name='record_id',
                  train_dataset:Subset=None, train_split_by_patient=False,
                  load_best_model_at_last=False, **train_kwargs):
        """Calibrate using a calibration dataset. 
        If train_dataset is not None, it will be used to fit 
        a re-projection from the base model embeddings.
        In either case, the calibration set will be used to construct the KDE classifier. 

        Args:
            cal_dataset (Subset): Calibration set.
            record_id_name (str, optional): the key/name of the unique index for records. 
                Defaults to 'record_id'.
            train_dataset (Subset, optional): Dataset to train the reprojection. 
                Defaults to None (no training).
            train_split_by_patient (bool, optional): 
                Whether to split by patient when training the embeddings. 
                That is, do we use samples from the same patient in KDE during *training*.
                Defaults to False.
            load_best_model_at_last (bool, optional): 
                Whether to load the best reprojection basing on the calibration set. 
                Defaults to False.
        """

        self.record_id_name = record_id_name
        if train_dataset is not None:
            self._fit(train_dataset, val_dataset=cal_dataset,
                      split_by_patient=train_split_by_patient,
                      load_best_model_at_last=load_best_model_at_last,
                      **train_kwargs)
        else:
            print("No `train_dataset` - using the raw embeddings from the base classifier.")

        _cal_data = _embed_dataset(self.model, cal_dataset, self.record_id_name, self.debug)
        if self.num_classes is None:
            self.num_classes = max(_cal_data['labels'].values) + 1
        self.cal_data['Y'] = torch.tensor(
            _cal_data['labels'].values, dtype=torch.long, device=self.device)
        self.cal_data['Y'] = torch.nn.functional.one_hot(
            self.cal_data['Y'], self.num_classes).float()
        with torch.no_grad():
            self.cal_data['X'] = self.proj.embed(torch.tensor(
                _cal_data['embed'], dtype=torch.float, device=self.device))

        # Choose bandwidth
        self.kern.set_bandwidth(fit_bandwidth(group=_cal_data['group'], **self.cal_data))


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        Returns:
            A dictionary with all results from the base model, with the following updated:
                y_prob: calibrated predicted probabilities.
                loss: Cross entropy loss (log-loss, to be precise) with the new y_prob.
        """
        ret = self.model(embed=True, **kwargs)
        X_pred = self.proj.embed(ret.pop('embed'))
        ret['y_prob'] = KDE_classification(kern=self.kern, X_pred=X_pred, **self.cal_data)
        ret['loss'] = self.proj.criterion.log_loss(ret['y_prob'], ret['y_true'])
        return ret
