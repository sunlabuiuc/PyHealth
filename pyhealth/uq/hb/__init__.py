"""
Histogram Binning.

Implementation based on https://github.com/aigen/df-posthoc-calibration

"""
from typing import Dict

import numpy as np
import torch

from pyhealth.models import BaseModel
from pyhealth.uq.base_classes import PostHocCalibrator
from pyhealth.uq.hb.calib import HB_binary
from pyhealth.uq.utils import LogLoss, one_hot_np, prepare_numpy_dataset

__all__ = ['HistogramBinning']


class HistogramBinning(PostHocCalibrator):
    """Histogram Binning

    Histogram binning amounts to creating bins and computing the accuracy 
    for each bin using the calibration dataset, and then predicting such 
    at test time. For multilabel/binary/multiclass classification tasks, 
    we calibrate each class independently following [1]. Users could choose 
    to renormalize the probability scores for multiclass tasks so they sum
    to 1. 


    Paper: 
        [1] Gupta, Chirag, and Aaditya Ramdas. 
        "Top-label calibration and multiclass-to-binary reductions." 
        ICLR 2022.
        
        [2] Zadrozny, Bianca, and Charles Elkan. 
        "Learning and making decisions when costs and probabilities are both unknown." 
        In Proceedings of the seventh ACM SIGKDD international conference on Knowledge 
        discovery and data mining, pp. 204-213. 2001.

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
        >>> cal_model = uq.HistogramBinning(model)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> print(Trainer(model=cal_model).evaluate(test_dl))
    """

    def __init__(self, model:BaseModel, debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.mode = self.model.mode
        for param in model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.device = model.device
        self.debug = debug

        self.num_classes = None
        self.calib = None

    def calibrate(self, cal_dataset):
        _cal_data = prepare_numpy_dataset(self.model, cal_dataset,
                                          ['y_true', 'y_prob'], debug=self.debug)
        if self.num_classes is None:
            self.num_classes = _cal_data['y_prob'].shape[1]
        if self.mode == 'binary':
            self.calib = [HB_binary().fit(_cal_data['y_prob'][:, 0], _cal_data['y_true'][:, 0])]
        else:
            self.calib = []
            y_true = _cal_data['y_true']
            if len(y_true.shape) == 1:
                y_true = one_hot_np(y_true, self.num_classes)
            for k in range(self.num_classes):
                self.calib.append(HB_binary().fit(_cal_data['y_prob'][:,k], y_true[:, k]))


    def forward(self, normalization='sum', **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        Returns:
            A dictionary with all results from the base model, with the following modified:
                y_prob: calibrated predicted probabilities.
                loss: Cross entropy loss  with the new y_prob.
        """
        assert normalization is None or normalization == 'sum'
        ret = self.model(**kwargs)
        y_prob = ret['y_prob'].cpu().numpy()
        for k in range(self.num_classes):
            y_prob[:, k] = self.calib[k].predict_proba(y_prob[:, k])
        if normalization == 'sum' and self.mode == 'multiclass':
            y_prob = y_prob / y_prob.sum(1)[:, np.newaxis]
        ret['y_prob'] = torch.tensor(y_prob, dtype=torch.float, device=self.device)
        if self.mode == 'multiclass':
            criterion = LogLoss()
        else:
            criterion = torch.nn.functional.binary_cross_entropy
        ret['loss'] = criterion(ret['y_prob'], ret['y_true'])
        ret.pop('logit')
        return ret
