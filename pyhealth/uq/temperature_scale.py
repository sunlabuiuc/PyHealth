"""
Temperature/Platt Scaling.

Implementation based on https://github.com/gpleiss/temperature_scaling
"""

from typing import Dict

import torch
from torch import optim

from pyhealth.models import BaseModel
from pyhealth.uq.base_classes import PostHocCalibrator
from pyhealth.uq.utils import prepare_numpy_dataset

__all__ = ['TemperatureScaling']

class TemperatureScaling(PostHocCalibrator):
    """Temperature Scaling 

    Temprature scaling refers to scaling the logits by a "temprature" tuned 
    on the calibration set. For binary classification tasks, this amounts to
    Platt scaling. For multilabel classification, users can use one temperature
    for all classes, or one for each. For multiclass classification, this is 
    a *confidence* calibration method: It tries to calibrate the predicted 
    class' predicted probability. 


    Paper: 
        [1] Guo, Chuan, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 
        "On calibration of modern neural networks." ICML 2017.
        
        [2] Platt, John. 
        "Probabilistic outputs for support vector machines and 
        comparisons to regularized likelihood methods." 
        Advances in large margin classifiers 10, no. 3 (1999): 61-74.

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
        >>> cal_model = uq.TemperatureScaling(model)
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

        self.temperature = torch.tensor(
            1.5, dtype=torch.float32, device=self.device, requires_grad=True)

    def calibrate(self, cal_dataset, lr=0.01, max_iter=50, mult_temp=False):
        """calibrate self.model using cal_dataset. 

        Args:
            cal_dataset (_type_): _description_
            lr (float, optional): learning rate. Defaults to 0.01.
            max_iter (int, optional): maximum numerb of iterations. Defaults to 50.
            mult_temp (bool): if mult_temp and mode='multilabel', 
                use one temperature for each class.

        Returns:
            None
        """
        _cal_data = prepare_numpy_dataset(self.model, cal_dataset, 
                                          ['y_true', 'logit'], debug=self.debug)
        
        if self.num_classes is None:
            self.num_classes = _cal_data['logit'].shape[1]
        if self.mode == 'multilabel' and mult_temp:
            self.temperature = torch.tensor([1.5 for _ in range(self.num_classes)], 
                                            dtype=torch.float32,
                                            device=self.device, requires_grad=True)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = self.model.get_loss_function()
        logits = torch.tensor(_cal_data['logit'], dtype=torch.float, device=self.device)
        label = torch.tensor(_cal_data['y_true'],
                             dtype=torch.long if self.model.mode == 'multiclass' else torch.float32,
                             device=self.device)
        def _eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, label)
            loss.backward()
            return loss
        self.train()
        optimizer.step(_eval)
        self.eval()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        Returns:
            A dictionary with all results from the base model, with the following modified:
                logit: temperature-scaled logits.
                y_prob: calibrated predicted probabilities.
                loss: Cross entropy loss with the new logits.
        """
        ret = self.model(**kwargs)
        ret['logit'] = ret['logit'] / self.temperature
        ret['y_prob'] = self.model.prepare_y_prob(ret['logit'])
        criterion = self.model.get_loss_function()
        ret['loss'] = criterion(ret['logit'], ret['y_true'])
        return ret
