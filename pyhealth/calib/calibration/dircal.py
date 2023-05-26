"""
Dirichlet Calibration.

Implementation based on https://github.com/dirichletcal/dirichlet_python
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Subset

from pyhealth.calib.base_classes import PostHocCalibrator
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel


def _get_identity_weights(n_classes, method="Full"):
    raw_weights = None
    if (method is None) or (method == "Full"):
        raw_weights = torch.zeros((n_classes, n_classes + 1)) + torch.hstack(
            [torch.eye(n_classes), torch.zeros((n_classes, 1))]
        )
    else:
        raise NotImplementedError
    return raw_weights.ravel()


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - torch.max(X, axis=1).values.reshape(-1, 1)
    exps = torch.exp(shiftx)
    return exps / torch.sum(exps, axis=1).reshape(-1, 1)


def _get_weights(params, k, ref_row, method):
    """Reshapes the given params (weights) into the full matrix including 0"""

    if method in ["Full", None]:
        raw_weights = params.reshape(-1, k + 1)
    if ref_row:
        weights = raw_weights - torch.repeat_interleave(
            raw_weights[-1, :].reshape(1, -1), k, dim=0
        )
    else:
        weights = raw_weights
    return weights


class DirichletCalibration(PostHocCalibrator):
    """Dirichlet Calibration

    Dirichlet calibration is similar to retraining a linear layer mapping from the
    old logits to the new logits with regularizations.
    This is a calibration method for *multiclass* classification only.

    Paper:

        [1] Kull, Meelis, Miquel Perello Nieto, Markus Kängsepp,
        Telmo Silva Filho, Hao Song, and Peter Flach.
        "Beyond temperature scaling: Obtaining well-calibrated multi-class
        probabilities with dirichlet calibration."
        Advances in neural information processing systems 32 (2019).

    :param model: A trained base model.
    :type model: BaseModel

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.calibration import DirichletCalibration
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate
        >>> cal_model = DirichletCalibration(model)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> print(Trainer(model=cal_model, metrics=['cwECEt_adapt', 'accuracy']).evaluate(test_dl))
        {'accuracy': 0.7096615988229524, 'cwECEt_adapt': 0.05336195546573208}
    """

    def __init__(self, model: BaseModel, debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.mode = self.model.mode
        for param in model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.device = model.device
        self.debug = debug

        self.num_classes = None

    def _forward(self, logits):
        logits = torch.hstack(
            [logits, torch.zeros((logits.shape[0], 1), device=self.device)]
        )
        weights = _get_weights(
            self.weights, k=self.num_classes, ref_row=True, method="Full"
        )
        new_logits = torch.matmul(logits, weights.permute(1, 0))
        return new_logits, weights

    def calibrate(self, cal_dataset: Subset, lr=0.01, max_iter=128, reg_lambda=1e-3):
        """Calibrate the base model using a calibration dataset.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        :param lr: learning rate, defaults to 0.01
        :type lr: float, optional
        :param max_iter: maximum iterations, defaults to 128
        :type max_iter: int, optional
        :param reg_lambda: regularization coefficient on the deviation from identity matrix.
            defaults to 1e-3
        :type reg_lambda: float, optional
        :return: None
        :rtype: None
        """

        _cal_data = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_true", "logit"], debug=self.debug
        )
        if self.num_classes is None:
            self.num_classes = _cal_data["logit"].shape[1]

        self.weights = torch.nn.Parameter(
            torch.tensor(_get_identity_weights(self.num_classes), device=self.device),
            requires_grad=True,
        )
        optimizer = optim.LBFGS([self.weights], lr=lr, max_iter=max_iter)
        logits = torch.tensor(_cal_data["logit"], dtype=torch.float, device=self.device)
        label = torch.tensor(
            F.one_hot(torch.tensor(_cal_data["y_true"]), num_classes=self.num_classes),
            dtype=torch.long if self.model.mode == "multiclass" else torch.float32,
            device=self.device,
        )

        reg_mu = None
        reg_format = "identity"

        def _eval():
            optimizer.zero_grad()
            new_logits, weights = self._forward(logits)
            probs = _softmax(new_logits)
            loss = torch.mean(-torch.log(torch.sum(label * probs, dim=1)))
            if reg_mu is None:
                if reg_format == "identity":
                    reg = torch.hstack(
                        [
                            torch.eye(self.num_classes),
                            torch.zeros((self.num_classes, 1)),
                        ]
                    )
                else:
                    reg = torch.zeros((self.num_classes, self.num_classes + 1))
                loss = loss + reg_lambda * torch.sum(
                    (weights - reg.to(self.device)) ** 2
                )
            else:
                weights_hat = weights - torch.hstack(
                    [
                        weights[:, :-1]
                        * torch.eye(self.num_classes, device=self.device),
                        torch.zeros((self.num_classes, 1), device=self.device),
                    ]
                )
                loss = (
                    loss
                    + reg_lambda * torch.sum(weights_hat[:, :-1] ** 2)
                    + reg_mu * torch.sum(weights_hat[:, -1] ** 2)
                )

            loss.backward()
            return loss

        self.train()
        optimizer.step(_eval)
        self.eval()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :param **kwargs: Additional arguments to the base model.

        :return:  A dictionary with all results from the base model, with the following modified:

            ``y_prob``: calibrated predicted probabilities.
            ``loss``: Cross entropy loss with the new y_prob.
            ``logit``: temperature-scaled logits.
        :rtype: Dict[str, torch.Tensor]
        """
        ret = self.model(**kwargs)
        ret["logit"] = self._forward(ret["logit"])[0]
        ret["y_prob"] = self.model.prepare_y_prob(ret["logit"])
        criterion = self.model.get_loss_function()
        ret["loss"] = criterion(ret["logit"], ret["y_true"])
        return ret


if __name__ == "__main__":
    from pyhealth.calib.calibration import DirichletCalibration
    from pyhealth.datasets import (ISRUCDataset, get_dataloader,
                                   split_by_patient)
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn

    sleep_ds = ISRUCDataset(
        root="/srv/local/data/trash/",
        dev=True,
    ).set_task(sleep_staging_isruc_fn)
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
    model = SparcNet(
        dataset=sleep_ds, feature_keys=["signal"], label_key="label", mode="multiclass"
    )
    # ... Train the model here ...
    # Calibrate
    cal_model = DirichletCalibration(model)
    cal_model.calibrate(cal_dataset=val_data)
    # Evaluate
    from pyhealth.trainer import Trainer

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    print(
        Trainer(model=cal_model, metrics=["cwECEt_adapt", "accuracy"]).evaluate(test_dl)
    )
