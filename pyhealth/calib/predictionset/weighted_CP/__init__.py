"""
WeightedCP: Weighted Conformal Prediction. Conformal Prediction with differnt weights to calibration samples. Base class for:
Conformal Prediction Under Covariate Shift
Locally Valid Discrminative Intervals
CoDrug: Conformal Drug Property Prediction with Denisty Estimation Under Covariate Shift

"""

from typing import Dict, Union
from typing import overload

import numpy as np
import torch
from torch.utils.data import Subset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

import logging
import bisect

from pyhealth.calib.utils import compute_weights_for_codrug 

__all__ = ["LABEL"]


def _query_quantile(scores, alpha):
    scores = np.sort(scores)
    N = len(scores)
    loc = int(np.floor(alpha * (N + 1))) - 1
    return -np.inf if loc == -1 else scores[loc]

def _query_weighted_quantile(scores, alpha, weights):
    qs = np.cumsum(weights)/np.sum(weights)
    idx = bisect.bisect_left(qs, alpha, lo=0, hi=len(qs)-1)
    return scores[idx]

class WeightedCP(SetPredictor):
    """LABEL: Least ambiguous set-valued classifiers with bounded error levels.

    This is a prediction-set constructor for multi-class classification problems.
    It controls either :math:`\\mathbb{P}\\{Y \\not \\in C(X) | Y=k\\}\\leq \\alpha_k`
    (when ``alpha`` is an array), or :math:`\\mathbb{P}\\{Y \\not \\in C(X)\\}\\leq \\alpha` (when ``alpha`` is a float).
    Here, :math:`C(X)` denotes the final prediction set.
    This is essentially a split conformal prediction method using the predicted scores.

    Paper:

        Sadinle, Mauricio, Jing Lei, and Larry Wasserman.
        "Least ambiguous set-valued classifiers with bounded error levels."
        Journal of the American Statistical Association 114, no. 525 (2019): 223-234.


    :param model: A trained base model.
    :type model: BaseModel
    :param alpha: Target mis-coverage rate(s).
    :type alpha: Union[float, np.ndarray]

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset import LABEL
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate the set classifier, with different class-specific mis-coverage rates
        >>> cal_model = LABEL(model, [0.15, 0.3, 0.15, 0.15, 0.15])
        >>> # Note that we used the test set here because ISRUCDataset has relatively few
        >>> # patients, and calibration set should be different from the validation set
        >>> # if the latter is used to pick checkpoint. In general, the calibration set
        >>> # should be something exchangeable with the test set. Please refer to the paper.
        >>> cal_model.calibrate(cal_dataset=test_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(test_dl, additional_outputs=['y_predset'])
        >>> print(get_metrics_fn(cal_model.mode)(
        ... y_true_all, y_prob_all, metrics=['accuracy', 'miscoverage_ps'],
        ... y_predset=extra_output['y_predset'])
        ... )
        {'accuracy': 0.709843241966832, 'miscoverage_ps': array([0.1499847 , 0.29997638, 0.14993964, 0.14994704, 0.14999252])}
    """

    def __init__(
        self, 
        model: BaseModel, 
        alpha: Union[float, np.ndarray], 
        debug=False,
         **kwargs
    ) -> None:
        super().__init__(model, **kwargs)
        if model.mode != "multiclass":
            raise NotImplementedError()
        self.mode = self.model.mode  # multiclass
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = model.device
        self.debug = debug
        if not isinstance(alpha, float):
            alpha = np.asarray(alpha)
        self.alpha = alpha

        self.t = None

    def calibrate(self, cal_dataset: Union[Subset, Dict], weights=None, calibration_type="inductive"):
        """Calibrate the thresholds used to construct the prediction set.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        """
        assert (calibration_type == "mondrian" or calibration_type == "inductive", "Expection calibration type is inductive or mondrian. Got: {calibration_type}")
        
        if not isinstance(cal_dataset, dict):
            cal_dataset = prepare_numpy_dataset(
                self.model, cal_dataset, ["y_prob", "y_true"], debug=self.debug
            )

        y_prob = cal_dataset["y_prob"]
        y_true = cal_dataset["y_true"]

        if weights is None:
            weights = [1.0 for i in range(len(y_prob))]
            logging.warning("No weights specified for calibration. Calibrating with equal weights for all samples")
        
        assert (len(weights) == len(y_prob), f"Weights should be of same size as y_prob, Expected {len(y_prob)} weights has {len(weights)} values")

        N, K = cal_dataset["y_prob"].shape
        if calibration_type == "inductive":
            if isinstance(self.alpha, float):
                t = _query_weighted_quantile(y_prob[np.arange(N), y_true], self.alpha, weights)
            else:
                t = [
                    _query_weighted_quantile(y_prob[np.arange(N), y_true], self.alpha[k], weights) for k in range(K)
                ]
        
        self.t = torch.tensor(t, device=self.device)


    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :return: A dictionary with all results from the base model, with the following updates:

                    y_predset: a bool tensor representing the prediction for each class.
        :rtype: Dict[str, torch.Tensor]
        """
        pred = self.model(**kwargs)
        pred["y_predset"] = pred["y_prob"].cpu() > self.t
        return pred


class CoDrug(WeightedCP):
    
    def __init__(self, model: BaseModel, alpha: Union[float, np.ndarray], debug=False, **kwargs) -> None:
        super().__init__(model, alpha, debug, **kwargs)

    
    def calibrate(self, cal_dataset: Subset, test_dataset: Subset, calibration_type="inductive"):
        cal_preds = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_prob", "y_true"], debug=self.debug
        )

        test_preds = prepare_numpy_dataset(
            self.model, test_dataset, ["y_prob", "y_true"], debug=self.debug
        )
        weights = compute_weights_for_codrug(cal_preds["y_prob"], test_preds["y_prob"])
        return super().calibrate(cal_preds, weights, calibration_type)


if __name__ == "__main__":

    from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn
    from pyhealth.calib.predictionset.weighted_CP import WeightedCP
    from pyhealth.calib.predictionset.weighted_CP import CoDrug

    sleep_ds = ISRUCDataset("/srv/scratch/sl160/temp", dev=True,download="True").set_task(
        sleep_staging_isruc_fn
    )
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
    model = SparcNet(
        dataset=sleep_ds, feature_keys=["signal"], label_key="label", mode="multiclass"
    )
    # ... Train the model here ...
    # Calibrate the set classifier, with different class-specific mis-coverage rates
    # cal_model = WeightedCP(model, [0.15, 0.3, 0.15, 0.15, 0.15])
    cal_model = CoDrug(model, [0.15, 0.3, 0.15, 0.15, 0.15])
    # Note that I used the test set here because ISRUCDataset has relatively few
    # patients, and calibration set should be different from the validation set
    # if the latter is used to pick checkpoint. In general, the calibration set
    # should be something exchangeable with the test set. Please refer to the paper.
    cal_model.calibrate(cal_dataset=val_data, test_dataset=test_data)

    # Evaluate
    from pyhealth.trainer import Trainer, get_metrics_fn

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    print(
        get_metrics_fn(cal_model.mode)(
            y_true_all,
            y_prob_all,
            metrics=["accuracy", "miscoverage_ps"],
            y_predset=extra_output["y_predset"],
        )
    )
