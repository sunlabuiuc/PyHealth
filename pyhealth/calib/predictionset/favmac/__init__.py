"""
Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)

Implementation based on https://github.com/zlin7/FavMac

"""

from importlib import reload
import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch

from pyhealth.calib.base_classes import SetPredictor
import pyhealth.calib.predictionset.favmac.core as _core
reload(_core)
#from pyhealth.calib.predictionset.favmac.core import FavMac_GreedyRatio
FavMac_GreedyRatio = _core.FavMac_GreedyRatio
#from pyhealth.calib.predictionset.favmac.set_function import \
#    AdditiveSetFunction
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["FavMac"]

INTEGER_SAFE_DELTA = 0.1


class AdditiveSetFunction:
    def __init__(self, values: Union[float, np.ndarray, int], mode=None, name='unknown') -> None:
        self.name = name
        self.values = values
        assert mode is None or mode in {'util', 'cost', 'proxy'}
        self.mode = mode
        #self._C_max = self.values.sum() if isinstance(self.values, np.ndarray) else None

    def is_additive(self):
        return True

    def __call__(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=100, target_cost=None) -> float:
        if self.mode == 'cost':
            assert pred is None
            return self.cost_call(S, Y)
        if self.mode == 'proxy':
            assert Y is None
            return self.proxy_call(S, pred, target_cost=target_cost)
        assert self.mode == 'util'
        return self.util_call(S, Y, pred, sample=sample)

    def naive_call(self, S: np.ndarray) -> float:
        #C_max = self._C_max or len(S) * self.values
        return np.sum(S * self.values) #/ C_max

    def util_call(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=1000) -> float:
        assert Y is None or pred is None
        if pred is not None:
            return self.naive_call(S * pred) # THis is because this is additive.
        if Y is not None: return self.naive_call(S * Y)
        return self.naive_call(S)

    def cost_call(self, S: np.ndarray, Y:np.ndarray) -> float:
        return self.naive_call(S * (1-Y))

    def proxy_call(self, S: np.ndarray, pred: np.ndarray, target_cost: float=None) -> float:
        return self.naive_call(S * (1-pred))
        #if target_cost is None: #
        #return self.quantile_method(S, pred, target_cost) #Does not need to be normalized

    def greedy_maximize(self, S: np.ndarray, pred: np.ndarray=None, d_proxy:np.ndarray=None, prev_util_and_proxy=None):
        # (prev_u, prev_p) = prev_util_and_proxy
        assert self.mode == 'util', "This is only used for util function"
        if (1-S).sum() == 0: return None
        d_util = self.values
        if pred is not None: d_util = d_util * pred

        objective = d_util / (1 if d_proxy is None else d_proxy.clip(1e-8))
        k = pd.Series((1-S) * objective).dropna().idxmax()
        return k, objective[k]

    def greedy_maximize_seq(self, pred: np.ndarray=None, d_proxy:np.ndarray=None):
        # if cost is also additive, then cost_proxy is fixed: weight * (1-p)
        assert self.mode == 'util', "This is only used for util function"
        d_util = self.values
        if pred is not None: d_util = d_util * pred

        objective = d_util / (1 if d_proxy is None else d_proxy.clip(1e-8))

        assert np.isnan(objective).sum() == 0
        ks = np.argsort(-objective)
        Ss = [np.zeros(len(objective), dtype=int)]
        for k in ks:
            Ss.append(Ss[-1].copy())
            Ss[-1][k] = 1
        return Ss, ks


class FavMac(SetPredictor):
    """Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)

    Only supports additive set functions for now.

    Paper:

        Lin, Zhen, Shubhendu Trivedi, Cao Xiao, and Jimeng Sun.
        "Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)."
        ICML 2023.

    """

    def __init__(
        self,
        model: BaseModel,
        value_weights: Union[float, np.ndarray] = 1.,
        cost_weights: Union[float, np.ndarray] = 1.,
        target_cost: float = 0.1, delta:float = None,
        debug=False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if model.mode != "multilabel":
            raise NotImplementedError()
        self.mode = self.model.mode  # multiclass
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = model.device
        self.debug = debug


        self._cost_weights = cost_weights
        self._value_weights = value_weights
        self.target_cost = target_cost
        self.delta = delta

        #TODO: figure out how the cost/util should be set (C_max)

    def calibrate(self, cal_dataset):
        """Calibrate/Search for the thresholds used to construct the prediction set.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        """
        _cal_data = prepare_numpy_dataset(
            self.model, cal_dataset, ["logit", "y_true"], debug=self.debug
        )

        if isinstance(self._cost_weights, np.ndarray):
            C_max = self._cost_weights.sum()
        else:
            C_max = _cal_data["logit"].shape[1] * self._cost_weights
        self._favmac = FavMac_GreedyRatio(
            cost_fn=AdditiveSetFunction(self._cost_weights / C_max, mode='cost'),
            util_fn=AdditiveSetFunction(self._value_weights, mode='util'),
            proxy_fn=AdditiveSetFunction(self._cost_weights / C_max, mode='proxy'),
            target_cost=self.target_cost/C_max, delta=self.delta, C_max=1.,
        )
        self._favmac.init_calibrate(_cal_data["logit"], _cal_data["y_true"])

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :return: A dictionary with all results from the base model, with the following updates:

                y_predset: a bool tensor representing the prediction for each class.
        :rtype: Dict[str, torch.Tensor]
        """
        ret = self.model(**kwargs)
        _logit = ret["logit"].cpu().numpy()
        y_predset = np.asarray([self._favmac(_)[0] for _ in _logit])

        ret["y_predset"] = torch.tensor(y_predset)
        return ret


if __name__ == "__main__":
    from pyhealth.calib.predictionset import FavMac
    from pyhealth.datasets import (ISRUCDataset, get_dataloader,
                                   split_by_patient)
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn
    from pyhealth.trainer import get_metrics_fn

    sleep_ds = ISRUCDataset("/srv/local/data/trash", dev=True).set_task(
        sleep_staging_isruc_fn
    )
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
    model = SparcNet(
        dataset=sleep_ds,
        feature_keys=["signal"],
        label_key="label",
        mode="multiclass",
    )
    # ... Train the model here ...
    # Calibrate the set classifier, with different class-specific risk targets
    cal_model = SCRIB(model, [0.2, 0.3, 0.1, 0.2, 0.1])
    # Note that I used the test set here because ISRUCDataset has relatively few
    # patients, and calibration set should be different from the validation set
    # if the latter is used to pick checkpoint. In general, the calibration set
    # should be something exchangeable with the test set. Please refer to the paper.
    cal_model.calibrate(cal_dataset=test_data)
    # Evaluate
    from pyhealth.trainer import Trainer

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    print(
        get_metrics_fn(cal_model.mode)(
            y_true_all,
            y_prob_all,
            metrics=["accuracy", "error_ps", "rejection_rate"],
            y_predset=extra_output["y_predset"],
        )
    )
