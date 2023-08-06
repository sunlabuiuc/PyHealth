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
from pyhealth.calib.predictionset.favmac.core import FavMac_GreedyRatio
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
        return np.sum(S * self.values)

    def util_call(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=1000) -> float:
        assert Y is None or pred is None
        if pred is not None:
            return self.naive_call(S * pred) # This is because of additivity.
        if Y is not None: return self.naive_call(S * Y)
        return self.naive_call(S)

    def cost_call(self, S: np.ndarray, Y:np.ndarray) -> float:
        return self.naive_call(S * (1-Y))

    def proxy_call(self, S: np.ndarray, pred: np.ndarray, target_cost: float=None) -> float:
        return self.naive_call(S * (1-pred))

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

    This is a prediction-set constructor for multi-label classification problems.
    FavMac could control the cost/risk while realizing high value on the prediction set.

    Value and cost functions are functions in the form of :math:`V(S;Y)` or :math:`C(S;Y)`,
    with S being the prediction set and Y being the label.
    For example, a classical cost function would be "numebr of false positives".
    Denote the ``target_cost`` as
    :math:`c`,
    if ``delta=None``, FavMac controls the expected cost in the following sense:

    :math:`\\mathbb{E}[C(S_{N+1};Y_{N+1}] \\leq c`.

    Otherwise, FavMac controls the violation probability in the following sense:

    :math:`\\mathbb{P}\\{C(S_{N+1};Y_{N+1})>c\\}\\leq delta`.

    Right now, this FavMac implementation only supports additive value and cost functions (unlike the
    implementation associated with [1]).
    That is, the value function is specified by the weights ``value_weights`` and the cost function
    is specified by ``cost_weights``.
    With :math:`k` denoting classes, the cost function is then computed as

    :math:`C(S;Y,w) = \\sum_{k} (1-Y_k)S_k w_k`

    Similarly, the value function is computed as

    :math:`V(S;Y,w) = \\sum_{k} Y_k S_k w_k`.

    Papers:

        [1] Lin, Zhen, Shubhendu Trivedi, Cao Xiao, and Jimeng Sun.
        "Fast Online Value-Maximizing Prediction Sets with Conformal Cost Control (FavMac)."
        ICML 2023.

        [2] Fisch, Adam, Tal Schuster, Tommi Jaakkola, and Regina Barzilay.
        "Conformal prediction sets with limited false positives."
        ICML 2022.

    Args:
        model (BaseModel): A trained model.
        value_weights (Union[float, np.ndarray]):
            weights for the value function. See description above.
            Defaults to 1.
        cost_weights (Union[float, np.ndarray]):
            weights for the cost function. See description above.
            Defaults to 1.
        target_cost (float): Target cost.
            When cost_weights is set to 1, this is essentially the number of false positive.
            Defaults to 1.
        delta (float): Violation target (in violation control).
            Defaults to None (which means expectation control instead of violation control).

    Examples:
        >>> from pyhealth.calib.predictionset import FavMac
        >>> from pyhealth.datasets import (MIMIC3Dataset, get_dataloader,split_by_patient)
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.tasks import drug_recommendation_mimic3_fn
        >>> from pyhealth.trainer import get_metrics_fn
        >>> base_dataset = MIMIC3Dataset(
        ...     root="/srv/scratch1/data/physionet.org/files/mimiciii/1.4",
        ...     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...     code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     refresh_cache=False)
        >>> sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
        >>> train_data, val_data, test_data = split_by_patient(sample_dataset, [0.6, 0.2, 0.2])
        >>> model = Transformer(dataset=sample_dataset, feature_keys=["conditions", "procedures"],
        ...             label_key="drugs", mode="multilabel")
        >>> # ... Train the model here ...
        >>> # Try to control false positive to <=3
        >>> cal_model = FavMac(model, target_cost=3, delta=None)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(
        ... test_dl, additional_outputs=["y_predset"])
        >>> print(get_metrics_fn(cal_model.mode)(
        ...     y_true_all, y_prob_all, metrics=['tp', 'fp'],
        ...     y_predset=extra_output["y_predset"])) # We get FP~=3
        {'tp': 0.5049893086243763, 'fp': 2.8442622950819674}
    """

    def __init__(
        self,
        model: BaseModel,
        value_weights: Union[float, np.ndarray] = 1.,
        cost_weights: Union[float, np.ndarray] = 1.,
        target_cost: float = 1., delta:float = None,
        debug=False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if model.mode != "multilabel":
            raise NotImplementedError()
        self.mode = self.model.mode  # multilabel
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = model.device
        self.debug = debug


        self._cost_weights = cost_weights
        self._value_weights = value_weights
        self.target_cost = target_cost
        self.delta = delta

    def calibrate(self, cal_dataset):
        """Calibrate the cost-control procedure.

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
    from pyhealth.datasets import (MIMIC3Dataset, get_dataloader,
                                   split_by_patient)
    from pyhealth.models import Transformer
    from pyhealth.tasks import drug_recommendation_mimic3_fn
    from pyhealth.trainer import get_metrics_fn

    base_dataset = MIMIC3Dataset(
        root="/srv/scratch1/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
    train_data, val_data, test_data = split_by_patient(sample_dataset, [0.6, 0.2, 0.2])
    model = Transformer(dataset=sample_dataset, feature_keys=["conditions", "procedures"],
                        label_key="drugs", mode="multilabel")
    # ... Train the model here ...
    # calibrate the prediction sets with FavMac
    cal_model = FavMac(model, cost_weights=1., target_cost=3, delta=None)
    cal_model.calibrate(cal_dataset=val_data)
    # Evaluate
    from pyhealth.trainer import Trainer

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(
        test_dl, additional_outputs=["y_predset"]
    )
    print(
        get_metrics_fn(cal_model.mode)(
            y_true_all, y_prob_all, metrics=['tp', 'fp'],
            y_predset=extra_output["y_predset"],
        )
    )
