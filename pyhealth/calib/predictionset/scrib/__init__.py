"""
SCRIB: Set-classifier with Class-specific Risk Bounds

Implementation based on https://github.com/zlin7/scrib

"""

import time
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

from . import quicksearch as qs

OVERALL_LOSSFUNC = "overall"
CLASSPECIFIC_LOSSFUNC = "classspec"

__all__ = ["SCRIB"]


class _CoordDescent:
    def __init__(
        self,
        model_output,
        labels,
        rks,
        loss_func=OVERALL_LOSSFUNC,
        loss_kwargs=None,
        restart_n=1000,
        restart_range=0.1,
        init_range=None,
        verbose=False,
    ):
        self.N, self.K = model_output.shape

        # quantities useful for loss eval
        self.loss_name = loss_func
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_kwargs = loss_kwargs
        if self.loss_name == OVERALL_LOSSFUNC:
            assert isinstance(rks, float)
        elif rks is not None:
            rks = np.asarray(rks)
        self.idx2rnk = np.asarray(
            pd.DataFrame(model_output).rank(ascending=True), np.int32
        )
        if np.min(self.idx2rnk) == 1:
            self.idx2rnk -= 1
        self.rnk2idx = np.asarray(np.argsort(model_output, axis=0), np.int32)
        if len(labels.shape) == 2:
            # one-hot -> class indices
            labels = np.argmax(labels, 1)
        self.labels = np.asarray(labels, np.int32)
        self.max_classes = np.argmax(model_output, 1)
        self.rks = rks

        self.model_output = model_output
        self.restart_n = restart_n
        self.restart_range = restart_range
        self.init_range = init_range or (int(np.ceil(self.N / 2)), self.N - 1)
        self.verbose = verbose

    def _search(self, ps):
        _search_fn = {
            CLASSPECIFIC_LOSSFUNC: qs.coord_desc_classspecific,
            OVERALL_LOSSFUNC: qs.coord_desc_overall,
        }[self.loss_name]
        return _search_fn(
            self.idx2rnk,
            self.rnk2idx,
            self.labels,
            self.max_classes,
            ps,
            self.rks,
            **self.loss_kwargs,
        )

    def _loss_eval(self, ps):
        _loss_fn = {
            CLASSPECIFIC_LOSSFUNC: qs.loss_classspecific,
            OVERALL_LOSSFUNC: qs.loss_overall,
        }[self.loss_name]
        return _loss_fn(
            self.idx2rnk,
            self.rnk2idx,
            self.labels,
            self.max_classes,
            ps,
            self.rks,
            **self.loss_kwargs,
        )

    def _p2t(self, p):
        # Translate ranks to thresholds
        return [self.model_output[self.rnk2idx[p[k], k], k] for k in range(self.K)]

    def _sample_new_loc(self, old_p, restart_range=0.1):
        diff = np.random.uniform(-restart_range, restart_range, self.K)
        new_p = old_p.copy()
        for k in range(self.K):
            new_p[k] = max(min(int(new_p[k] + diff[k] * self.N), self.N - 1), 0)
        return new_p

    def search_once(self, seed=7):
        def print_(s):
            if self.verbose:
                print(s)

        np.random.seed(seed)
        best_ps = np.random.randint(*self.init_range, self.K)

        st = time.time()
        best_loss, best_ps, _ = self._search(best_ps)
        ed1 = time.time()
        if self.restart_n > 0:
            keep_going = True
            while keep_going:
                keep_going = False
                curr_restart_best_loss, curr_restart_best_ps = np.inf, None

                for _ in range(self.restart_n):
                    # Restart in neighborhood
                    new_ps_ = self._sample_new_loc(best_ps, self.restart_range)
                    loss_ = self._loss_eval(new_ps_)
                    if loss_ < best_loss:
                        print_(
                            "Neighborhood has a better loc with "
                            f"loss={loss_} < {best_loss}"
                        )
                        best_loss, best_ps, _ = self._search(new_ps_)

                        keep_going = True
                        break
                    elif loss_ < curr_restart_best_loss:
                        curr_restart_best_loss, curr_restart_best_ps = loss_, new_ps_
                if not keep_going:
                    print_(
                        f"Tried {curr_restart_best_ps} vs {best_ps}, "
                        f"loss:{curr_restart_best_loss} > {best_loss}"
                    )
        ed2 = time.time()
        print_(f"{ed1-st:.3f} + {ed2-ed1:.3f} seconds")
        return self._p2t(best_ps), best_loss

    @classmethod
    def search(
        cls,
        prob: np.ndarray,
        label: np.ndarray,
        rks: Union[float, np.ndarray],
        loss_func,
        B: int = 10,
        **kwargs,
    ):
        # label is not one-hot
        best_loss, best_ts = np.inf, None
        searcher = cls(prob, label, rks, loss_func=loss_func, **kwargs)
        for seed in range(B):
            np.random.seed(seed)
            ts, _l = searcher.search_once(seed + 1)
            print(f"{seed}: loss={_l}")
            if _l < best_loss:
                best_loss, best_ts = _l, ts
        return best_ts, best_loss


class SCRIB(SetPredictor):
    """SCRIB: Set-classifier with Class-specific Risk Bounds

    This is a prediction-set constructor for multi-class classification problems.
    SCRIB tries to control class-specific risk while minimizing the ambiguity.
    To to this, it selects class-specific thresholds for the predictions, on a calibration set.


    If ``risk`` is a float (say 0.1), SCRIB controls the overall risk:
    :math:`\\mathbb{P}\\{Y \\not \\in C(X) | |C(X)| = 1\\}\\leq risk`.
    If ``risk`` is an array (say `np.asarray([0.1] * 5)`), SCRIB controls the class specific risks:
    :math:`\\mathbb{P}\\{Y \\not \\in C(X) | Y=k \\land |C(X)| = 1\\}\\leq risk_k`
    Here, :math:`C(X)` denotes the final prediction set.

    Paper:

        Lin, Zhen, Lucas Glass, M. Brandon Westover, Cao Xiao, and Jimeng Sun.
        "SCRIB: Set-classifier with Class-specific Risk Bounds for Blackbox Models."
        AAAI 2022.

    Args:
        model (BaseModel): A trained model.
        risk (Union[float, np.ndarray]): risk targets.
        loss_kwargs (dict, optional): Additional loss parameters (including hyperparameters).
            It could contain the following float/int hyperparameters:
                lk: The coefficient for the loss term associated with risk violation penalty.
                    The higher the lk, the more penalty on risk violation (likely higher ambiguity).
                fill_max: Whether to fill the class with max predicted score
                    when no class exceeds the threshold. In other words, if fill_max,
                    the null region will be filled with max-prediction class.
            Defaults to {'lk': 1e4, 'fill_max': False}
        fill_max (bool, optional): Whether to fill the empty prediction set with the max-predicted class.
            Defaults to True.


    Examples:
        >>> from pyhealth.data import ISRUCDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset import SCRIB
        >>> from pyhealth.trainer import get_metrics_fn
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate the set classifier, with different class-specific risk targets
        >>> cal_model = SCRIB(model, [0.2, 0.3, 0.1, 0.2, 0.1])
        >>> # Note that we used the test set here because ISRUCDataset has relatively few
        >>> # patients, and calibration set should be different from the validation set
        >>> # if the latter is used to pick checkpoint. In general, the calibration set
        >>> # should be something exchangeable with the test set. Please refer to the paper.
        >>> cal_model.calibrate(cal_dataset=test_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(test_dl, additional_outputs=['y_predset'])
        >>> print(get_metrics_fn(cal_model.mode)(
        ... y_true_all, y_prob_all, metrics=['accuracy', 'error_ps', 'rejection_rate'],
        ... y_predset=extra_output['y_predset'])
        ... )
        {'accuracy': 0.709843241966832, 'rejection_rate': 0.6381305287631919,
        'error_ps': array([0.32161874, 0.36654135, 0.11461734, 0.23728814, 0.14993925])}
    """

    def __init__(
        self,
        model: BaseModel,
        risk: Union[float, np.ndarray],
        loss_kwargs: dict = None,
        debug=False,
        fill_max=True,
        **kwargs,
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

        if isinstance(risk, float):
            self.loss_name = OVERALL_LOSSFUNC
        else:
            risk = np.asarray(risk)
            self.loss_name = CLASSPECIFIC_LOSSFUNC
        self.risk = risk
        if loss_kwargs is None:
            loss_kwargs = {"lk": 1e4, "fill_max": fill_max}
        self.loss_kwargs = loss_kwargs

        self.t = None

    def calibrate(self, cal_dataset):
        """Calibrate/Search for the thresholds used to construct the prediction set.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        """
        cal_dataset = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_prob", "y_true"], debug=self.debug
        )
        if self.loss_name == CLASSPECIFIC_LOSSFUNC:
            assert len(self.risk) == cal_dataset["y_prob"].shape[1]
        best_ts, _ = _CoordDescent.search(
            cal_dataset["y_prob"],
            cal_dataset["y_true"],
            self.risk,
            self.loss_name,
            loss_kwargs=self.loss_kwargs,
            verbose=self.debug,
        )
        self.t = torch.nn.Parameter(torch.tensor(best_ts, device=self.device))

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :return: A dictionary with all results from the base model, with the following updates:

                    y_predset: a bool tensor representing the prediction for each class.
        :rtype: Dict[str, torch.Tensor]
        """
        ret = self.model(**kwargs)
        ret["y_predset"] = ret["y_prob"] > self.t
        return ret


if __name__ == "__main__":
    from pyhealth.calib.predictionset import SCRIB
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
