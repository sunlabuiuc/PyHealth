"""
Histogram Binning.

Implementation based on https://github.com/aigen/df-posthoc-calibration

"""
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Subset

from pyhealth.calib.base_classes import PostHocCalibrator
from pyhealth.calib.utils import LogLoss, one_hot_np, prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["HistogramBinning"]


def _nudge(matrix, delta):
    return (matrix + np.random.uniform(low=0, high=delta, size=matrix.shape)) / (
        1 + delta
    )


def _bin_points(scores, bin_edges):
    assert bin_edges is not None, "Bins have not been defined"
    scores = scores.squeeze()
    assert np.size(scores.shape) < 2, "scores should be a 1D vector or singleton"
    scores = np.reshape(scores, (scores.size, 1))
    bin_edges = np.reshape(bin_edges, (1, bin_edges.size))
    return np.sum(scores > bin_edges, axis=1)


def _get_uniform_mass_bins(probs, n_bins):
    assert probs.size >= n_bins, "Fewer points than bins"

    probs_sorted = np.sort(probs)

    # split probabilities into groups of approx equal size
    groups = np.array_split(probs_sorted, n_bins)
    bin_upper_edges = list()

    for cur_group in range(n_bins - 1):
        bin_upper_edges += [max(groups[cur_group])]
    bin_upper_edges += [np.inf]

    return np.array(bin_upper_edges)


class HB_binary(object):
    def __init__(self, n_bins=15):
        ### Hyperparameters
        self.delta = 1e-10
        self.n_bins = n_bins

        ### Parameters to be learnt
        self.bin_upper_edges = None
        self.mean_pred_values = None
        self.num_calibration_examples_in_bin = None

        ### Internal variables
        self.fitted = False

    def fit(self, y_score, y):
        assert self.n_bins is not None, "Number of bins has to be specified"
        y_score = y_score.squeeze()
        y = y.squeeze()
        assert y_score.size == y.size, "Check dimensions of input matrices"
        assert (
            y.size >= self.n_bins
        ), "Number of bins should be less than the number of calibration points"

        ### All required (hyper-)parameters have been passed correctly
        ### Uniform-mass binning/histogram binning code starts below

        # delta-randomization
        y_score = _nudge(y_score, self.delta)

        # compute uniform-mass-bins using calibration data
        self.bin_upper_edges = _get_uniform_mass_bins(y_score, self.n_bins)

        # assign calibration data to bins
        bin_assignment = _bin_points(y_score, self.bin_upper_edges)

        # compute bias of each bin
        self.num_calibration_examples_in_bin = np.zeros([self.n_bins, 1])
        self.mean_pred_values = np.empty(self.n_bins)
        for i in range(self.n_bins):
            bin_idx = bin_assignment == i
            self.num_calibration_examples_in_bin[i] = sum(bin_idx)

            # nudge performs delta-randomization
            if sum(bin_idx) > 0:
                self.mean_pred_values[i] = _nudge(y[bin_idx].mean(), self.delta)
            else:
                self.mean_pred_values[i] = _nudge(0.5, self.delta)

        # check that my code is correct
        assert np.sum(self.num_calibration_examples_in_bin) == y.size

        # histogram binning done
        self.fitted = True
        return self

    def predict_proba(self, y_score):
        assert self.fitted is True, "Call HB_binary.fit() first"
        y_score = y_score.squeeze()

        # delta-randomization
        y_score = _nudge(y_score, self.delta)

        # assign test data to bins
        y_bins = _bin_points(y_score, self.bin_upper_edges)

        # get calibrated predicted probabilities
        y_pred_prob = self.mean_pred_values[y_bins]
        return y_pred_prob


# HB_toplabel is removed as it has no use in our tasks


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

    :param model: A trained base model.
    :type model: BaseModel

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, get_dataloader, split_by_patient
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.calibration import HistogramBinning
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> # Calibrate
        >>> cal_model = HistogramBinning(model)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> print(Trainer(model=cal_model, metrics=['cwECEt_adapt', 'accuracy']).evaluate(test_dl))
        {'accuracy': 0.7189072348464207, 'cwECEt_adapt': 0.04455814993598299}
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
        self.calib = None

    def calibrate(self, cal_dataset: Subset, nbins: int = 15):
        """Calibrate the base model using a calibration dataset.

        :param cal_dataset: Calibration set.
        :type cal_dataset: Subset
        :param nbins: number of bins to use, defaults to 15
        :type nbins: int, optional
        """
        _cal_data = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_true", "y_prob"], debug=self.debug
        )
        if self.num_classes is None:
            self.num_classes = _cal_data["y_prob"].shape[1]
        if self.mode == "binary":
            self.calib = [
                HB_binary(nbins).fit(
                    _cal_data["y_prob"][:, 0], _cal_data["y_true"][:, 0]
                )
            ]
        else:
            self.calib = []
            y_true = _cal_data["y_true"]
            if len(y_true.shape) == 1:
                y_true = one_hot_np(y_true, self.num_classes)
            for k in range(self.num_classes):
                self.calib.append(
                    HB_binary().fit(_cal_data["y_prob"][:, k], y_true[:, k])
                )

    def forward(self, normalization="sum", **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :param normalization: how to normalize the calibrated probability.
            Defaults to 'sum' (and only 'sum' is supported for now).
        :type normalization: str, optional
        :param **kwargs: Additional arguments to the base model.

        :return:  A dictionary with all results from the base model, with the following modified:

            ``y_prob``: calibrated predicted probabilities.
            ``loss``: Cross entropy loss  with the new y_prob.
        :rtype: Dict[str, torch.Tensor]
        """
        assert normalization is None or normalization == "sum"
        ret = self.model(**kwargs)
        y_prob = ret["y_prob"].cpu().numpy()
        for k in range(self.num_classes):
            y_prob[:, k] = self.calib[k].predict_proba(y_prob[:, k])
        if normalization == "sum" and self.mode == "multiclass":
            y_prob = y_prob / y_prob.sum(1)[:, np.newaxis]
        ret["y_prob"] = torch.tensor(y_prob, dtype=torch.float, device=self.device)
        if self.mode == "multiclass":
            criterion = LogLoss()
        else:
            criterion = torch.nn.functional.binary_cross_entropy
        ret["loss"] = criterion(ret["y_prob"], ret["y_true"])
        ret.pop("logit")
        return ret


if __name__ == "__main__":
    from pyhealth.datasets import ISRUCDataset, get_dataloader, split_by_patient
    from pyhealth.models import SparcNet
    from pyhealth.tasks import sleep_staging_isruc_fn
    from pyhealth.calib.calibration import HistogramBinning

    sleep_ds = ISRUCDataset(
        root="/srv/local/data/trash/",
        dev=True,
    ).set_task(sleep_staging_isruc_fn)
    train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
    model = SparcNet(
        dataset=sleep_ds,
        feature_keys=["signal"],
        label_key="label",
        mode="multiclass",
    )
    # ... Train the model here ...
    # Calibrate
    cal_model = HistogramBinning(model)
    cal_model.calibrate(cal_dataset=val_data)
    # Evaluate
    from pyhealth.trainer import Trainer

    test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
    print(
        Trainer(model=cal_model, metrics=["cwECEt_adapt", "accuracy"]).evaluate(test_dl)
    )
