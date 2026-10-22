"""
Label shift conformal prediction.

Weighted split conformal prediction for the label shift setting, where the
label distribution changes between calibration and deployment while the
class-conditional distribution of the inputs stays the same.

Papers:
    Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and
    Aaditya Ramdas. "Conformal prediction under covariate shift."
    Advances in neural information processing systems 32 (2019).
    https://arxiv.org/abs/1904.06019

    Podkopaev, Alexander, and Aaditya Ramdas. "Distribution-free uncertainty
    quantification for classification under label shift." UAI (2021).
    https://arxiv.org/abs/2103.03323
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.predictionset.scores import (
    SUPPORTED_SCORE_TYPES,
    all_class_nc_scores,
    true_class_nc_scores,
)
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["LabelShift"]


def _label_shift_threshold(
    nc_scores: np.ndarray,
    cal_weights: np.ndarray,
    alpha: float,
    candidate_weight: float,
) -> float:
    """Weighted conformal threshold for one candidate label.

    The calibration scores are weighted by ``cal_weights`` and the candidate
    label contributes ``candidate_weight`` of additional mass at +inf, the
    least conforming value, since the test point's own score is unknown when
    the threshold is set (Tibshirani et al. 2019).

    Returns:
        The smallest calibration score whose cumulative normalized weight
        reaches ``1 - alpha``, or +inf when the calibration mass alone
        cannot reach it, in which case the label is always included.
    """
    order = np.argsort(nc_scores)
    sorted_scores = nc_scores[order]
    sorted_weights = cal_weights[order]
    total_weight = np.sum(sorted_weights) + candidate_weight
    if total_weight <= 0:
        return np.inf
    cum_weights = np.cumsum(sorted_weights) / total_weight
    idx = np.searchsorted(cum_weights, 1 - alpha, side="left")
    if idx >= len(sorted_scores):
        return np.inf
    return float(sorted_scores[idx])


def _compute_label_shift_thresholds(
    nc_scores: np.ndarray,
    y_true: np.ndarray,
    target_label_dist: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class thresholds for weighted conformal prediction under label shift.

    Each calibration example with label y is weighted by the label ratio
    w(y) = target_label_dist[y] / source_label_dist[y], where the source
    distribution is the empirical label distribution of the calibration set.
    The threshold for candidate label y reserves an additional mass w(y) at
    +inf for the test point itself, so every class gets its own threshold.

    Returns:
        Tuple of (thresholds, label_ratios, source_label_dist), each of
        shape (num_classes,).
    """
    num_classes = len(target_label_dist)
    source_label_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)

    label_ratios = np.zeros(num_classes)
    for k in range(num_classes):
        if source_label_dist[k] > 0:
            label_ratios[k] = target_label_dist[k] / source_label_dist[k]
        elif target_label_dist[k] > 0:
            # The class never appears in the calibration set but has mass
            # under the target distribution, so no calibration score can
            # speak for it; its candidate weight is unbounded and the class
            # ends up always included.
            label_ratios[k] = np.inf

    cal_weights = label_ratios[y_true]
    thresholds = np.array(
        [
            _label_shift_threshold(nc_scores, cal_weights, alpha, label_ratios[k])
            for k in range(num_classes)
        ]
    )
    return thresholds, label_ratios, source_label_dist


class LabelShift(SetPredictor):
    """Label shift conformal prediction for multiclass classification.

    Under label shift, the label distribution changes between the
    calibration distribution P and the deployment (target) distribution Q,
    while the class-conditional input distribution stays the same:
    Q(X, Y) = Q(Y) P(X | Y), with Q(Y) != P(Y). Split conformal prediction
    calibrated under P can then undercover under Q, because labels that
    became more common are underrepresented in the calibration set.

    This class restores marginal coverage under Q by weighted conformal
    prediction (Tibshirani et al. 2019; Podkopaev and Ramdas 2021): each
    calibration example with label y is weighted by the label ratio
    w(y) = Q(y) / P(y), and the threshold for a candidate label y is the
    weighted (1 - alpha) quantile of the calibration nonconformity scores
    with an additional mass w(y) reserved at +inf for the test point
    itself. The result is one threshold per class, stored in ``self.t``.

    The target label distribution Q(Y) is supplied by the caller to
    ``calibrate`` (for example, a known deployment prevalence). When Q(Y)
    equals the empirical calibration label distribution, every weight is 1
    and the thresholds reduce exactly to those of BaseConformal.

    Note:
        Unlike CovariateLabel, whose weights are a function of the inputs
        (density ratios Q(X) / P(X)), the weights here are a function of
        the label only. The two corrections address different shift
        assumptions and are not interchangeable.

    Papers:
        Tibshirani, Ryan J., Rina Foygel Barber, Emmanuel Candes, and
        Aaditya Ramdas. "Conformal prediction under covariate shift."
        Advances in neural information processing systems 32 (2019).

        Podkopaev, Alexander, and Aaditya Ramdas. "Distribution-free
        uncertainty quantification for classification under label shift."
        UAI (2021).

    Args:
        model: A trained base model that outputs predicted probabilities.
        alpha: Target miscoverage rate under the target distribution,
            a single float in (0, 1): P(Y not in C(X)) <= alpha for
            (X, Y) drawn from Q.
        score_type: Type of nonconformity score to use: "threshold"
            (default) or "aps". See
            :mod:`pyhealth.calib.predictionset.scores` for the exact
            formulas.
        random_state: Optional int seed for the RNG used by
            score_type="aps". Ignored for score_type="threshold".
        debug: Whether to use debug mode (processes fewer samples).

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset import LabelShift
        >>>
        >>> sleep_ds = ISRUCDataset("/data/ISRUC-I").set_task(
        ...     sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(
        ...     sleep_ds, [0.6, 0.2, 0.2])
        >>>
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...                  label_key="label", mode="multiclass")
        >>> # ... training code ...
        >>>
        >>> # The calibration labels are roughly uniform over the 5 sleep
        >>> # stages, but at deployment stage 0 is twice as common.
        >>> target_dist = [0.4, 0.15, 0.15, 0.15, 0.15]
        >>> conformal_model = LabelShift(model, alpha=0.1)
        >>> conformal_model.calibrate(cal_dataset=val_data,
        ...     target_label_dist=target_dist)
        >>>
        >>> # Evaluate on the (shifted) test set
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(
        ...     model=conformal_model).inference(test_dl,
        ...     additional_outputs=['y_predset'])
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: float,
        score_type: str = "threshold",
        random_state: int | None = None,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if model.mode != "multiclass":
            raise NotImplementedError(
                "LabelShift only supports multiclass classification"
            )
        if score_type not in SUPPORTED_SCORE_TYPES:
            raise ValueError(
                f"Unknown score_type: {score_type!r}. Supported: "
                f"{SUPPORTED_SCORE_TYPES}."
            )
        if not isinstance(alpha, (float, np.floating)) or not 0.0 < alpha < 1.0:
            raise ValueError(
                "LabelShift controls marginal coverage under the target "
                f"distribution, so alpha must be a float in (0, 1), "
                f"got {alpha!r}."
            )

        self.mode = self.model.mode

        # Freeze model parameters
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.device = model.device
        self.debug = debug
        self.score_type = score_type
        self.rng = np.random.default_rng(random_state)
        self.alpha = float(alpha)

        # Will be set during calibration
        self.t = None
        self.label_ratios_ = None
        self.source_label_dist_ = None
        self.target_label_dist_ = None

    def calibrate(
        self,
        cal_dataset: IterableDataset,
        target_label_dist,
    ):
        """Calibrate the per-class thresholds for prediction set construction.

        Args:
            cal_dataset: Calibration set (held-out data from the source
                distribution P).
            target_label_dist: Label distribution Q(Y) of the target
                distribution, one entry per class. It is normalized to sum
                to 1 before use.
        """
        cal_dataset_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            debug=self.debug,
        )

        y_prob = cal_dataset_dict["y_prob"]
        y_true = np.asarray(cal_dataset_dict["y_true"], dtype=np.int64)
        _, num_classes = y_prob.shape

        target_label_dist = np.asarray(target_label_dist, dtype=np.float64)
        if target_label_dist.shape != (num_classes,):
            raise ValueError(
                f"target_label_dist must have one entry per class "
                f"({num_classes}), got shape {target_label_dist.shape}."
            )
        if not np.all(np.isfinite(target_label_dist)) or np.any(
            target_label_dist < 0
        ):
            raise ValueError(
                "target_label_dist entries must be finite and non-negative."
            )
        total = target_label_dist.sum()
        if total <= 0:
            raise ValueError("target_label_dist must have positive total mass.")
        target_label_dist = target_label_dist / total

        # Non-conformity scores of the true labels (higher = less conforming)
        nc_scores = true_class_nc_scores(
            y_prob, y_true, score_type=self.score_type, rng=self.rng
        )

        thresholds, label_ratios, source_label_dist = (
            _compute_label_shift_thresholds(
                nc_scores, y_true, target_label_dist, self.alpha
            )
        )

        self.t = torch.tensor(thresholds, device=self.device)
        self.label_ratios_ = label_ratios
        self.source_label_dist_ = source_label_dist
        self.target_label_dist_ = target_label_dist

        if self.debug:
            print(f"Calibrated thresholds: {self.t}")

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation with prediction set construction.

        Returns:
            Dictionary with all results from base model, plus:
                - y_predset: Boolean tensor indicating which classes
                    are in the prediction set
        """
        if self.t is None:
            raise RuntimeError(
                "Model must be calibrated before inference. "
                "Call calibrate() first."
            )

        pred = self.model(**kwargs)

        y_prob = pred["y_prob"].detach().cpu().numpy()
        nc_scores = all_class_nc_scores(
            y_prob, score_type=self.score_type, rng=self.rng
        )
        nc_scores = torch.as_tensor(
            nc_scores, device=pred["y_prob"].device, dtype=pred["y_prob"].dtype
        )
        # Include class y if its NC score <= its class threshold self.t[y]
        pred["y_predset"] = nc_scores <= self.t

        return pred
