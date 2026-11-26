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

from pyhealth.calib.predictionset.base_conformal import (
    BaseConformal,
    _query_weighted_quantile,
)
from pyhealth.calib.predictionset.scores import true_class_nc_scores
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

    This is a thin adapter over the shared weighted-quantile helper in
    ``base_conformal``, not a second implementation of it. The covariate
    helper is deliberately not reused here: it works on conformity
    scores with the test mass reserved at -inf, and flipping its sign
    gives the alpha quantile from the bottom of the conformity scale,
    which is not the smallest nonconformity score whose cumulative
    weight reaches ``1 - alpha`` once a candidate mass sits at +inf.

    Returns:
        The smallest calibration score whose cumulative normalized weight
        reaches ``1 - alpha``, or +inf when the calibration mass alone
        cannot reach it, in which case the label is always included.
    """
    return _query_weighted_quantile(
        nc_scores,
        1 - alpha,
        cal_weights,
        candidate_weight=candidate_weight,
        return_inf_on_exhaustion=True,
    )


def _validate_label_distribution(dist, num_classes: int, name: str) -> np.ndarray:
    """Validate a label distribution and return it normalized to sum to 1.

    Entries are scaled by their maximum before normalizing, so very large
    finite values cannot overflow the sum.
    """
    dist = np.asarray(dist, dtype=np.float64)
    if dist.ndim != 1 or dist.shape[0] != num_classes:
        raise ValueError(
            f"{name} must have one entry per class ({num_classes}), "
            f"got shape {dist.shape}."
        )
    if not np.all(np.isfinite(dist)):
        raise ValueError(f"{name} entries must be finite.")
    if np.any(dist < 0):
        raise ValueError(f"{name} entries must be non-negative.")
    scale = dist.max()
    if scale <= 0:
        raise ValueError(f"{name} must have positive total mass.")
    dist = dist / scale
    return dist / dist.sum()


def _validate_calibration_labels(
    y_true: np.ndarray, num_rows: int, num_classes: int
) -> np.ndarray:
    """Validate calibration labels and return them as an int64 array.

    Raises an informative ValueError for an empty calibration set, labels
    that are not one integer per prediction row, or labels outside
    [0, num_classes), instead of surfacing low-level indexing, casting, or
    division errors from later in the calibration code.
    """
    if y_true.ndim == 0 or len(y_true) == 0:
        raise ValueError("calibration dataset is empty.")
    if y_true.ndim != 1 or len(y_true) != num_rows:
        raise ValueError(
            "calibration labels must be one-dimensional with one label "
            f"per prediction row ({num_rows}), got shape {y_true.shape}."
        )
    if not np.issubdtype(y_true.dtype, np.integer):
        if not np.all(np.isfinite(y_true)) or not np.all(
            y_true == np.floor(y_true)
        ):
            raise ValueError("calibration labels must be integers.")
        y_true = y_true.astype(np.int64)
    if y_true.min() < 0 or y_true.max() >= num_classes:
        raise ValueError(
            f"calibration labels must lie in [0, {num_classes}), got "
            f"labels from {y_true.min()} to {y_true.max()}."
        )
    return y_true


def _compute_label_shift_thresholds(
    nc_scores: np.ndarray,
    y_true: np.ndarray,
    target_label_dist: np.ndarray,
    source_label_dist: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class thresholds for weighted conformal prediction under label shift.

    Each calibration example with label y is weighted by the label ratio
    w(y) = target_label_dist[y] / source_label_dist[y]. The threshold for
    candidate label y reserves an additional mass w(y) at +inf for the
    test point itself, so every class gets its own threshold. Both
    distributions are inputs: the caller decides whether the source
    distribution is a known population distribution or the empirical
    calibration distribution (a plug-in estimate).

    Returns:
        Tuple of (thresholds, label_ratios), each of shape (num_classes,).
    """
    num_classes = len(target_label_dist)

    label_ratios = np.zeros(num_classes)
    for k in range(num_classes):
        if source_label_dist[k] > 0:
            label_ratios[k] = target_label_dist[k] / source_label_dist[k]
        elif target_label_dist[k] > 0:
            # The class has no mass under the source distribution but has
            # mass under the target, so no calibration score can speak for
            # it; its candidate weight is unbounded and the class ends up
            # always included.
            label_ratios[k] = np.inf

    cal_weights = label_ratios[y_true]
    thresholds = np.array(
        [
            _label_shift_threshold(nc_scores, cal_weights, alpha, label_ratios[k])
            for k in range(num_classes)
        ]
    )
    return thresholds, label_ratios


class LabelShift(BaseConformal):
    """Label shift conformal prediction for multiclass classification.

    This subclasses :class:`BaseConformal` and inherits its ``forward``:
    per-class thresholds stored in ``self.t`` broadcast against every
    candidate class, so only ``calibrate`` differs from the base class.
    That difference is intentional and includes the signature:
    ``calibrate`` requires ``target_label_dist``, unlike
    ``BaseConformal.calibrate``, so a LabelShift instance cannot be
    calibrated through a generic BaseConformal call that passes only a
    calibration dataset. Label shift calibration inherently needs the
    target distribution, so no default is supplied.

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
    ``calibrate`` (for example, a known deployment prevalence). When the
    source distribution is the empirical calibration distribution (the
    default) and Q(Y) equals it, every weight is 1 for classes present
    in calibration, and their thresholds reduce exactly to those of
    BaseConformal. A class absent from calibration is the exception:
    its label ratio is 0, so it reserves no candidate mass, and its
    threshold is computed from the N calibration scores alone instead
    of the N + 1 scores BaseConformal uses. It can therefore sit one
    rank below the BaseConformal threshold, including the case where
    BaseConformal returns +inf while the absent class gets the largest
    calibration score.

    The finite-sample coverage guarantee under Q holds when the label
    ratios are known, that is, when both P(Y) and Q(Y) are known. The
    source distribution P(Y) can be passed to ``calibrate`` directly; by
    default it is estimated from the calibration labels instead, which
    makes the ratios plug-in estimates and the coverage under Q
    approximate rather than exact.

    A class with zero target mass contributes no weight: its calibration
    examples do not count toward any threshold, and its own threshold is
    the weighted quantile of the remaining mass, so the class is not
    excluded from prediction sets outright, it just needs one of the most
    conforming scores to appear.

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
        if model.mode != "multiclass":
            raise NotImplementedError(
                "LabelShift only supports multiclass classification"
            )
        if not isinstance(alpha, (float, np.floating)) or not 0.0 < alpha < 1.0:
            raise ValueError(
                "LabelShift controls marginal coverage under the target "
                f"distribution, so alpha must be a float in (0, 1), "
                f"got {alpha!r}."
            )
        super().__init__(
            model,
            alpha,
            score_type=score_type,
            random_state=random_state,
            debug=debug,
            **kwargs,
        )
        self.alpha = float(alpha)

        # Will be set during calibration
        self.label_ratios_ = None
        self.source_label_dist_ = None
        self.target_label_dist_ = None

    def calibrate(
        self,
        cal_dataset: IterableDataset,
        target_label_dist,
        source_label_dist=None,
    ):
        """Calibrate the per-class thresholds for prediction set construction.

        Args:
            cal_dataset: Calibration set (held-out data from the source
                distribution P).
            target_label_dist: Label distribution Q(Y) of the target
                distribution, one entry per class. It is normalized to sum
                to 1 before use.
            source_label_dist: Label distribution P(Y) of the source
                distribution, one entry per class, normalized before use.
                If None (default), P(Y) is estimated from the calibration
                labels; the label ratios are then plug-in estimates and
                coverage under Q is approximate. Pass a known P(Y) for the
                exact finite-sample guarantee. Calibration labels with
                zero probability under a supplied P(Y) are rejected as
                inconsistent with it. A class gets an infinite threshold
                when the calibration mass alone cannot reach ``1 - alpha``
                of the total weight including that class's candidate mass.
                That happens when its source probability is zero while its
                target probability is positive (its ratio is infinite),
                and it can also happen with a positive source probability
                if its candidate ratio is large relative to the total
                calibration weight, so a positive source mass alone does
                not guarantee a finite threshold.
        """
        try:
            num_cal = len(cal_dataset)
        except TypeError:
            num_cal = None
        if num_cal == 0:
            raise ValueError("calibration dataset is empty.")

        cal_dataset_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            debug=self.debug,
        )

        y_prob = cal_dataset_dict["y_prob"]
        num_rows, num_classes = y_prob.shape
        y_true = _validate_calibration_labels(
            np.asarray(cal_dataset_dict["y_true"]), num_rows, num_classes
        )

        target_label_dist = _validate_label_distribution(
            target_label_dist, num_classes, "target_label_dist"
        )
        if source_label_dist is None:
            source_label_dist = np.bincount(y_true, minlength=num_classes) / len(
                y_true
            )
        else:
            source_label_dist = _validate_label_distribution(
                source_label_dist, num_classes, "source_label_dist"
            )
            inconsistent = np.unique(y_true[source_label_dist[y_true] == 0])
            if len(inconsistent) > 0:
                raise ValueError(
                    "calibration labels are inconsistent with the supplied "
                    "source_label_dist: labels "
                    f"{inconsistent.tolist()} appear in the calibration "
                    "data but have zero probability under it."
                )

        # Non-conformity scores of the true labels (higher = less conforming)
        nc_scores = true_class_nc_scores(
            y_prob, y_true, score_type=self.score_type, rng=self.rng
        )

        thresholds, label_ratios = _compute_label_shift_thresholds(
            nc_scores, y_true, target_label_dist, source_label_dist, self.alpha
        )

        self.t = torch.tensor(thresholds, device=self.device)
        self.label_ratios_ = label_ratios
        self.source_label_dist_ = source_label_dist
        self.target_label_dist_ = target_label_dist

        if self.debug:
            print(f"Calibrated thresholds: {self.t}")
