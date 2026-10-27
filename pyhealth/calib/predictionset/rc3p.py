"""
RC3P: Rank Calibrated Class-conditional Conformal Prediction.

Paper:

    Shi, Yuanjie, Subhankar Ghosh, Taha Belkhouja, Janardhan Rao
    Doppa, and Yan Yan. "Conformal Prediction for Class-wise Coverage
    via Augmented Label Rank Calibration." NeurIPS (2024).
    https://arxiv.org/abs/2406.06818

"""

import numpy as np
import torch
from torch.utils.data import Subset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.predictionset.scores import (
    SUPPORTED_SCORE_TYPES,
    all_class_nc_scores,
    true_class_nc_scores,
)
from pyhealth.calib.utils import prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["RC3P"]


def _candidate_ranks(y_prob: np.ndarray) -> np.ndarray:
    """Rank of every candidate label in each row of probabilities.

    The rank of label y is the number of classes whose predicted
    probability is at least the probability of y, following the
    paper's rank definition. Tied labels therefore share the worse
    rank: for probabilities [0.2, 0.2, 0.6] the ranks are [3, 3, 1].

    Computed from sorted probabilities, so the memory is N x K, not
    N x K x K: the rank of y is K minus the number of probabilities
    strictly below the probability of y.
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    num_classes = y_prob.shape[1]
    sorted_probs = np.sort(y_prob, axis=1)
    ranks = np.empty(y_prob.shape, dtype=np.int64)
    for i, row in enumerate(y_prob):
        below = np.searchsorted(sorted_probs[i], row, side="left")
        ranks[i] = num_classes - below
    return ranks


def _rank_calibration(
    rank_y_prob: np.ndarray, rank_y_true: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Select per-class rank cutoffs and adjusted miscoverage rates.

    For each class y, the top-k error is the fraction of rank-subset
    samples with true label y whose true-label rank exceeds k. The
    cutoff is the smallest k with top-k error below alpha, and the
    adjusted miscoverage is alpha minus that error. A class absent
    from the rank subset gets cutoff K (no rank filtering) and
    adjusted miscoverage alpha.
    """
    rank_y_prob = np.asarray(rank_y_prob, dtype=np.float64)
    rank_y_true = np.asarray(rank_y_true, dtype=np.int64)
    num_classes = rank_y_prob.shape[1]
    ranks = _candidate_ranks(rank_y_prob)
    true_ranks = ranks[np.arange(len(rank_y_true)), rank_y_true]
    k_hat = np.full(num_classes, num_classes, dtype=np.int64)
    alpha_hat = np.full(num_classes, alpha, dtype=np.float64)
    for y in range(num_classes):
        class_ranks = true_ranks[rank_y_true == y]
        if len(class_ranks) == 0:
            continue
        for k in range(1, num_classes + 1):
            eps = float(np.mean(class_ranks > k))
            if eps < alpha:
                k_hat[y] = k
                alpha_hat[y] = alpha - eps
                break
    return k_hat, alpha_hat


def _predset_mask(
    nc_scores: np.ndarray,
    thresholds: np.ndarray,
    candidate_ranks: np.ndarray,
    k_hat: np.ndarray,
) -> np.ndarray:
    """RC3P prediction-set mask: score pass AND rank pass."""
    return (nc_scores <= thresholds) & (candidate_ranks <= k_hat)


class RC3P(SetPredictor):
    """RC3P: Rank Calibrated Class-conditional Conformal Prediction.

    Standard class-conditional conformal prediction (CCP, the
    per-class version of LABEL) includes a label whenever its
    nonconformity score passes its class threshold, scanning every
    label. RC3P adds a label-rank calibration step: a label is
    included only if it also ranks within the class-specific cutoff
    ``k_hat(y)`` calibrated from the top-k error of that class. The
    score threshold for class y is then computed at the adjusted
    miscoverage ``alpha_hat(y) = alpha - eps_y^{k_hat(y)}`` instead of
    at alpha, which is what targets per-class coverage while the rank
    filter is intended to shrink the sets in regimes where
    class-conditional sets are large. The paper's coverage theorem is stated
    for known population top-k errors; this implementation estimates
    those errors from a finite rank subset, so the procedure is an
    empirical calibration targeting class-wise coverage, not a proven
    finite-sample guarantee. Splitting rank and score calibration
    keeps the cutoff selection independent of the score thresholds,
    which is the part of the argument the split supports.

    Calibration uses two disjoint subsets, because the paper takes
    the top-k errors as inputs and estimating them from the same
    samples used for the score quantiles would couple the cutoff
    selection to the thresholds. The rank subset estimates the
    top-k errors and selects the cutoffs and adjusted rates; the
    score subset computes the per-class score quantiles at those
    adjusted rates. Splitting costs data on both sides: each subset
    is smaller than the full calibration set, so both the rank-error
    estimates and the score quantiles are noisier than a
    single-subset procedure would be. A class absent from either
    subset falls back to threshold +inf and cutoff K, so it is always
    included, which preserves coverage at the cost of larger sets.

    This is a prediction-set constructor for multi-class
    classification problems. It targets class-wise miscoverage at
    level alpha, but its finite-sample coverage depends on the
    accuracy of the estimated top-k errors and is not guaranteed by
    this implementation alone.

    Paper:

        Shi, Yuanjie, Subhankar Ghosh, Taha Belkhouja, Janardhan
        Rao Doppa, and Yan Yan. "Conformal Prediction for
        Class-wise Coverage via Augmented Label Rank Calibration."
        NeurIPS (2024). https://arxiv.org/abs/2406.06818

    :param model: A trained base model.
    :type model: BaseModel
    :param alpha: Target mis-coverage rate, a float in (0, 1).
    :type alpha: float
    :param score_type: Nonconformity score to use: "threshold" (default,
        the LAC score, NC score = 1 - p(true class)) or "aps"
        (Adaptive Prediction Sets, Romano, Sesia, and Candes 2020). See
        :mod:`pyhealth.calib.predictionset.scores` for the exact formulas.
    :type score_type: str
    :param random_state: Optional int seed for the RNG used for the
        internal calibration split and by score_type="aps".
    :type random_state: int | None
    :param rank_fraction: Fraction of each class's calibration samples
        used for rank calibration when no separate rank dataset is
        given. Each class keeps at least one sample in each subset
        when it has at least two samples.
    :type rank_fraction: float

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset, split_by_patient, get_dataloader
        >>> from pyhealth.models import SparcNet
        >>> from pyhealth.tasks import sleep_staging_isruc_fn
        >>> from pyhealth.calib.predictionset import RC3P
        >>> sleep_ds = ISRUCDataset("/srv/scratch1/data/ISRUC-I").set_task(sleep_staging_isruc_fn)
        >>> train_data, val_data, test_data = split_by_patient(sleep_ds, [0.6, 0.2, 0.2])
        >>> model = SparcNet(dataset=sleep_ds, feature_keys=["signal"],
        ...     label_key="label", mode="multiclass")
        >>> # ... Train the model here ...
        >>> cal_model = RC3P(model, 0.1)
        >>> cal_model.calibrate(cal_dataset=val_data)
        >>> # Evaluate
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>> test_dl = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> y_true_all, y_prob_all, _, extra_output = Trainer(model=cal_model).inference(test_dl, additional_outputs=['y_predset'])
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: float,
        score_type: str = "threshold",
        random_state: int | None = None,
        rank_fraction: float = 0.5,
        debug=False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        if model.mode != "multiclass":
            raise NotImplementedError()
        if score_type not in SUPPORTED_SCORE_TYPES:
            raise ValueError(
                f"Unknown score_type: {score_type!r}. Supported: "
                f"{SUPPORTED_SCORE_TYPES}."
            )
        if isinstance(alpha, bool) or not isinstance(alpha, (float, int)):
            raise TypeError("alpha must be a float in (0, 1).")
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be a float in (0, 1).")
        if not 0 < rank_fraction < 1:
            raise ValueError("rank_fraction must be in (0, 1).")
        self.mode = self.model.mode
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = model.device
        self.debug = debug
        self.score_type = score_type
        self.rng = np.random.default_rng(random_state)
        self.alpha = alpha
        self.rank_fraction = rank_fraction
        self.t = None
        self.k_hat = None
        self.alpha_hat = None

    def _split_calibration(self, y_true: np.ndarray):
        """Stratified disjoint split into rank and score index arrays."""
        rank_idx, score_idx = [], []
        for y in np.unique(y_true):
            class_idx = np.where(y_true == y)[0]
            perm = self.rng.permutation(class_idx)
            if len(perm) >= 2:
                n_rank = round(self.rank_fraction * len(perm))
                n_rank = min(max(n_rank, 1), len(perm) - 1)
            else:
                n_rank = len(perm)
            rank_idx.append(perm[:n_rank])
            score_idx.append(perm[n_rank:])
        rank_idx = (
            np.concatenate(rank_idx) if rank_idx else np.array([], dtype=int)
        )
        score_idx = (
            np.concatenate(score_idx) if score_idx else np.array([], dtype=int)
        )
        return rank_idx, score_idx

    def calibrate(
        self,
        cal_dataset: Subset,
        rank_cal_dataset: Subset | None = None,
    ):
        """Calibrate the RC3P thresholds and rank cutoffs.

        :param cal_dataset: Calibration set for the score quantiles.
        :type cal_dataset: Subset
        :param rank_cal_dataset: Optional separate calibration set for
            the rank calibration. When omitted, cal_dataset is split
            internally, stratified by class, into disjoint rank and
            score subsets.
        :type rank_cal_dataset: Subset | None
        """
        score_data = prepare_numpy_dataset(
            self.model, cal_dataset, ["y_prob", "y_true"], debug=self.debug
        )
        if len(score_data["y_true"]) == 0:
            raise ValueError("calibration dataset is empty.")
        score_y_prob = score_data["y_prob"]
        score_y_true = score_data["y_true"].astype(np.int64)
        num_classes = score_y_prob.shape[1]
        if rank_cal_dataset is not None:
            rank_data = prepare_numpy_dataset(
                self.model,
                rank_cal_dataset,
                ["y_prob", "y_true"],
                debug=self.debug,
            )
            if len(rank_data["y_true"]) == 0:
                raise ValueError("rank calibration dataset is empty.")
            rank_y_prob = rank_data["y_prob"]
            rank_y_true = rank_data["y_true"].astype(np.int64)
        else:
            rank_idx, score_idx = self._split_calibration(score_y_true)
            rank_y_prob = score_y_prob[rank_idx]
            rank_y_true = score_y_true[rank_idx]
            score_y_prob = score_y_prob[score_idx]
            score_y_true = score_y_true[score_idx]
        k_hat, alpha_hat = _rank_calibration(
            rank_y_prob, rank_y_true, self.alpha
        )
        score_nc = true_class_nc_scores(
            score_y_prob,
            score_y_true,
            score_type=self.score_type,
            rng=self.rng,
        )
        thresholds = np.full(num_classes, np.inf, dtype=np.float64)
        for y in range(num_classes):
            class_scores = score_nc[score_y_true == y]
            if len(class_scores) > 0:
                thresholds[y] = _query_quantile(class_scores, alpha_hat[y])
        # Always-include fallback: a class missing from either subset
        # gets both fallback values, not just the one its own subset
        # would give. A finite threshold with cutoff K still filters
        # nothing, but an infinite threshold with a cutoff below K
        # would wrongly exclude high-rank candidates, so both must
        # be forced together.
        rank_present = set(np.unique(rank_y_true).tolist())
        score_present = set(np.unique(score_y_true).tolist())
        for y in range(num_classes):
            if y not in rank_present or y not in score_present:
                thresholds[y] = np.inf
                k_hat[y] = num_classes
                # With cutoff K, the top-K error is 0, so the
                # adjusted rate for the forced cutoff is alpha
                # itself, not the rate computed for the original
                # cutoff. Keeping the old rate would leave the
                # stored calibration state internally inconsistent.
                alpha_hat[y] = self.alpha
        self.t = torch.tensor(thresholds, device=self.device)
        self.k_hat = k_hat
        self.alpha_hat = alpha_hat

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation (just like the original model).

        :return: A dictionary with all results from the base model, with the following updates:

                    y_predset: a bool tensor representing the prediction for each class.
        :rtype: dict[str, torch.Tensor]
        """
        if self.t is None or self.k_hat is None:
            raise RuntimeError("RC3P must be calibrated before forward.")
        pred = self.model(**kwargs)
        y_prob = pred["y_prob"].detach().cpu().numpy()
        nc_scores = all_class_nc_scores(
            y_prob, score_type=self.score_type, rng=self.rng
        )
        nc_scores = torch.as_tensor(
            nc_scores, device=pred["y_prob"].device, dtype=pred["y_prob"].dtype
        )
        ranks = _candidate_ranks(y_prob)
        thresholds = self.t.detach().cpu().numpy()
        mask = _predset_mask(
            nc_scores.detach().cpu().numpy(), thresholds, ranks, self.k_hat
        )
        pred["y_predset"] = torch.as_tensor(
            mask, device=pred["y_prob"].device, dtype=torch.bool
        )
        return pred
