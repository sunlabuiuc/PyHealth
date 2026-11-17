"""Group-Conditional (Fair) Conformal Prediction.

Contributor: Ibrahim Rabiu (GitHub: @alhibb, alhibb01@gmail.com)
Paper: Lu, Charles, Andreanne Lemay, Ken Chang, Katharina Hobel, and
    Jayashree Kalpathy-Cramer. "Fair Conformal Predictors for Applications
    in Medical Imaging." AAAI Conference on Artificial Intelligence (2022).
Paper link: https://arxiv.org/abs/2109.02058

Constructs prediction sets with guaranteed equalized coverage across
demographic subgroups (e.g., race, sex, age, skin tone) by computing
group-specific nonconformity thresholds.
"""

import warnings
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.predictionset.scores import (
    SUPPORTED_SCORE_TYPES,
    all_class_nc_scores,
    true_class_nc_scores,
)
from pyhealth.calib.utils import prepare_numpy_dataset

__all__ = ["GroupConditionalConformal"]


def _query_quantile(nc_scores: np.ndarray, alpha: float) -> float:
    """Compute the conformal quantile threshold on nonconformity scores.

    Implements the standard split conformal quantile:
        q = ceil((1 - alpha) * (N + 1)) - 1 (0-indexed).

    Returns +inf when the calibration set is too small (N < 1/alpha - 1),
    meaning all classes are included to preserve coverage.
    """
    nc_scores = np.sort(nc_scores)
    n = len(nc_scores)
    loc = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    if loc >= n:
        return float(np.inf)
    return float(nc_scores[loc])


class GroupConditionalConformal(SetPredictor):
    """Group-Conditional Conformal Prediction for Multiclass Classification.

    Constructs prediction sets with guaranteed equalized coverage across
    demographic subgroups (e.g., race, sex, age, skin tone). Standard marginal
    conformal prediction guarantees marginal coverage :math:`P(Y \\in C(X)) \\ge 1 - \\alpha`,
    but can exhibit severe under-coverage for protected or minority subgroups.
    Group-conditional conformal prediction computes group-specific nonconformity
    thresholds :math:`\\hat{q}_g` such that:

    .. math::
        P(Y \\in C(X) \\mid G = g) \\ge 1 - \\alpha_g, \\quad \\forall g \\in \\mathcal{G}

    Papers:
        Lu, Charles, Andreanne Lemay, Ken Chang, Katharina Hobel, and
        Jayashree Kalpathy-Cramer. "Fair Conformal Predictors for Applications
        in Medical Imaging." AAAI 2022.

    Args:
        model: A trained classification model (e.g., :class:`BaseModel`) that
            outputs predicted probabilities or logits.
        alpha: Target miscoverage rate. Can be a single float :math:`\\alpha \\in (0, 1)`
            applied to all groups, or a dictionary mapping each group identifier
            to its specific target miscoverage rate. Default is 0.1 (90% coverage).
        score_type: Nonconformity score function to use:
            - ``"threshold"`` (default): :math:`1 - \\hat{p}(x, y)`, the standard
              score from Sadinle et al. (2019).
            - ``"aps"``: Adaptive Prediction Sets from Romano et al. (2020),
              which adapts prediction set sizes to the model confidence profile.
        group_key: Key in the sample/batch dictionary identifying the demographic
            or sensitive subgroup. Default is ``"subgroup"``.
        random_state: Optional seed for the random number generator used by the
            ``"aps"`` score. Ignored for ``"threshold"``.
        debug: Whether to run in debug mode (evaluating fewer batches during
            calibration). Default is False.

    Examples:
        >>> import torch
        >>> from pyhealth.calib.predictionset import GroupConditionalConformal
        >>> class ToyModel(torch.nn.Module):
        ...     def forward(self, **kwargs):
        ...         probs = torch.tensor([[0.8, 0.15, 0.05], [0.2, 0.7, 0.1]])
        ...         return {"y_prob": probs, "logit": probs}
        >>> model = ToyModel()
        >>> conf = GroupConditionalConformal(model, alpha=0.1, group_key="subgroup")
        >>> y_prob = np.array([
        ...     [0.8, 0.15, 0.05],
        ...     [0.1, 0.8, 0.1],
        ...     [0.2, 0.2, 0.6],
        ...     [0.7, 0.2, 0.1],
        ... ])
        >>> y_true = np.array([0, 1, 2, 0])
        >>> groups = np.array(["group_A", "group_A", "group_B", "group_B"])
        >>> conf.calibrate(scores=y_prob, labels=y_true, groups=groups)
        >>> out = conf(y_prob=torch.tensor([[0.75, 0.2, 0.05]]), subgroup=["group_A"])
        >>> out["y_predset"].shape
        torch.Size([1, 3])
    """

    def __init__(
        self,
        model: Any,
        alpha: float | dict[Any, float] = 0.1,
        score_type: str = "threshold",
        group_key: str = "subgroup",
        random_state: int | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(model)
        if hasattr(self.model, "mode") and self.model.mode != "multiclass":
            raise NotImplementedError(
                f"GroupConditionalConformal only supports multiclass classification, "
                f"got {self.model.mode}"
            )
        # Freeze model parameters and set eval mode
        if hasattr(self.model, "parameters"):
            for param in self.model.parameters():
                param.requires_grad = False
        if hasattr(self.model, "eval"):
            self.model.eval()

        if isinstance(alpha, (int, float)):
            if not 0.0 < float(alpha) < 1.0:
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        elif isinstance(alpha, dict):
            if not alpha:
                raise ValueError("alpha dict must not be empty.")
            for g, a in alpha.items():
                if not 0.0 < float(a) < 1.0:
                    raise ValueError(f"alpha for group {g!r} must be in (0, 1), got {a}")
        else:
            raise TypeError(f"alpha must be float or dict, got {type(alpha).__name__}")

        if score_type not in SUPPORTED_SCORE_TYPES:
            raise ValueError(
                f"Unknown score_type: {score_type!r}. Supported: {SUPPORTED_SCORE_TYPES}."
            )

        self.alpha = alpha
        self.score_type = score_type
        self.group_key = group_key
        self.random_state = random_state
        self.debug = debug

        self.q_hat: dict[Any, float] = {}
        self.q_hat_marginal: float = float("inf")
        self.calibrated: bool = False
        self._rng = np.random.default_rng(random_state)

    @property
    def mode(self) -> str | None:
        """Returns the base model's task mode if defined."""
        return getattr(self.model, "mode", None)

    def _get_alpha_for_group(self, group: Any) -> float:
        if isinstance(self.alpha, dict):
            if group in self.alpha:
                return float(self.alpha[group])
            raise KeyError(
                f"Group {group!r} not found in alpha mapping: {list(self.alpha.keys())}"
            )
        return float(self.alpha)

    def calibrate(
        self,
        cal_dataset: Any = None,
        scores: np.ndarray | Tensor | None = None,
        labels: np.ndarray | Tensor | None = None,
        groups: np.ndarray | Tensor | list | None = None,
    ) -> None:
        """Calibrates group-specific nonconformity thresholds.

        Accepts either a PyHealth dataset or precomputed scores, labels, and groups.

        Args:
            cal_dataset: Calibration dataset compatible with :func:`prepare_numpy_dataset`.
            scores: Array of predicted probabilities of shape :math:`(N, K)`.
            labels: Ground truth class indices of shape :math:`(N,)`.
            groups: Demographic subgroup indicators of shape :math:`(N,)`.

        Raises:
            ValueError: If neither ``cal_dataset`` nor array inputs are provided,
                or if shapes do not match, or if task is not multiclass.
        """
        if hasattr(self.model, "eval"):
            self.model.eval()

        if cal_dataset is not None:
            data = prepare_numpy_dataset(
                self.model,
                cal_dataset,
                keys=["y_prob", "y_true"],
                incl_data_keys=[self.group_key],
                debug=self.debug,
            )
            y_prob = np.asarray(data["y_prob"])
            y_true = np.asarray(data["y_true"])
            subgroups = np.asarray(data[self.group_key])
        elif scores is not None and labels is not None and groups is not None:
            y_prob = (
                scores.detach().cpu().numpy()
                if isinstance(scores, Tensor)
                else np.asarray(scores)
            )
            y_true = (
                labels.detach().cpu().numpy()
                if isinstance(labels, Tensor)
                else np.asarray(labels)
            )
            subgroups = (
                groups.detach().cpu().numpy()
                if isinstance(groups, Tensor)
                else np.asarray(groups)
            )
        else:
            raise ValueError(
                "Either cal_dataset or all of (scores, labels, groups) must be provided."
            )

        if y_prob.ndim != 2 or y_prob.shape[1] < 2:
            raise ValueError(
                f"GroupConditionalConformal requires multiclass probabilities of "
                f"shape (N, K) with K >= 2, got shape {y_prob.shape}"
            )

        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)
        y_true = y_true.astype(np.int64)

        if len(y_prob) != len(y_true) or len(y_prob) != len(subgroups):
            raise ValueError(
                f"Shape mismatch: scores ({len(y_prob)}), labels ({len(y_true)}), "
                f"and groups ({len(subgroups)}) must have identical lengths."
            )

        # Compute true class nonconformity scores for all calibration samples
        cal_nc_scores = true_class_nc_scores(
            y_prob, y_true, score_type=self.score_type, rng=self._rng
        )

        # Fallback marginal threshold across all groups
        marginal_alpha = (
            np.mean(list(self.alpha.values()))
            if isinstance(self.alpha, dict)
            else float(self.alpha)
        )
        self.q_hat_marginal = _query_quantile(cal_nc_scores, marginal_alpha)

        # Compute group-specific thresholds
        self.q_hat = {}
        unique_groups = np.unique(subgroups)
        for g in unique_groups:
            idx = np.where(subgroups == g)[0]
            group_alpha = self._get_alpha_for_group(g)
            self.q_hat[g] = _query_quantile(cal_nc_scores[idx], group_alpha)

        self.calibrated = True

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        """Runs inference and constructs prediction sets using group thresholds.

        Args:
            **kwargs: Input features passed to the base model, optionally including
                the group attribute specified by ``self.group_key``, predicted probabilities
                ``"y_prob"``, and model outputs (e.g. ``"logit"``, ``"loss"``, ``"y_true"``).

        Returns:
            Dictionary containing:
                - ``"y_predset"``: Boolean tensor of shape :math:`(N, K)` indicating
                  whether class :math:`k` is included in the prediction set.
                - ``"y_prob"``: Predicted class probabilities tensor of shape :math:`(N, K)`.
                - Model outputs (e.g. ``"logit"``, ``"loss"``, ``"y_true"``) when
                  forwarding through the base model, or passed through if provided.
        """
        if not self.calibrated:
            raise RuntimeError(
                "GroupConditionalConformal must be calibrated before forward()."
            )

        # If y_prob is already provided, skip base model forward
        if "y_prob" in kwargs and isinstance(kwargs["y_prob"], (Tensor, np.ndarray)):
            y_prob_raw = kwargs["y_prob"]
            y_prob = (
                y_prob_raw
                if isinstance(y_prob_raw, Tensor)
                else torch.from_numpy(y_prob_raw)
            )
            res: dict[str, Any] = {"y_prob": y_prob}
            for passthrough_k in ["logit", "loss", "y_true"]:
                if passthrough_k in kwargs:
                    res[passthrough_k] = kwargs[passthrough_k]
        else:
            res = self.model(**kwargs)
            y_prob = res["y_prob"]

        y_prob_np = y_prob.detach().cpu().numpy()
        n_samples, _n_classes = y_prob_np.shape

        # Retrieve subgroup values from kwargs
        subgroups = kwargs.get(self.group_key)
        if subgroups is None:
            warnings.warn(
                f"Subgroup key '{self.group_key}' not found in forward kwargs; "
                "falling back to marginal threshold for all samples.",
                UserWarning,
                stacklevel=2,
            )
            q_values = np.full(n_samples, self.q_hat_marginal)
        else:
            if isinstance(subgroups, Tensor):
                subgroups = subgroups.detach().cpu().numpy()
            elif isinstance(subgroups, (list, tuple)):
                subgroups = np.asarray(subgroups)

            q_values = np.empty(n_samples, dtype=np.float64)
            for i in range(n_samples):
                g = subgroups[i]
                if isinstance(g, np.ndarray) and g.ndim == 0:
                    g = g.item()
                if g not in self.q_hat:
                    warnings.warn(
                        f"Unseen subgroup '{g}' encountered; falling back to "
                        "marginal conformal threshold.",
                        UserWarning,
                        stacklevel=2,
                    )
                q_values[i] = self.q_hat.get(g, self.q_hat_marginal)

        # Calculate all-class nonconformity scores
        nc_scores = all_class_nc_scores(
            y_prob_np, score_type=self.score_type, rng=self._rng
        )

        # Include class k if nc_score <= q_g
        y_predset_np = nc_scores <= q_values[:, None]

        device = y_prob.device if isinstance(y_prob, Tensor) else torch.device("cpu")
        res["y_predset"] = torch.from_numpy(y_predset_np).to(device)

        return res
