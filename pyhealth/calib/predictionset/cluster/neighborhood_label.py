"""
Neighborhood Conformal Prediction (NCP).

"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import IterableDataset

from pyhealth.calib.base_classes import SetPredictor
from pyhealth.calib.predictionset.base_conformal import _query_weighted_quantile
from pyhealth.calib.utils import extract_embeddings, prepare_numpy_dataset
from pyhealth.models import BaseModel

__all__ = ["NeighborhoodLabel"]


class NeighborhoodLabel(SetPredictor):
    """Neighborhood Conformal Prediction (NCP) for multiclass classification.

    Reference:
        Ghosh, S., Belkhouja, T., Yan, Y., & Doppa, J. R. (2023).
        Improving Uncertainty Quantification of Deep Classifiers via
        Neighborhood Conformal Prediction.

    Args:
        model: A trained base model that supports embedding extraction
            (must support `embed=True` in forward pass).
        alpha: Target miscoverage rate; marginal coverage P(Y not in C(X)) <= alpha.
        k_neighbors: Number of nearest calibration neighbors. Default 50.
        lambda_L: Temperature for exponential weights; smaller => more localization.
            Default 100.0.
        debug: If True, process fewer samples for faster iteration.

    Examples:
        >>> from pyhealth.datasets import TUEVDataset, split_by_sample_conformal
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.models import ContraWR
        >>> from pyhealth.tasks import EEGEventsTUEV
        >>> from pyhealth.calib.predictionset.cluster import NeighborhoodLabel
        >>> from pyhealth.calib.utils import extract_embeddings
        >>> from pyhealth.trainer import Trainer, get_metrics_fn
        >>>
        >>> dataset = TUEVDataset(root="path/to/tuev")
        >>> sample_dataset = dataset.set_task(EEGEventsTUEV())
        >>> train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal(
        ...     sample_dataset, ratios=[0.6, 0.1, 0.15, 0.15], seed=42
        ... )
        >>> model = ContraWR(dataset=sample_dataset)
        >>> cal_embeddings = extract_embeddings(model, cal_ds, batch_size=32)
        >>> ncp = NeighborhoodLabel(model=model, alpha=0.1, k_neighbors=50)
        >>> ncp.calibrate(cal_dataset=cal_ds, cal_embeddings=cal_embeddings)
        >>> test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
        >>> y_true, y_prob, _, extra = Trainer(model=ncp).inference(
        ...     test_loader, additional_outputs=["y_predset"]
        ... )
        >>> metrics = get_metrics_fn(ncp.mode)(
        ...     y_true, y_prob, metrics=["accuracy", "miscoverage_ps"],
        ...     y_predset=extra["y_predset"]
        ... )
    """

    def __init__(
        self,
        model: BaseModel,
        alpha: float,
        k_neighbors: int = 50,
        lambda_L: float = 100.0,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)

        if model.mode != "multiclass":
            raise NotImplementedError(
                "NeighborhoodLabel only supports multiclass classification"
            )

        self.mode = self.model.mode

        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.device = model.device
        self.debug = debug

        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.alpha = float(alpha)

        if not isinstance(k_neighbors, int) or k_neighbors <= 0:
            raise ValueError(
                f"k_neighbors must be a positive integer, got {k_neighbors!r}"
            )
        self.k_neighbors = k_neighbors
        self.lambda_L = float(lambda_L)

        self.cal_embeddings_ = None
        self.cal_conformity_scores_ = None
        self.alpha_tilde_ = None
        self._nn = None

    def calibrate(
        self,
        cal_dataset: IterableDataset,
        cal_embeddings: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> None:
        """Calibrate NCP steps:

        Step 1: For each calibration point i, compute Q̃^NCP (weighted quantile)
        over its k-NN in calibration using weights.
        Step 2: Find ã^NCP(α) = largest ã such that empirical coverage on the
        calibration set is >= 1-α; store as alpha_tilde_ for use at test time.

        Args:
            cal_dataset: Calibration dataset (for labels and predictions if
                cal_embeddings not provided).
            cal_embeddings: Optional precomputed calibration embeddings
                (n_cal, embedding_dim). If None, extracted from cal_dataset.
            batch_size: Batch size for embedding extraction when cal_embeddings
                is not provided.
        """
        cal_dict = prepare_numpy_dataset(
            self.model,
            cal_dataset,
            ["y_prob", "y_true"],
            debug=self.debug,
        )
        y_prob = cal_dict["y_prob"]
        y_true = cal_dict["y_true"]
        N = y_prob.shape[0]

        if cal_embeddings is None:
            cal_embeddings = extract_embeddings(
                self.model, cal_dataset, batch_size=batch_size, device=self.device
            )
        else:
            cal_embeddings = np.asarray(cal_embeddings)

        if cal_embeddings.shape[0] != N:
            raise ValueError(
                f"cal_embeddings length {cal_embeddings.shape[0]} must match "
                f"cal_dataset size {N}"
            )

        conformity_scores = y_prob[np.arange(N), y_true]

        k = min(self.k_neighbors, N)
        self._nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(
            np.atleast_2d(cal_embeddings)
        )
        self.cal_embeddings_ = np.atleast_2d(cal_embeddings)
        self.cal_conformity_scores_ = np.asarray(conformity_scores, dtype=np.float64)

       # this is the ncp calibration step
        distances_cal, indices_cal = self._nn.kneighbors(
            self.cal_embeddings_, n_neighbors=k
        )
        cal_weights = np.exp(-distances_cal / self.lambda_L)
        cal_weights = cal_weights / cal_weights.sum(axis=1, keepdims=True)

        def _empirical_coverage(alpha_tilde_cand: float) -> float:
            t_all = np.zeros(N, dtype=np.float64)
            for i in range(N):
                t_all[i] = _query_weighted_quantile(
                    self.cal_conformity_scores_[indices_cal[i]],
                    1.0 - alpha_tilde_cand,
                    cal_weights[i],
                )
            return float(np.mean(self.cal_conformity_scores_ <= t_all))

        low, high = 0.0, 1.0
        for _ in range(50):
            mid = (low + high) / 2
            if _empirical_coverage(mid) >= 1.0 - self.alpha:
                low = mid
            else:
                high = mid
        self.alpha_tilde_ = float(low)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward with NCP: per-sample weighted quantile threshold (Eq 2, Q̃^NCP)."""
        if (
            self.cal_embeddings_ is None
            or self.cal_conformity_scores_ is None
            or self.alpha_tilde_ is None
        ):
            raise RuntimeError(
                "NeighborhoodLabel must be calibrated before inference. "
                "Call calibrate() first."
            )

        pred = self.model(**{**kwargs, "embed": True})
        if "embed" not in pred:
            raise ValueError(
                f"Model {type(self.model).__name__} does not return "
                "embeddings. Ensure it supports embed=True in forward()."
            )

        test_emb = pred["embed"].detach().cpu().numpy()
        test_emb = np.atleast_2d(test_emb)
        batch_size = test_emb.shape[0]
        n_cal = self.cal_conformity_scores_.shape[0]
        k = min(self.k_neighbors, n_cal)

        distances, indices = self._nn.kneighbors(test_emb, n_neighbors=k)
        thresholds = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            w = np.exp(-distances[i] / self.lambda_L)
            w = w / np.sum(w)
            scores_i = self.cal_conformity_scores_[indices[i]]
            thresholds[i] = _query_weighted_quantile(
                scores_i, 1.0 - self.alpha_tilde_, w
            )

        th = torch.as_tensor(
            thresholds, device=self.device, dtype=pred["y_prob"].dtype
        )
        if pred["y_prob"].ndim > 1:
            th = th.view(-1, *([1] * (pred["y_prob"].ndim - 1)))
        pred["y_predset"] = pred["y_prob"] >= th
        pred.pop("embed", None)
        return pred
