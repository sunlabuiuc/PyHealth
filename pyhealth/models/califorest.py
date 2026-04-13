# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Calibrated random forest using OOB variance-weighted calibration
"""CaliForest: Calibrated Random Forest for Health Data.

This module implements CaliForest (Park and Ho, 2020), a calibrated
random forest that uses out-of-bag (OOB) prediction variance to learn
a robust calibration function without requiring a held-out calibration
set.

Paper: Y. Park and J. C. Ho. "CaliForest: Calibrated Random Forest
for Health Data." ACM Conference on Health, Inference, and Learning,
2020. https://doi.org/10.1145/3368555.3384461

Note:
    CaliForest wraps scikit-learn's ``DecisionTreeClassifier`` and
    calibration estimators internally. The ``forward`` method accepts
    and returns ``torch.Tensor`` objects to conform to PyHealth's
    ``BaseModel`` interface.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


class CaliForest(BaseModel):
    """Calibrated Random Forest for health data.

    CaliForest retains per-tree OOB predictions for each training
    sample and estimates prediction uncertainty via an Inverse-Gamma
    conjugate prior on the noise variance. Samples with low variance
    (high certainty) receive larger weights when fitting the
    calibration function, yielding better-calibrated probabilities
    without sacrificing discrimination.

    Two calibration modes are supported:

    * ``"isotonic"`` (CF-Iso): non-parametric isotonic regression.
    * ``"logistic"`` (CF-Logit): Platt scaling via logistic
      regression.
    * ``"beta"`` (CF-Beta): Beta calibration via logistic
      regression on log-transformed predictions.

    Args:
        dataset: A ``SampleDataset`` used to query label information.
        feature_keys: List of feature keys in the input batch.
        label_key: Key for the label in the input batch.
        n_estimators: Number of trees in the forest. Default ``300``.
        max_depth: Maximum depth of each tree. Default ``10``.
        min_samples_split: Minimum samples to split a node.
            Default ``3``.
        min_samples_leaf: Minimum samples in a leaf. Default ``1``.
        ctype: Calibration type, ``"isotonic"``, ``"logistic"``,
            or ``"beta"``. Default ``"isotonic"``.
        alpha0: Prior shape parameter for the Inverse-Gamma
            distribution. Default ``100``.
        beta0: Prior scale parameter for the Inverse-Gamma
            distribution. Default ``25`` (prior variance =
            beta0 / alpha0 = 0.25).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "features": [0.1] * 10,
        ...         "label": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "features": [0.9] * 10,
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="demo",
        ... )
        >>> model = CaliForest(
        ...     dataset=dataset,
        ...     feature_keys=["features"],
        ...     label_key="label",
        ...     n_estimators=50,
        ...     max_depth=5,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        n_estimators: int = 300,
        max_depth: int = 10,
        min_samples_split: int = 3,
        min_samples_leaf: int = 1,
        ctype: str = "isotonic",
        alpha0: float = 100.0,
        beta0: float = 25.0,
    ) -> None:
        super().__init__(dataset)
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.estimators_: List[DecisionTreeClassifier] = []
        self.calibrator_: Optional[Any] = None
        self.is_fitted_ = False

        # Dummy parameter so BaseModel.device works
        self._dummy_param = nn.Parameter(torch.empty(0))

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def fit(
        self,
        train_dataloader: Any,
    ) -> "CaliForest":
        """Fits the CaliForest model on training data.

        Collects all batches from the dataloader, trains individual
        decision trees with bootstrap sampling, computes OOB
        predictions and inverse-variance sample weights, then fits
        the calibration function.

        Args:
            train_dataloader: A PyTorch ``DataLoader`` yielding
                batches with keys matching ``feature_keys`` and
                ``label_key``.

        Returns:
            The fitted ``CaliForest`` instance.
        """
        X_parts, y_parts = [], []
        for batch in train_dataloader:
            feats = [batch[k].numpy() for k in self.feature_keys]
            X_parts.append(np.concatenate(feats, axis=1))
            y_parts.append(batch[self.label_key].numpy().ravel())
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        self._fit_numpy(X, y)
        return self

    def _fit_numpy(self, X: np.ndarray, y: np.ndarray) -> None:
        """Core fitting logic on numpy arrays.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary label array of shape ``(n_samples,)``.
        """
        n = X.shape[0]
        self.estimators_ = []

        # Build calibrator
        if self.ctype == "logistic":
            self.calibrator_ = LogisticRegression(
                penalty=None, solver="saga", max_iter=5000
            )
        elif self.ctype == "beta":
            self.calibrator_ = LogisticRegression(
                penalty=None, solver="saga", max_iter=5000
            )
        else:
            self.calibrator_ = IsotonicRegression(
                y_min=0, y_max=1, out_of_bounds="clip"
            )

        # OOB prediction matrix: (n_samples, n_estimators), NaN if
        # sample was in-bag for that tree
        Y_oob = np.full((n, self.n_estimators), np.nan)
        n_oob = np.zeros(n)
        IB = np.zeros((n, self.n_estimators), dtype=int)
        OOB = np.full((n, self.n_estimators), True)

        for eid in range(self.n_estimators):
            IB[:, eid] = np.random.choice(n, n)
            OOB[IB[:, eid], eid] = False

        for eid in range(self.n_estimators):
            est = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            ib_idx = IB[:, eid]
            oob_mask = OOB[:, eid]
            est.fit(X[ib_idx], y[ib_idx])
            proba = est.predict_proba(X[oob_mask])
            # Handle case where tree only sees one class
            if proba.shape[1] == 2:
                Y_oob[oob_mask, eid] = proba[:, 1]
            else:
                Y_oob[oob_mask, eid] = proba[:, 0]
            n_oob[oob_mask] += 1
            self.estimators_.append(est)

        # Compute inverse-variance weights via Inverse-Gamma update
        valid = n_oob > 1
        Y_oob_valid = Y_oob[valid]
        n_oob_valid = n_oob[valid]
        z_hat = np.nanmean(Y_oob_valid, axis=1)
        z_true = y[valid]
        beta = (
            self.beta0
            + np.nanvar(Y_oob_valid, axis=1) * n_oob_valid / 2
        )
        alpha = self.alpha0 + n_oob_valid / 2
        z_weight = alpha / beta

        # Fit calibrator
        if self.ctype == "logistic":
            self.calibrator_.fit(
                z_hat[:, np.newaxis], z_true, z_weight
            )
        elif self.ctype == "beta":
            self.calibrator_.fit(
                self._beta_features(z_hat), z_true, z_weight
            )
        else:
            self.calibrator_.fit(z_hat, z_true, z_weight)

        self.is_fitted_ = True

    # ----------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------

    def _beta_features(self, p: np.ndarray) -> np.ndarray:
        """Compute Beta calibration features from probabilities.

        Beta calibration (Kull et al., 2017) fits a logistic
        regression on three log-transformed features of the
        predicted probability.

        Args:
            p: Predicted probabilities of shape ``(n,)``.

        Returns:
            Feature matrix of shape ``(n, 3)``.
        """
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        return np.column_stack([
            np.log(p / (1 - p)),
            np.log(p),
            np.log(1 - p),
        ])

    def _predict_proba_numpy(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated probabilities on numpy input.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples,)`` with calibrated
            probabilities for the positive class.
        """
        z = np.zeros(X.shape[0])
        for est in self.estimators_:
            proba = est.predict_proba(X)
            if proba.shape[1] == 2:
                z += proba[:, 1]
            else:
                z += proba[:, 0]
        z /= len(self.estimators_)

        if self.ctype == "logistic":
            return self.calibrator_.predict_proba(
                z[:, np.newaxis]
            )[:, 1]
        if self.ctype == "beta":
            return self.calibrator_.predict_proba(
                self._beta_features(z)
            )[:, 1]
        return self.calibrator_.predict(z)

    # ----------------------------------------------------------
    # PyHealth interface
    # ----------------------------------------------------------

    def forward(
        self, **kwargs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass conforming to PyHealth's ``BaseModel``.

        During training (``self.training is True`` and model is not
        yet fitted), this method stores data and returns a dummy loss.
        After ``fit()`` has been called, it returns calibrated
        predictions.

        Args:
            **kwargs: Keyword arguments containing tensors for each
                feature key and the label key.

        Returns:
            A dictionary with keys ``loss``, ``y_prob``, ``y_true``,
            and ``logit``.
        """
        feats = [kwargs[k] for k in self.feature_keys]
        X_tensor = torch.cat(feats, dim=-1).float()
        X_np = X_tensor.detach().cpu().numpy()

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].float()
        else:
            y_true = torch.zeros(X_np.shape[0])

        if not self.is_fitted_:
            # Before fit: return dummy outputs
            y_prob = torch.full_like(y_true, 0.5)
            logit = torch.zeros_like(y_true)
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            proba = self._predict_proba_numpy(X_np)
            y_prob = torch.tensor(
                proba, dtype=torch.float32
            ).unsqueeze(-1)
            logit = torch.log(
                y_prob.clamp(1e-7, 1 - 1e-7)
                / (1 - y_prob.clamp(1e-7, 1 - 1e-7))
            )
            loss = nn.functional.binary_cross_entropy(
                y_prob.squeeze(-1),
                y_true.squeeze(-1).float(),
            )

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }
