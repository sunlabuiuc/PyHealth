"""
Author: Kobe Guo
NetID: kobeg2

Paper: CaliForest: Calibrated Random Forests for Healthcare Prediction
Link: https://joyceho.github.io/assets/pdf/paper/park-chil20.pdf

Description:
Implementation of CaliForest, a calibrated random forest model that applies
post-hoc calibration (isotonic or logistic) to improve probability estimates
for healthcare prediction tasks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class CaliForest(BaseModel):
    """CaliForest model for calibrated probability prediction.

    This model wraps a RandomForestClassifier and applies a post-hoc
    calibration step using out-of-bag (OOB) predictions and prediction
    variance to improve probability estimates.

    Important:
        CaliForest is fit once on the full training set using fit(train_loader).
        After fitting, forward() should be used only for inference/evaluation.
        This implementation currently supports binary classification only.

    The overall procedure is:
        1. train a random forest classifier,
        2. compute OOB probabilities for each training sample,
        3. estimate prediction uncertainty using variance across tree outputs,
        4. fit a calibration model using uncertainty-weighted samples.

    Args:
        dataset: the dataset used to initialize feature and label schemas.
        n_estimators: number of trees in the random forest. Default is 100.
        max_depth: maximum depth of each tree. Default is None.
        calibration: calibration method. Supported values are ``"isotonic"``
            and ``"logistic"``. Default is ``"isotonic"``.
        random_state: random seed for reproducibility. Default is 42.
        **kwargs: additional compatibility arguments.

    Example:
        model = CaliForest(dataset=dataset, n_estimators=10)
        model.fit(train_loader)
        ret = model(**batch)
        print(ret["y_prob"].shape)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        calibration: str = "isotonic",
        random_state: int = 42,
        **kwargs,
    ):
        super(CaliForest, self).__init__(dataset)

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.calibration = calibration
        self.random_state = random_state

        if self.calibration not in {"isotonic", "logistic"}:
            raise ValueError(f"Unsupported calibration: {self.calibration}")

        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
        )

        self.calibrator = None
        self.is_fitted = False

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _build_feature_matrix(self, **kwargs) -> np.ndarray:
        """Convert PyHealth batch into NumPy feature matrix."""
        features: List[np.ndarray] = []

        for key in self.feature_keys:
            x = kwargs[key]

            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)

            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)

            features.append(arr.astype(np.float32))

        return np.concatenate(features, axis=1)

    def _build_labels(self, **kwargs) -> np.ndarray:
        y = kwargs[self.label_key]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        else:
            y = np.asarray(y)
        return y.reshape(-1)
    
    def fit(self, train_loader):
        """Fit CaliForest on the full training dataloader"""
        X_list = []
        y_list = []

        for batch in train_loader:
            X_list.append(self._build_feature_matrix(**batch))
            y_list.append(self._build_labels(**batch))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        self.fit_model(features=X, labels=y)
        return self
    
    def fit_model(self, **kwargs) -> None:
        """Fit RF + calibration model."""
        if "features" in kwargs and "labels" in kwargs:
            X = kwargs["features"]
            y = kwargs["labels"]
        else:
            X = self._build_feature_matrix(**kwargs)
            y = self._build_labels(**kwargs)

        unique_labels = np.unique(y)
        if set(unique_labels.tolist()) != {0, 1}:
            raise ValueError(
                "CaliForest currently supports binary classification only. "
                f"Got labels: {unique_labels.tolist()}"
            )
        self.rf.fit(X, y)

        if not hasattr(self.rf, "oob_decision_function_"):
            raise RuntimeError("OOB predictions not available.")

        oob_probs = self.rf.oob_decision_function_[:, 1]

        tree_probs = np.stack(
            [t.predict_proba(X)[:, 1] for t in self.rf.estimators_],
            axis=0,
        )
        variances = np.var(tree_probs, axis=0)

        # CaliForest uses inverse tree-level variance so more stable
        # predictions have greater influence during calibrator fitting.
        weights = 1.0 / (variances + 1e-6)

        if self.calibration == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(oob_probs, y, sample_weight=weights)
            self.calibrator = calibrator
        else:
            calibrator = LogisticRegression()
            calibrator.fit(
                oob_probs.reshape(-1, 1),
                y,
                sample_weight=weights,
            )
            self.calibrator = calibrator

        self.is_fitted = True

    def predict_proba_numpy(self, **kwargs) -> np.ndarray:
        """Predict calibrated probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first.")

        X = self._build_feature_matrix(**kwargs)
        rf_probs = self.rf.predict_proba(X)[:, 1]

        if self.calibration == "isotonic":
            calibrated = self.calibrator.predict(rf_probs)
        else:
            calibrated = self.calibrator.predict_proba(
                rf_probs.reshape(-1, 1)
            )[:, 1]

        return calibrated.reshape(-1, 1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """PyHealth forward pass."""
        if not self.is_fitted:
            raise RuntimeError(
                "CaliForest must be fitted before inference. "
                "Call model.fit(train_loader) first."
            )

        y_prob_np = self.predict_proba_numpy(**kwargs)

        y_prob = torch.tensor(
            y_prob_np, dtype=torch.float32, device=self.device
        )

        eps = 1e-6
        logits = torch.log(
            torch.clamp(y_prob, eps, 1 - eps) /
            torch.clamp(1 - y_prob, eps, 1 - eps)
        )

        logits = logits * self.logit_scale

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }