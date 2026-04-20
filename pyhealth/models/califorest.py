"""CaliForest: Calibrated Random Forest for Clinical Prediction.

This module implements CaliForest, which uses Out-of-Bag (OOB) predictions
for probability calibration without sacrificing training data.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pyhealth.models import BaseModel


class CaliForest(BaseModel):
    """Calibrated Random Forest using OOB-based calibration.

    CaliForest trains a Random Forest with OOB scoring enabled, then uses
    the OOB predictions to train a calibration model (Isotonic Regression
    or Platt Scaling) without requiring a separate data split.

    Args:
        dataset: PyHealth SampleDataset object.
        feature_keys: List of keys to extract features from batches.
        label_key: Key for the target label in batches.
        mode: Task mode. Must be "binary" for CaliForest.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        calibration_method: "isotonic" or "platt".
        min_oob_trees: Minimum OOB trees required for reliable samples.
        random_state: Control reproducibility.
    """

    def __init__(
        self,
        dataset: Any,
        feature_keys: List[str],
        label_key: str,
        mode: str = "binary",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        calibration_method: str = "isotonic",
        min_oob_trees: int = 1,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        if mode != "binary":
            raise ValueError(f"CaliForest strictly requires mode='binary', got '{mode}'")

        # Handle PyHealth's varying BaseModel __init__ signatures safely
        try:
            super().__init__(
                dataset=dataset,
                feature_keys=feature_keys,
                label_key=label_key,
                mode=mode,
                **kwargs
            )
        except TypeError:
            super().__init__(dataset=dataset, **kwargs)
            self.feature_keys = feature_keys
            self.label_key = label_key
            self.mode = mode

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.calibration_method = calibration_method
        self.min_oob_trees = min_oob_trees
        self.random_state = random_state

        self.rf_: Optional[RandomForestClassifier] = None
        self.calibrator_: Optional[Any] = None
        self.is_fitted_: bool = False
        self.oob_tree_counts_: Optional[np.ndarray] = None

        # Required by PyHealth Trainer (Optimizer expects >0 parameters)
        self._dummy_param = nn.Parameter(torch.empty(0))

    def _extract_x_y(self, kwargs: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts and formats X and y from the PyTorch batch dict."""
        X_list = []
        for key in self.feature_keys:
            val = kwargs[key].cpu().numpy()
            if val.ndim == 1:
                val = val.reshape(-1, 1)
            X_list.append(val)

        X = np.concatenate(X_list, axis=1)
        y = kwargs[self.label_key].cpu().numpy()
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CaliForest":
        """Fits the underlying random forest and the OOB calibration model."""
        if len(np.unique(y)) != 2:
            raise ValueError("CaliForest fit requires exactly binary targets.")

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            oob_score=True,
            random_state=self.random_state,
        )
        self.rf_.fit(X, y)

        n_samples = X.shape[0]
        oob_sum = np.zeros(n_samples)
        oob_count = np.zeros(n_samples)

        for i, (tree, samples_mask) in enumerate(
            zip(self.rf_.estimators_, self.rf_.estimators_samples_)
        ):
            oob_mask = ~samples_mask
            if not np.any(oob_mask):
                continue

            try:
                probs = tree.predict_proba(X[oob_mask])
                if probs.shape[1] == 1:
                    pos_prob = np.ones(len(probs)) if tree.classes_[0] == 1 else np.zeros(len(probs))
                else:
                    pos_idx = np.where(tree.classes_ == 1)[0][0]
                    pos_prob = probs[:, pos_idx]

                oob_sum[oob_mask] += pos_prob
                oob_count[oob_mask] += 1
            except (IndexError, ValueError) as e:
                warnings.warn(f"Failed to extract OOB probability for tree {i}: {e}")
                continue

        self.oob_tree_counts_ = oob_count
        valid_mask = oob_count >= self.min_oob_trees
        if np.sum(valid_mask) < 2:
            raise ValueError(f"Insufficient samples >= min_oob_trees={self.min_oob_trees}.")

        oob_probas = oob_sum[valid_mask] / oob_count[valid_mask]
        y_valid = y[valid_mask]

        if self.calibration_method == "isotonic":
            self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator_.fit(oob_probas, y_valid)
        elif self.calibration_method == "platt":
            self.calibrator_ = LogisticRegression()
            self.calibrator_.fit(oob_probas.reshape(-1, 1), y_valid)
        else:
            raise ValueError(f"Unsupported calibration method: {self.calibration_method}")

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predict_proba.")

        raw_probs = self.rf_.predict_proba(X)
        pos_idx = np.where(self.rf_.classes_ == 1)[0][0]
        raw_pos = raw_probs[:, pos_idx]

        if self.calibration_method == "isotonic":
            calibrated_pos = self.calibrator_.predict(raw_pos)
            # Removed redundant clip; IsotonicRegression handles out_of_bounds
        else:
            calibrated_pos = self.calibrator_.predict_proba(raw_pos.reshape(-1, 1))[:, 1]

        return np.column_stack([1.0 - calibrated_pos, calibrated_pos])

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """PyHealth Trainer target API passing logic."""
        X, y = self._extract_x_y(kwargs)

        if self.training and not self.is_fitted_:
            self.fit(X, y)

        if not self.is_fitted_:
            y_prob_np = np.full((len(y),), 0.5)
        else:
            y_prob_np = self.predict_proba(X)[:, 1]

        device = kwargs[self.label_key].device
        y_prob = torch.tensor(y_prob_np, dtype=torch.float32, device=device)
        y_true = kwargs[self.label_key].float()

        y_prob_clamped = torch.clamp(y_prob, 1e-7, 1 - 1e-7)
        loss = nn.BCELoss()(y_prob_clamped, y_true)
        
        # Attach the dummy parameter to the computation graph so PyHealth's 
        # optimizer.backward() doesn't crash with "tensor does not require grad".
        loss = loss + 0.0 * self._dummy_param.sum()
        
        logit = torch.log(y_prob_clamped / (1 - y_prob_clamped))

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }

    def get_oob_calibration_data(self) -> Tuple[np.ndarray, float]:
        """Returns the OOB tree counts distribution."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted.")
        valid_ratio = np.sum(self.oob_tree_counts_ >= self.min_oob_trees) / len(self.oob_tree_counts_)
        return self.oob_tree_counts_, valid_ratio
