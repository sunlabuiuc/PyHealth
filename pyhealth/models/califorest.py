"""CaliForest: Calibrated Random Forest using OOB-based Calibration.

This module implements CaliForest, a model that addresses poor calibration
in Random Forests by using Out-of-Bag (OOB) predictions as an internal
calibration set. This avoids the need for a separate holdout set.

Note on PyHealth Architecture: Since CaliForest is a traditional machine
learning ensemble based on Scikit-Learn (not a PyTorch neural network), it
inherits from `BaseEstimator` and `ClassifierMixin` rather than PyHealth's
PyTorch `BaseModel`. It implements `fit` and `predict_proba` instead of a
tensor `forward` pass.
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class CaliForest(BaseEstimator, ClassifierMixin):
    """Calibrated Random Forest using OOB-based calibration.

    CaliForest trains a Random Forest with OOB scoring enabled, then uses
    the OOB predictions to train a calibration model (Isotonic Regression
    or Platt Scaling) without requiring a separate data split.

    Attributes:
        rf_ (RandomForestClassifier): Fitted Random Forest estimator.
        calibrator_ (Union[IsotonicRegression, LogisticRegression]): Fitted
            calibration model.
        classes_ (np.ndarray): Class labels.
        oob_tree_counts_ (np.ndarray): Number of OOB trees per training sample.

    Example:
        >>> import numpy as np
        >>> from pyhealth.models import CaliForest
        >>> X = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> model = CaliForest(n_estimators=50)
        >>> model.fit(X, y)
        >>> probas = model.predict_proba(X[:5])
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        calibration_method: Literal["isotonic", "platt"] = "isotonic",
        min_oob_trees: int = 1,
        use_sample_weights: bool = False,
    ) -> None:
        """Initializes the CaliForest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees. None means unlimited.
            min_samples_split: Minimum samples required to split an internal node.
            min_samples_leaf: Minimum samples required at a leaf node.
            max_features: Number of features to consider for best split.
            random_state: Random seed for reproducibility.
            n_jobs: Number of parallel jobs. -1 means use all processors.
            calibration_method: "isotonic" (Isotonic Regression) or "platt"
                (Logistic Regression).
            min_oob_trees: Minimum number of OOB trees required for a sample
                to be included in the reliable calibration training set.
            use_sample_weights: If True, weights calibration samples by their
                OOB tree count reliability (forces Platt scaling).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.calibration_method = calibration_method
        self.min_oob_trees = min_oob_trees
        self.use_sample_weights = use_sample_weights

    def _compute_oob_predictions(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes average OOB predictions for each training sample.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels.

        Returns:
            Tuple containing:
                - np.ndarray: OOB probability predictions (n_samples,).
                - np.ndarray: Number of OOB trees per sample (n_samples,).
                - np.ndarray: Boolean mask for valid samples with enough trees.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        oob_decision = np.zeros((n_samples, n_classes))
        oob_tree_counts = np.zeros(n_samples, dtype=np.int32)

        for tree, sample_indices in zip(
            self.rf_.estimators_, self.rf_.estimators_samples_
        ):
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[list(sample_indices)] = False

            if oob_mask.sum() == 0:
                continue

            oob_samples = X[oob_mask]

            try:
                tree_pred = tree.predict_proba(oob_samples)
            except Exception:
                continue

            if tree_pred.shape[1] != n_classes:
                full_pred = np.zeros((oob_samples.shape[0], n_classes))
                for j, cls in enumerate(tree.classes_):
                    cls_idx = np.where(self.classes_ == cls)[0]
                    if len(cls_idx) > 0:
                        full_pred[:, cls_idx[0]] = tree_pred[:, j]
                tree_pred = full_pred

            oob_decision[oob_mask] += tree_pred
            oob_tree_counts[oob_mask] += 1

        valid_mask = oob_tree_counts >= self.min_oob_trees
        oob_predictions = np.zeros(n_samples)

        if valid_mask.sum() > 0:
            pos_idx = 1 if n_classes > 1 else 0
            with np.errstate(divide="ignore", invalid="ignore"):
                oob_predictions[valid_mask] = (
                    oob_decision[valid_mask, pos_idx] / oob_tree_counts[valid_mask]
                )

        return oob_predictions, oob_tree_counts, valid_mask

    def _compute_sample_weights(self, oob_tree_counts: np.ndarray) -> np.ndarray:
        """Computes confidence weights based on OOB tree counts.

        Args:
            oob_tree_counts: Array of OOB tree counts per sample.

        Returns:
            np.ndarray: Normalized sample weights.
        """
        max_count = oob_tree_counts.max()
        if max_count == 0:
            return np.ones_like(oob_tree_counts, dtype=float)
        return oob_tree_counts.astype(float) / max_count

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CaliForest":
        """Fits the CaliForest model and internal calibrator.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            CaliForest: The fitted model instance.

        Raises:
            ValueError: If an unsupported calibration method is provided.
            ValueError: If no valid OOB samples meet min_oob_trees criteria.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.calibration_method not in ["isotonic", "platt"]:
            raise ValueError(f"Unsupported calibration: {self.calibration_method}")

        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,
            bootstrap=True,
        )
        self.rf_.fit(X, y)
        self.classes_ = self.rf_.classes_

        oob_predictions, self.oob_tree_counts_, valid_mask = (
            self._compute_oob_predictions(X, y)
        )

        if valid_mask.sum() == 0:
            raise ValueError(
                f"No samples have >= {self.min_oob_trees} OOB trees. "
                "Decrease min_oob_trees or increase n_estimators."
            )

        oob_preds_valid = oob_predictions[valid_mask]
        y_valid = y[valid_mask]

        if self.use_sample_weights:
            weights = self._compute_sample_weights(self.oob_tree_counts_[valid_mask])
            self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(
                oob_preds_valid.reshape(-1, 1), y_valid, sample_weight=weights
            )
            self._calibration_type = "platt"
        elif self.calibration_method == "isotonic":
            self.calibrator_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator_.fit(oob_preds_valid, y_valid)
            self._calibration_type = "isotonic"
        else:
            self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(oob_preds_valid.reshape(-1, 1), y_valid)
            self._calibration_type = "platt"

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts calibrated class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        check_is_fitted(self, ["rf_", "calibrator_", "classes_"])

        X = np.asarray(X)
        raw_proba = self.rf_.predict_proba(X)[:, 1]

        if self._calibration_type == "isotonic":
            calibrated_proba = self.calibrator_.predict(raw_proba)
        else:
            calibrated_proba = self.calibrator_.predict_proba(
                raw_proba.reshape(-1, 1)
            )[:, 1]

        calibrated_proba = np.clip(calibrated_proba, 0.0, 1.0)
        return np.column_stack([1 - calibrated_proba, calibrated_proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class labels of shape (n_samples,).
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba to align loosely with PyHealth conventions.

        Args:
            X: Features of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        return self.predict_proba(X)

    def get_oob_calibration_data(self) -> Tuple[np.ndarray, float]:
        """Retrieves OOB prediction metadata.

        Returns:
            Tuple containing:
                - np.ndarray: Array of OOB tree counts.
                - float: Ratio of samples that met the minimum OOB tree threshold.
        """
        check_is_fitted(self, ["oob_tree_counts_"])
        valid_ratio = (self.oob_tree_counts_ >= self.min_oob_trees).mean()
        return self.oob_tree_counts_, float(valid_ratio)
