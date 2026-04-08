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
    or Platt Scaling) without requiring a separate calibration set.

    Args:
        n_estimators: Number of trees in the forest. Defaults to 100.
        max_depth: Maximum depth of trees. None means unlimited.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Number of features to consider for best split.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs. -1 uses all processors.
        calibration_method: Calibration method to use.
            - "isotonic": Isotonic Regression (default)
            - "platt": Platt Scaling (Logistic Regression)
        min_oob_trees: Minimum number of OOB trees required for a sample
            to be included in calibration training.
        use_sample_weights: If True, weight calibration samples by OOB
            tree count (novel extension).

    Attributes:
        rf_: Fitted RandomForestClassifier.
        calibrator_: Fitted calibration model.
        classes_: Class labels.
        oob_tree_counts_: Number of OOB trees per training sample.

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
        """Compute OOB predictions for each training sample.

        For each sample, predictions are averaged only from trees where
        that sample was out-of-bag.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels (used to determine n_classes).

        Returns:
            Tuple containing:
                - oob_predictions: OOB probability predictions (n_samples,)
                - oob_tree_counts: Number of OOB trees per sample (n_samples,)
                - valid_mask: Boolean mask for samples with enough OOB trees
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Accumulate predictions from OOB trees
        oob_decision = np.zeros((n_samples, n_classes))
        oob_tree_counts = np.zeros(n_samples, dtype=np.int32)

        # Get the indices of samples used to train each tree
        # estimators_samples_ contains the indices of samples in the bootstrap
        for i, (tree, sample_indices) in enumerate(
            zip(self.rf_.estimators_, self.rf_.estimators_samples_)
        ):
            # Create mask for OOB samples (samples NOT in this tree's bootstrap)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[list(sample_indices)] = False

            if oob_mask.sum() == 0:
                continue

            # Get predictions for OOB samples
            oob_samples = X[oob_mask]
            
            try:
                tree_pred = tree.predict_proba(oob_samples)
            except Exception:
                continue

            # Handle case where tree may not have seen all classes
            if tree_pred.shape[1] != n_classes:
                full_pred = np.zeros((oob_samples.shape[0], n_classes))
                for j, cls in enumerate(tree.classes_):
                    cls_idx = np.where(self.classes_ == cls)[0]
                    if len(cls_idx) > 0:
                        full_pred[:, cls_idx[0]] = tree_pred[:, j]
                tree_pred = full_pred

            oob_decision[oob_mask] += tree_pred
            oob_tree_counts[oob_mask] += 1

        # Compute average predictions
        valid_mask = oob_tree_counts >= self.min_oob_trees
        oob_predictions = np.zeros(n_samples)

        if valid_mask.sum() > 0:
            # Get probability of positive class (class index 1)
            positive_class_idx = 1 if n_classes > 1 else 0
            with np.errstate(divide='ignore', invalid='ignore'):
                oob_predictions[valid_mask] = (
                    oob_decision[valid_mask, positive_class_idx] 
                    / oob_tree_counts[valid_mask]
                )

        return oob_predictions, oob_tree_counts, valid_mask

    def _compute_sample_weights(self, oob_tree_counts: np.ndarray) -> np.ndarray:
        """Compute confidence weights based on OOB tree counts.

        Samples with more OOB trees are more reliable and receive higher
        weight during calibration training.

        Args:
            oob_tree_counts: Number of OOB trees per sample.

        Returns:
            Normalized weights array.
        """
        max_count = oob_tree_counts.max()
        if max_count == 0:
            return np.ones_like(oob_tree_counts, dtype=float)
        weights = oob_tree_counts.astype(float) / max_count
        return weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CaliForest":
        """Fit the CaliForest model.

        This involves:
        1. Training a Random Forest with OOB scoring
        2. Computing OOB predictions for training samples
        3. Training calibration model on (OOB prediction, true label) pairs

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            Self: The fitted model.

        Raises:
            ValueError: If calibration_method is not supported.
            ValueError: If no valid OOB samples available for calibration.
        """
        # Validate inputs
        X = np.asarray(X)
        y = np.asarray(y)

        if self.calibration_method not in ["isotonic", "platt"]:
            raise ValueError(
                f"calibration_method must be 'isotonic' or 'platt', "
                f"got '{self.calibration_method}'"
            )

        # Step 1: Train Random Forest with OOB enabled
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

        # Step 2: Compute OOB predictions
        oob_predictions, self.oob_tree_counts_, valid_mask = (
            self._compute_oob_predictions(X, y)
        )

        if valid_mask.sum() == 0:
            raise ValueError(
                f"No samples have at least {self.min_oob_trees} OOB trees. "
                "Try reducing min_oob_trees or increasing n_estimators."
            )

        # Step 3: Train calibration model on valid OOB samples
        oob_preds_valid = oob_predictions[valid_mask]
        y_valid = y[valid_mask]

        if self.use_sample_weights:
            # Novel: weight by OOB reliability (forces Platt scaling)
            weights = self._compute_sample_weights(
                self.oob_tree_counts_[valid_mask]
            )
            self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(
                oob_preds_valid.reshape(-1, 1),
                y_valid,
                sample_weight=weights
            )
            self._calibration_type = "platt"
        elif self.calibration_method == "isotonic":
            self.calibrator_ = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator_.fit(oob_preds_valid, y_valid)
            self._calibration_type = "isotonic"
        else:  # platt
            self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator_.fit(oob_preds_valid.reshape(-1, 1), y_valid)
            self._calibration_type = "platt"

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Calibrated probabilities of shape (n_samples, n_classes).
        """
        check_is_fitted(self, ["rf_", "calibrator_", "classes_"])

        X = np.asarray(X)

        # Get raw RF predictions (probability of positive class)
        raw_proba = self.rf_.predict_proba(X)[:, 1]

        # Apply calibration
        if self._calibration_type == "isotonic":
            calibrated_proba = self.calibrator_.predict(raw_proba)
        else:  # platt
            calibrated_proba = self.calibrator_.predict_proba(
                raw_proba.reshape(-1, 1)
            )[:, 1]

        # Ensure valid probability range
        calibrated_proba = np.clip(calibrated_proba, 0.0, 1.0)

        # Return probabilities for both classes
        return np.column_stack([1 - calibrated_proba, calibrated_proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_oob_calibration_data(self) -> Tuple[np.ndarray, float]:
        """Get the OOB predictions and tree counts used for calibration.

        Returns:
            Tuple of (oob_tree_counts, valid_sample_ratio).

        Raises:
            NotFittedError: If model has not been fitted.
        """
        check_is_fitted(self, ["oob_tree_counts_"])
        valid_ratio = (self.oob_tree_counts_ >= self.min_oob_trees).mean()
        return self.oob_tree_counts_, valid_ratio