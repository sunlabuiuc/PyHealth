import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class STLBRF(ClassifierMixin, BaseEstimator):
    """
    standardized threshold, and loops based random forest

    Feng H, Ju Y, Yin X, Qiu W, Zhang X.
    STLBRF: an improved random forest algorithm based on standardized-threshold
    for feature screening of gene expression data.
    Brief Funct Genomics.
    2025;24:elae048. doi:10.1093/bfgp/elae048

    Feature selection algorithm that works by sequentially reducing features
    whose importance falls below a specified percentile, while maintaining
    a performance threshold.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        error_increment=0.01,
        importance_percentile_threshold=0.25,  # New parameter
        n_splits=5,
        min_features=2,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.error_increment = error_increment
        self.importance_percentile_threshold = (
            importance_percentile_threshold  # New parameter
        )
        self.n_splits = n_splits
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        # Input validation
        X, y = check_X_y(X, y)

        # Initialize model and k-fold cross-validation
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.kf_ = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        # Start with all features
        feature_mask = np.arange(X.shape[1])
        self.best_score_ = 0.0
        self.selected_features_ = np.copy(
            feature_mask
        )  # Keep track of the best set of features

        # Sequential feature elimination
        while len(feature_mask) > self.min_features:
            fold_scores = []
            # We need to fit on the current feature set to get importances
            X_current = X[:, feature_mask]
            self.rf_.fit(
                X_current, y
            )  # Fit on the whole data (temporarily) for importances

            for train_idx, val_idx in self.kf_.split(X):
                X_train, X_val = (
                    X[train_idx][:, feature_mask],
                    X[val_idx][:, feature_mask],
                )
                y_train, y_val = y[train_idx], y[val_idx]
                # Fit RF on the training fold using current features
                rf_fold = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                rf_fold.fit(X_train, y_train)
                preds = rf_fold.predict(X_val)
                fold_scores.append(accuracy_score(y_val, preds))

            mean_score = np.mean(fold_scores)

            # If performance drops below threshold, stop and revert to previous best features
            if mean_score < (self.best_score_ - self.error_increment):
                break  # Stop elimination

            # Update best score and store current feature mask as the best one so far
            self.best_score_ = max(self.best_score_, mean_score)
            self.selected_features_ = np.copy(
                feature_mask
            )  # Store the features that gave this score

            # Calculate feature importances based on the fit on the whole current data
            importances = self.rf_.feature_importances_

            # Determine the importance threshold based on the percentile
            # We want to KEEP features above this threshold, so we use (1 - threshold)
            # e.g., if threshold is 0.25, we keep top 25% features, so we remove those below 75th percentile
            importance_threshold = np.percentile(
                importances, (1 - self.importance_percentile_threshold) * 100
            )

            # Identify indices (relative to current mask) of features below the threshold
            indices_below_threshold = np.where(importances < importance_threshold)[0]

            # If no features are below the threshold, or removing them leaves too few, stop
            if len(indices_below_threshold) == 0:
                break  # Stop if no features to remove

            num_remaining_features = len(feature_mask) - len(indices_below_threshold)
            if num_remaining_features < self.min_features:
                break  # Stop if removal leads to too few features

            # Remove the features below the threshold
            feature_mask = np.delete(feature_mask, indices_below_threshold)

        # Final training on the best selected features with all data
        X_selected = X[:, self.selected_features_]
        self.rf_.fit(X_selected, y)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply feature selection to X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_"])
        X = check_array(X)
        # Ensure X has the original number of features before selection
        if X.shape[1] != len(
            np.arange(X.shape[1])
        ):  # A bit redundant check, usually check_array handles dim
            pass  # Should align with original features count if check_array passes
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Predict class for X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_"])
        # Transform uses self.selected_features_ which are the original indices
        X_transformed = self.transform(X)
        return self.rf_.predict(X_transformed)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_"])
        # Transform uses self.selected_features_ which are the original indices
        X_transformed = self.transform(X)
        return self.rf_.predict_proba(X_transformed)
