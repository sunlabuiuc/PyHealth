import numpy as np
from sklearn.ensemble import RandomForestClassifier


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from venn_abers import VennAbers


class VennAbersRandomForests(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=30,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )

        # Fit the model on all training data
        self.model.fit(X, y)

        # Fit the calibrator on the same data
        self.calibrator = VennAbers()
        y_est = self.model.predict_proba(X)
        self.calibrator.fit(y_est, y)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        # Get raw predictions from the model first
        raw_proba = self.model.predict_proba(X)

        # Then calibrate these predictions
        return self.calibrator.predict_proba(raw_proba)[0]

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
