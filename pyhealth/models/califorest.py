import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from pyhealth.models.base_model import BaseModel


class CaliForest(ClassifierMixin, BaseEstimator):
    """
    CaliForest is a class that implements a random forest classifier using the CaliForest algorithm.

    Y. Park and J. C. Ho. 2020. CaliForest: Calibrated Random Forest for Health Data. ACM Conference on Health, Inference, and Learning (2020)
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        ctype="isotonic",
        alpha0=100,
        beta0=25,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.estimators = []
        self.calibrator = None

        # Create decision tree estimators
        for _ in range(self.n_estimators):
            self.estimators.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features="sqrt",
                )
            )

        # Setup calibrators
        if self.ctype == "logistic":
            self.calibrator = LogisticRegression(
                penalty=None, solver="saga", max_iter=5000
            )
        elif self.ctype == "isotonic":
            self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")

        # Begin out-of-bag training setup
        n, m = X.shape
        Y_oob = np.full((n, self.n_estimators), np.nan)
        n_oob = np.zeros(n)
        IB = np.zeros((n, self.n_estimators), dtype=int)
        OOB = np.full((n, self.n_estimators), True)

        for eid in range(self.n_estimators):
            IB[:, eid] = np.random.choice(n, n)
            OOB[IB[:, eid], eid] = False

        for eid, est in enumerate(self.estimators):
            ib_idx = IB[:, eid]
            oob_idx = OOB[:, eid]
            est.fit(X[ib_idx, :], y[ib_idx])
            Y_oob[oob_idx, eid] = est.predict_proba(X[oob_idx, :])[:, 1]
            n_oob[oob_idx] += 1

        # Get out-of-bag predictions
        oob_idx = n_oob > 1
        Y_oob_ = Y_oob[oob_idx, :]
        n_oob_ = n_oob[oob_idx]
        z_hat = np.nanmean(Y_oob_, axis=1)
        z_true = y[oob_idx]

        beta = self.beta0 + np.nanvar(Y_oob_, axis=1) * n_oob_ / 2
        alpha = self.alpha0 + n_oob_ / 2
        z_weight = alpha / beta

        # Fit calibrators
        if self.ctype == "logistic":
            self.calibrator.fit(z_hat[:, np.newaxis], z_true, z_weight)
        elif self.ctype == "isotonic":
            self.calibrator.fit(z_hat, z_true, z_weight)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n, m = X.shape
        n_est = len(self.estimators)
        z = np.zeros(n)
        y_mat = np.zeros((n, 2))

        # Get base predictions from all trees
        for _, est in enumerate(self.estimators):
            z += est.predict_proba(X)[:, 1]
        z /= n_est

        # Use standard calibration with single calibrator
        if self.ctype == "logistic":
            y_mat[:, 1] = self.calibrator.predict_proba(z[:, np.newaxis])[:, 1]
        elif self.ctype == "isotonic":
            y_mat[:, 1] = self.calibrator.predict(z)

        y_mat[:, 0] = 1 - y_mat[:, 1]
        return y_mat

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
