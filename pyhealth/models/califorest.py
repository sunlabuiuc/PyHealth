from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

class Califorest:
    def __init__(self,
                 n_estimators=300,
                 criterion='gini',
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 ctype='isotonic',
                 alpha0=100,
                 beta0=25):
        """
        Implementation of CaliForest that works with modern scikit-learn

        Parameters:
        -----------
        n_estimators : int, default=300
            Number of trees in the forest
        criterion : str, default='gini'
            Function to measure the quality of a split
        max_depth : int, default=5
            Maximum depth of the trees
        min_samples_split : int, default=2
            Minimum number of samples required to split a node
        min_samples_leaf : int, default=1
            Minimum number of samples required at a leaf node
        ctype : str, default='isotonic'
            Calibration method ('isotonic' or 'sigmoid')
        alpha0, beta0 : float, default=100, 25
            Prior parameters for calibration
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ctype = ctype
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Create base estimator
        self.base_estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",  # Using 'sqrt' instead of 'auto'
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X, y):
        """Fit the calibrated classifier"""
        # Use 'estimator' instead of 'base_estimator'
        self.calibrated_clf = CalibratedClassifierCV(
            estimator=self.base_estimator,  # Updated parameter name
            method=self.ctype,
            cv=5
        )
        self.calibrated_clf.fit(X, y)
        self.classes_ = self.calibrated_clf.classes_
        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.calibrated_clf.predict_proba(X)

    def predict(self, X):
        """Predict class labels"""
        return self.calibrated_clf.predict(X)