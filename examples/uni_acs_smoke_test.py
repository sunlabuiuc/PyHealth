import numpy as np
import pandas as pd
import shap

from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pyhealth.interpret.uni_acs import UniACS


# Generate synthetic data
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=0,
    random_state=42,
)

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Train and calibrate RandomForest
model = CalibratedClassifierCV(
    estimator=RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    method="isotonic",
    cv=3,
)
model.fit(X_train, y_train)

baseline_prob = model.predict_proba(X_test)[:, 1]


# Calculate log-odds SHAP contributions
shap_list = []
expected_values = []

for clf in model.calibrated_classifiers_:
    explainer = shap.TreeExplainer(clf.estimator)
    values = explainer.shap_values(X_train)

    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    base_prob = float(explainer.expected_value[1])
    adjusted = np.clip(values + base_prob, 0.0001, 0.9999)

    shap_list.append(logit(adjusted) - logit(base_prob))
    expected_values.append(logit(base_prob))

logodds_shap = np.mean(shap_list, axis=0)
expected_value = np.mean(expected_values)


# Compare scoring methods
print("Baseline AUROC:", round(roc_auc_score(y_test, baseline_prob), 4))
print()

for method in ["quantile", "novel"]:
    uni_acs = UniACS(top_k=5, method=method)
    uni_acs.fit(X_train, logodds_shap, expected_value)

    scores = uni_acs.predict_score(X_test)
    probs = uni_acs.predict_proba(X_test)
    table = uni_acs.get_score_table()

    print("Method:", method)
    print("Categories:", len(table))
    print("AUROC:", round(roc_auc_score(y_test, scores), 4))
    print("AUPRC:", round(average_precision_score(y_test, probs), 4))
    print("Brier:", round(brier_score_loss(y_test, probs), 4))
    print()
