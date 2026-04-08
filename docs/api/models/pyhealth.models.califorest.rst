pyhealth.models.califorest
==========================

.. automodule:: pyhealth.models.califorest
   :members:
   :undoc-members:
   :show-inheritance:

CaliForest: Calibrated Random Forest
------------------------------------

CaliForest is a calibrated Random Forest model that uses Out-of-Bag (OOB) 
predictions for internal calibration, eliminating the need for a separate 
calibration holdout set.

Key Features
^^^^^^^^^^^^

- **No Data Splitting**: Uses OOB predictions for calibration, preserving all 
  training data for the forest
- **Two Calibration Methods**: Supports Isotonic Regression and Platt Scaling
- **Reliability Filtering**: Option to filter unreliable OOB scores based on 
  minimum tree count

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

    from pyhealth.models import CaliForest
    from sklearn.metrics import brier_score_loss, roc_auc_score

    # Initialize model
    model = CaliForest(
        n_estimators=100,
        calibration_method="isotonic",
        min_oob_trees=5,
        random_state=42
    )

    # Fit and predict
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]

    # Evaluate
    auroc = roc_auc_score(y_test, probas)
    brier = brier_score_loss(y_test, probas)

Reference
^^^^^^^^^

If you use this model, please cite the original paper on CaliFores