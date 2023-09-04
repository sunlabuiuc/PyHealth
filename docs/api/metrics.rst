Metrics
===============

We provide easy to use metrics (the same style and args as `sklearn.metrics`) for binary classification,
multiclass classification, multilabel classification.
For applicable tasks, we provide the relevant metrics for model calibration, as well as those for prediction set evaluation.
Among these we also provide metrics related to uncertainty quantification, for model calibration, as well as metrics that measure the quality of prediction sets
We also provide other metrics specically for healthcare
tasks, such as drug drug interaction (DDI) rate.


.. toctree::
    :maxdepth: 3

    metrics/pyhealth.metrics.multiclass
    metrics/pyhealth.metrics.multilabel
    metrics/pyhealth.metrics.binary
    metrics/pyhealth.metrics.calibration
    metrics/pyhealth.metrics.prediction_set
    metrics/pyhealth.metrics.fairness
