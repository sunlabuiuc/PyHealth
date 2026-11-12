pyhealth.metrics.binary
===================================

.. currentmodule:: pyhealth.metrics.binary

.. autofunction:: binary_metrics_fn

Prediction-set metrics
----------------------

Pass ``y_predset`` with shape ``(N, 2)``: column 0 includes encoded label 0 and
column 1 includes encoded label 1. ``y_prob`` retains the native positive-class
probability. Prediction-set rates follow the denominators and NaN rules in
:mod:`pyhealth.metrics.prediction_set`. These metrics are skipped when
``y_predset`` is omitted::

    import numpy as np
    from pyhealth.metrics import binary_metrics_fn

    scores = binary_metrics_fn(
        np.array([[0.0], [1.0]]),
        np.array([[0.2], [0.8]]),
        metrics=["accuracy", "set_size", "miscoverage_overall_ps"],
        y_predset=np.array([[1, 0], [1, 1]]),
    )
    assert scores == {"accuracy": 1.0, "set_size": 1.5, "miscoverage_overall_ps": 0.0}
