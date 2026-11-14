pyhealth.metrics.multiclass
===================================

.. currentmodule:: pyhealth.metrics.multiclass

.. autofunction:: multiclass_metrics_fn

Ranking metrics
---------------

``multiclass_metrics_fn`` also reports how highly the true class is ranked among the predicted probabilities:

- ``"hits@K"`` (for any positive integer ``K``, e.g. ``"hits@3"``) returns
    ``HITS@K``, the fraction of samples whose true class is in the top ``K``.
- ``"hits@n"`` returns ``HITS@1``, ``HITS@5`` and ``HITS@10`` together.
- ``"mean_rank"`` returns ``mean_rank`` and ``mean_reciprocal_rank``.

.. code-block:: python

    from pyhealth.metrics import multiclass_metrics_fn

    multiclass_metrics_fn(y_true, y_prob, metrics=["hits@3", "mean_rank"])

Undefined prediction-set measurements
-------------------------------------

Undefined rates return NaN. Class averages exclude classes with no applicable
observations, retain genuine zeros and return NaN if no defined classes remain.
An empty batch or singleton error with no singleton predictions returns NaN::

    import numpy as np
    from pyhealth.metrics import multiclass_metrics_fn

    labels = np.array([0, 1])
    predictions = np.array([[1, 0, 0], [1, 0, 0]])
    result = multiclass_metrics_fn(
        labels,
        predictions.astype(float),
        metrics=["miscoverage_ps", "miscoverage_mean_ps"],
        y_predset=predictions,
    )
    np.testing.assert_allclose(result["miscoverage_ps"], [0, 1, np.nan], equal_nan=True)
    assert result["miscoverage_mean_ps"] == 0.5
