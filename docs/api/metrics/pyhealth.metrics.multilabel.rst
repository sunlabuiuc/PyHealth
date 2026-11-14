pyhealth.metrics.multilabel
===================================

.. currentmodule:: pyhealth.metrics.multilabel

.. autofunction:: multilabel_metrics_fn

The ``precision@K``, ``recall@K`` and ``ndcg@K`` metrics are computed by
:func:`pyhealth.metrics.ranking.multilabel_ranking_at_k`.


Metric order
------------

``multilabel_metrics_fn`` walks the ``metrics`` list one entry at a time, and
no entry changes the arrays the others read, so the order of the list does not
change the numbers. ``metrics=["ddi", "jaccard_samples"]`` and
``metrics=["jaccard_samples", "ddi"]`` return the same values.

DDI matrix
----------

``"ddi"`` needs the drug-drug interaction matrix of the label vocabulary. Pass
it as ``ddi_adj``. ARMR, GAMENet, MICRON, MoleRec and SafeDrug store theirs in
``model.ddi_adj_matrix``, and ``Trainer.evaluate`` passes that matrix for you
when the evaluated model has one:

.. code-block:: python

    scores = multilabel_metrics_fn(
        y_true, y_prob, metrics=["ddi"], ddi_adj=model.ddi_adj_matrix
    )

Without ``ddi_adj``, ``"ddi"`` loads ``ddi_adj.npy`` from the PyHealth cache
directory, and raises ``FileNotFoundError`` if it is missing. The five models
write that file when they are built (atomically, so a reader never sees a
partial file), but it is shared: it holds the matrix of whichever model was
built last, in any process. If another run or a model with a different drug
vocabulary was built in between, the file belongs to that model, so pass
``ddi_adj`` whenever you call the function yourself.

Prediction-set label miscoverage
--------------------------------

``miscoverage_overall_ps`` computes the fraction of true labels omitted from the
prediction set for each sample, then averages these fractions with equal weight.
Samples with no true labels are excluded; if all targets are empty, the result is
NaN. This differs from the fraction of samples missing any true label::

    import numpy as np
    from pyhealth.metrics import multilabel_metrics_fn

    y_true = np.array([[1, 1, 0], [0, 0, 1]])
    y_predset = np.array([[1, 0, 1], [0, 0, 1]])
    result = multilabel_metrics_fn(
        y_true,
        np.zeros_like(y_true, dtype=float),  # Probabilities are unused here.
        metrics=["miscoverage_overall_ps"],
        y_predset=y_predset,
    )
    assert result == {"miscoverage_overall_ps": 0.25}  # Mean of 1/2 and 0.

The metric uses ``y_predset`` directly. Like the existing ``tp`` and ``fp``
metrics, it is skipped when ``y_predset`` is omitted.

Coverage is ``1 - result["miscoverage_overall_ps"]``. The same metric name is
already supported by the multiclass scorer for one-dimensional class labels.

Set size and per-class miscoverage
----------------------------------

``set_size`` uses the existing prediction-set size helper. ``miscoverage_ps``
returns one miss rate per encoded class, counting only samples whose true labels
contain that class. An unobserved class returns NaN. ``miscoverage_mean_ps``
averages the defined class rates with equal weight per class, excluding NaNs.
If every class is undefined, this average is NaN. These class averages generally
differ from ``miscoverage_overall_ps``, which gives equal weight to samples.

Weighted false-positive cost and true-positive value
----------------------------------------------------

``fp_cost`` computes ``mean(sum(cost_weights * S * (1 - Y)))``.
``tp_value`` computes ``mean(sum(value_weights * S * Y))``. Weights are finite,
non-negative scalars or vectors in encoded class order. Values retain the
original units of the supplied weights; they are not normalized::

    import numpy as np
    from pyhealth.metrics import multilabel_metrics_fn

    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_predset = np.array([[1, 1, 1], [1, 1, 0]])
    result = multilabel_metrics_fn(
        y_true,
        np.zeros_like(y_true, dtype=float),
        metrics=["fp", "tp", "fp_cost", "tp_value", "set_size"],
        y_predset=y_predset,
        cost_weights=[2, 3, 5],
        value_weights=[2, 3, 5],
    )
    assert result == {
        "fp": 1.0, "tp": 1.5, "fp_cost": 2.5, "tp_value": 5.0, "set_size": 2.5,
    }

Unit weights make ``fp_cost`` equal ``fp`` and ``tp_value`` equal ``tp``.
Supplying weights never changes the existing unweighted ``fp``/``tp`` metrics.
All four average over every sample, including those with empty true-label sets;
an empty batch returns NaN. All prediction-set metrics are skipped if
``y_predset`` is absent. Missing or undefined measurements are not converted
into zero or JSON null here: serialization belongs to the caller.

Cost-limit violation probability
--------------------------------

The probability of exceeding a cost limit cannot be recovered from the mean cost
alone. ``fp_cost_violation_rate`` uses each sample's weighted false-positive cost
and counts strictly greater costs; values equal to the limit are not violations.
It uses all samples, even when true label sets are empty, and returns NaN for an
empty batch. Supply a finite,
non-negative target_cost explicitly; omitted limits are rejected when the metric
is requested with prediction sets::

    import numpy as np
    from pyhealth.metrics import multilabel_metrics_fn

    y_true = np.zeros((2, 2), dtype=int)
    y_predset = np.array([[0, 0], [1, 1]])
    result = multilabel_metrics_fn(
        y_true,
        np.zeros_like(y_true, dtype=float),
        metrics=["fp_cost", "fp_cost_violation_rate"],
        y_predset=y_predset,
        cost_weights=1.0,
        target_cost=1.0,
    )
    assert result == {"fp_cost": 1.0, "fp_cost_violation_rate": 0.5}
