[core] prediction_set
===================================

.. automodule:: pyhealth.metrics.prediction_set
    :members:




Denominators and undefined results
----------------------------------

Set size, rejection rate, weighted cost/value and cost-limit violation rate
average over all samples. Overall multilabel miscoverage averages each sample's
missed fraction, excluding samples with no true labels. Per-class rates use
samples containing that class; singleton errors additionally require set size one.

Multiclass targets must have an integer dtype and index prediction-set columns.

Undefined measurements return NaN. Class averages exclude undefined classes and
retain genuine zeros; if no defined classes remain, the average is NaN. Empty
batches return NaN without RuntimeWarning.

Request ``miscoverage_mean_ps`` or ``error_mean_ps`` for class averages, or use
``np.nanmean`` on a class array after checking that at least one rate is defined.
A zero for an unobserved class looks like perfect coverage and can make a class
average look better than the observations support.

Cost-limit comparisons
----------------------

``fp_cost_violation_rate`` counts costs strictly above ``target_cost`` in the
supplied weight units. The relative tolerance is the larger of ``1e-12`` and
the floating-point precision of the weights and limit. Costs within that
tolerance are treated as equal; the absolute tolerance is zero. A positive cost therefore
exceeds a zero limit. Limits may be scalars, zero-dimensional NumPy arrays or
Torch tensors; vectors are rejected.
