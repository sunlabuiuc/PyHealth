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

Undefined measurements return NaN. Class averages exclude undefined classes and
retain genuine zeros; if no defined classes remain, the average is NaN. Empty
batches return NaN without RuntimeWarning.

Behaviour change
----------------

Undefined prediction-set rates now return NaN. This changes:

- ``miscoverage_ps`` and ``error_ps`` return NaN, rather than ``0.0``, for classes
  with no applicable true labels.
- ``error_overall_ps`` returns NaN, rather than ``0.0``, when there are no
  singleton predictions.
- ``miscoverage_overall_ps`` returns NaN, rather than ``0.0``, for an empty
  multiclass batch.
- Scorer values for ``miscoverage_mean_ps`` and ``error_mean_ps`` exclude
  unobserved classes instead of counting them as zeros. If every class is
  undefined, the average is NaN.

Request ``miscoverage_mean_ps`` or ``error_mean_ps`` for class averages, or use
``np.nanmean`` on a class array after checking that at least one rate is defined.
A zero for an unobserved class looks like perfect coverage and can make a class
average look better than the observations support.

Cost-limit comparisons
----------------------

``fp_cost_violation_rate`` counts costs strictly above ``target_cost`` in the
supplied weight units. Costs within relative tolerance ``1e-12`` of the limit
are treated as equal; the absolute tolerance is zero. A positive cost therefore
exceeds a zero limit. Limits may be scalars, zero-dimensional NumPy arrays or
Torch tensors; vectors are rejected.
