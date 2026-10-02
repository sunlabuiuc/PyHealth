pyhealth.datasets.splitter
===================================

Several data splitting function for `pyhealth.datasets` module to obtain training / validation / test sets.

Split ratio requirements
------------------------

Ratios must be provided as a list, tuple, or one-dimensional NumPy array.
`split_by_visit`, `split_by_patient`, and `split_by_sample` require three
values. Their conformal variants require four values, ordered as
train / validation / calibration / test. `split_by_patient_tuh` and
`split_by_sample_tuh` require two values; their conformal variants require
three. Every value must be a finite real number in the inclusive range
`[0, 1]`.

The generic splitters use ratios for the returned partitions in the order
described above. TUH splitters use ratios to divide the official training
pool; the official evaluation pool remains the returned test partition.

Ratios must sum to `1.0` within an absolute tolerance of `1e-6`. Values are
not normalized. Negative, greater-than-one, NaN, or infinite values and
incorrect totals raise `ValueError`. Unsupported ratio containers or
non-real elements raise `TypeError`. Validation runs before the dataset is
accessed and uses explicit exceptions, so it remains active when Python runs
with optimization enabled.

Zero ratios are allowed, and integer rounding can produce empty partitions
for small datasets. The splitter assigns any trailing remainder according to
its existing partition logic. Patient-level splitting keeps each patient in
one partition; sample-level splitting does not provide that patient-level
separation.

See `examples/split_ratio_validation.py` for a runnable synthetic example.

.. automodule:: pyhealth.datasets.splitter
    :members:
    :undoc-members:
    :show-inheritance:
