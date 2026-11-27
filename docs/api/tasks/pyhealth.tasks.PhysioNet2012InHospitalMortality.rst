pyhealth.tasks.PhysioNet2012InHospitalMortality
===============================================

In-hospital mortality prediction on PhysioNet 2012, in the input layout of
TransEHR (Xu et al., "TransEHR: Self-Supervised Transformer for Clinical Time
Series Data", ML4H 2023, https://proceedings.mlr.press/v225/xu23a.html). The
first 48 hours of each ICU stay become

* an hourly grid of the 36 time series (hourly means, 0 where unobserved, with
  an ``observed`` mask and a ``padding_mask``),
* an irregular stream of lab observation times and types (padded to 400 events),
* eight static features (age, gender, ICU type, height),

and the label is in-hospital death (about 14% of the 12,000 stays). The
``feature_groups`` option selects vitals only, labs only or both, and
``icu_types`` restricts the cohort, so input variants can be compared with a
single task definition.

Use it with :class:`~pyhealth.datasets.PhysioNet2012Dataset`. Normalize with
:func:`~pyhealth.tasks.physionet2012_normalization` over the training records
only.

.. autoclass:: pyhealth.tasks.PhysioNet2012InHospitalMortality
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.physionet2012_normalization

.. autofunction:: pyhealth.tasks.build_mortality_sample
