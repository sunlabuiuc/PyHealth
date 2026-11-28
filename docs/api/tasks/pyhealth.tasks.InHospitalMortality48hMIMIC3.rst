pyhealth.tasks.InHospitalMortality48hMIMIC3
=============================================

48-hour in-hospital mortality on MIMIC-III, as 34 raw-unit summary
statistics (mean and standard deviation of 17 clinical variables),
reproducing the feature extraction of Hegselmann et al. (2020) (itself
adapted from the Harutyunyan et al. 2019 benchmark). Works with
:class:`~pyhealth.datasets.MIMIC3Dataset`.

Missing values are kept as an explicit sentinel (``-1.0``) rather than
imputed or z-scored, matching :class:`~pyhealth.models.GA2M`'s expectation
of a dedicated "missing" bin. See ``examples/mimic3_mortality_ga2m.py`` for
a full reproduction pipeline against that model.

.. autoclass:: pyhealth.tasks.in_hospital_mortality_48h_mimic3.InHospitalMortality48hMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:
