pyhealth.tasks.survival_dcm_support2
====================================

Time-to-event survival prediction on :class:`~pyhealth.datasets.Support2Dataset`
in the representation used by Deep Cox Mixtures (Nagpal et al., MLHC 2021): one
sample per patient with a 44-value ``features`` vector and ``duration: [days]``
and ``event: [indicator]`` labels. Event 1 is death; event 0 is censored.

All 9,105 patients are kept. The features are 18 numeric covariates followed by
6 categorical covariates one-hot coded with every level kept, in the order of
``DCM_SUPPORT2_FEATURES``. A missing categorical value sets that covariate's
columns to 0. A missing numeric value is NaN in raw output and 0 after
standardization, which is mean imputation.

This differs from :class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`, which
uses the DeepSurv benchmark representation (8,873 complete cases, 27 features).

Fit ``feature_mean`` and ``feature_scale`` on training patients and reuse them
for validation and test patients. Raw output is only for fitting those
statistics.

.. autoclass:: pyhealth.tasks.SurvivalDCMSupport2
    :members:
    :undoc-members:
    :show-inheritance:
