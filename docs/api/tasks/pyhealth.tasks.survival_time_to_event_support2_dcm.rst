pyhealth.tasks.SurvivalTimeToEventSupport2DCM
=============================================

Time to death on SUPPORT2 with the covariates of the Deep Cox Mixtures paper
(`Nagpal et al., MLHC 2021 <https://proceedings.mlr.press/v149/nagpal21a.html>`_).
Use it with :class:`~pyhealth.datasets.Support2Dataset`.

All 9,105 patients are kept. Each sample has ``features: [44]``: 18 numeric
covariates followed by one-hot codes (every level kept) of sex, disease group,
disease class, income, race and cancer status, named by
``SUPPORT2_DCM_FEATURES``. A missing numeric value is NaN and a missing
category is all zeros, as in the paper's released preprocessing. Labels are
``duration: [d.time]`` in days and ``event: [death]``. Each sample also carries
``race`` (the recorded race or ``"missing"``) for subgroup evaluation; the
paper's minority group is every patient whose race is not ``"white"``.

The samples are not imputed. As in the paper, impute missing values with the
mean and standardize every column, fitting both on training patients only,
before training a model such as :class:`~pyhealth.models.DeepCoxMixtures`.

This differs from :class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`, which
keeps the 8,873-patient complete-case DeepSurv cohort with 14 covariates
encoded as 27 features.

.. autoclass:: pyhealth.tasks.SurvivalTimeToEventSupport2DCM
    :members:
    :show-inheritance:
