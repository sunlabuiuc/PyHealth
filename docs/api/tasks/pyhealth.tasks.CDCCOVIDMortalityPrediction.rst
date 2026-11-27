CDC COVID-19 Mortality Prediction
==================================

This case-level binary task predicts the public surveillance field
``death_yn``. It is designed for temporal evaluation by ``case_month`` and
provides demographic-only and expanded pre-outcome feature sets for ablation.
It does not claim that rows belong to unique people or that the retrospective
features were available at a uniform prospective prediction time.

.. autoclass:: pyhealth.tasks.CDCCOVIDMortalityPrediction
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.cdc_covid_mortality.feature_names
