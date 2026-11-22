pyhealth.tasks.Support2SurvivalTask
===================================

Censored time-to-death samples from :class:`~pyhealth.datasets.Support2Dataset`
for survival models such as :class:`~pyhealth.models.DeepCoxMixtures`. Each
sample has the 24 raw covariates used by Deep Cox Mixtures (missing values are
``None``), the follow-up ``time`` (``d.time``, days) and ``event``
(``death``). No preprocessing is fitted, so each training split can fit its
own imputation, encoding and scaling.

Unlike :class:`~pyhealth.tasks.SurvivalPreprocessSupport2`, which predicts the
dataset's model-derived survival probability, and
:class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`, which follows the
DeepSurv/Survival MDN 27-feature complete-case protocol, this task keeps every
row, the raw covariates and the censoring indicator.

.. autoclass:: pyhealth.tasks.Support2SurvivalTask
    :members:
    :show-inheritance:

.. autodata:: pyhealth.tasks.support2_survival.SUPPORT2_DCM_FEATURES
