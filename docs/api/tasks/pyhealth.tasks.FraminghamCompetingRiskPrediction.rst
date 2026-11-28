pyhealth.tasks.FraminghamCompetingRiskPrediction
================================================

Creates one continuous-time competing-risk sample per Framingham participant:
the 18 raw baseline covariates, follow-up ``time`` and an ``event`` cause
(0=censored, 1=cardiovascular disease, 2=death). Keys match
:class:`~pyhealth.tasks.CompetingRisksPBC`, so samples go through
:class:`~pyhealth.tasks.CompetingRisksPreprocessor` and
:class:`~pyhealth.models.NeuralFineGray` unchanged. Optional administrative
censoring supports horizon ablations.

.. autoclass:: pyhealth.tasks.FraminghamCompetingRiskPrediction
   :members:
   :show-inheritance:
