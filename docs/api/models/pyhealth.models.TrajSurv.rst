pyhealth.models.TrajSurv
========================

Overview
--------

TrajSurv (Zeng et al., `TrajSurv: Learning Continuous Latent Trajectories from
Electronic Health Records for Trustworthy Survival Prediction
<https://arxiv.org/abs/2508.00657>`_, MLHC 2025) encodes a patient's hourly
labs and vitals with a neural controlled differential equation (CDE) into a
continuous latent trajectory, and predicts the risk of an event with a Cox
head on the last latent state. An optional Rank-N-Contrast term aligns the
latent states with a clinical progression signal; the paper uses the SOFA
score and its trend.

This is a port of the authors' code (https://github.com/zengsihang/TrajSurv,
MIT License). The CDE is solved in pure PyTorch (natural cubic spline path,
fixed-step Runge-Kutta 4), so ``torchcde`` is not required. The DSM-Weibull
head and the clustering and vector-field analyses are not ported.

Inputs and outputs
------------------

The input is an hourly series: the ``dynamic`` field of
:class:`~pyhealth.tasks.InHospitalSurvivalMIMIC4`, or any dense ``[T, F]``
tensor field. Labels follow :class:`~pyhealth.models.SurvivalMDN`:
``duration`` and ``event`` tensor outputs. The model returns ``risk`` (log
hazard ratio; higher means earlier death), ``loss`` (Cox partial likelihood,
plus the alignment loss when enabled), ``y_true`` (duration, event) and
``y_prob``, survival curves on ``time_grid``. Call ``fit_baseline_hazard`` on
the training data to calibrate the survival curves (Breslow estimator).

Evaluate with :mod:`pyhealth.metrics.survival`. ``mode`` is ``None``, as for
SurvivalMDN, so use a survival training loop and these metrics rather than
the classification metric dispatcher. The example notebook
``examples/trajsurv_mimic4_survival.ipynb`` trains TrajSurv, SurvivalMDN and a
linear Cox baseline on MIMIC-IV.

API
---

.. autoclass:: pyhealth.models.TrajSurv
   :members:
   :show-inheritance:

.. autofunction:: pyhealth.models.trajsurv.cox_partial_likelihood_loss

.. autofunction:: pyhealth.models.trajsurv.rank_n_contrast_loss
