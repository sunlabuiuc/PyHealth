pyhealth.models.DeepCoxMixtures
===============================

Overview
--------

Deep Cox Mixtures (DCM) for right-censored survival regression, following
`Nagpal et al., MLHC 2021 <https://proceedings.mlr.press/v149/nagpal21a.html>`_.
A patient's survival curve is a mixture of ``K`` Cox models,

.. math::

    S(t \mid x) = \sum_{k=1}^{K} \pi_k(x)\, S_{0k}(t)^{\exp f_k(x)},

where a shared MLP encoder feeds a gating head (latent-group probabilities
:math:`\pi_k`) and a head of per-group log hazard ratios :math:`f_k`, and each
group has its own Breslow baseline survival :math:`S_{0k}`. Proportional
hazards hold within a group but not across the mixture, so the model can
represent patient groups whose survival curves cross. ``num_components=1``
gives a neural Cox model (DeepSurv-style) with a Breslow baseline, and
``hidden_dims=()`` gives the paper's linear "Cox Mixture".

Training
--------

:meth:`~pyhealth.models.DeepCoxMixtures.fit` runs the paper's approximate
Monte Carlo EM. For each mini-batch it computes the posterior group
responsibilities from the observed time and censoring status (E-step), samples
hard assignments from them, takes one Adam step on the gate log-likelihood plus
one Cox partial likelihood per sampled group (Breslow ties, mini-batch risk
sets), and then refits every group's Breslow baseline on the full training set.
With validation data it stops early on the validation negative log-likelihood
and restores the best weights and baselines. The generic
:class:`~pyhealth.trainer.Trainer` is not used because the baselines have to be
refit between gradient steps.

Each group's cumulative baseline hazard is interpolated linearly between its
own event times (a piecewise-constant hazard) and held constant after its last
event time, so survival curves stay in [0, 1], never increase, and event
densities in the E-step are nonnegative. The paper smooths the Breslow curve
with a cubic spline instead.

Inputs and outputs
------------------

Features must be one dense, finite vector per patient (impute and scale them on
training patients first). Labels are ``duration`` (strictly positive) and
``event`` (1 for an event, 0 for censoring); the keys are configurable with
``feature_key``, ``duration_key`` and ``event_key``. ``forward`` returns
``log_risk`` and ``mixture_weights`` (both ``[batch, K]``), survival at
``time_grid`` as ``survival``/``y_prob`` once baselines are fitted, and, with
labels, the EM ``loss``, the responsibilities ``posterior`` and ``y_true``.
:meth:`~pyhealth.models.DeepCoxMixtures.predict_survival` gives ``[N, T]``
survival at any times, and
:meth:`~pyhealth.models.DeepCoxMixtures.predict_latent` the ``[N, K]`` group
probabilities. Fitted baselines are part of ``state_dict()``. ``mode`` is
``None``; score predictions with censoring-aware survival metrics.

Example and ablation
--------------------

``examples/support2_survival_deep_cox_mixtures.py`` runs SUPPORT2 through
:class:`~pyhealth.datasets.Support2Dataset` and
:class:`~pyhealth.tasks.SurvivalTimeToEventSupport2DCM`, trains DCM with the
paper's settings, compares ``K = 1`` with mixtures of several sizes and
reports calibration and discrimination at the paper's horizons. Use
``--synthetic`` for a quick run without data. Setup, commands and measured
results are documented in the script.

API
---

.. autoclass:: pyhealth.models.DeepCoxMixtures
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.deep_cox_mixtures.cox_partial_log_likelihood

.. autofunction:: pyhealth.models.deep_cox_mixtures.breslow_cumulative_hazard
