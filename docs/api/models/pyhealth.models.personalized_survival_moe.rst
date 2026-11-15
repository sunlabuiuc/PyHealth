pyhealth.models.PersonalizedSurvivalMoE
=======================================

Overview
--------

Personalized mixture-of-experts model for discrete-time survival prediction,
following `Morrill et al. <https://proceedings.mlr.press/v297/morrill26a.html>`_.
Patient-dependent routing combines expert event-time distributions. Training
uses a right-censored negative log-likelihood and an expert-balancing penalty.

Inputs and outputs
------------------

Provide dense, preprocessed continuous features and, when embeddings are enabled,
integer categorical indices. Configure dataset field names with ``feature_key``,
``categories_key``, ``time_bin_key`` and ``event_key``; their defaults are
``features``, ``categories``, ``time_bin`` and ``event``. Dataset schemas must
contain the configured input and output keys.

Time labels are zero-based bin indices; the final bin represents later/no event.
Event flags are one for an observed event and zero for censoring. Fit preprocessing
and time-bin boundaries on training data only. Omit both labels for inference.

The model returns ``pmf`` with shape ``[batch, num_time_bins]``,
``survival``/``y_prob`` with shape ``[batch, num_time_bins - 1]``, and expert
routing weights in ``mixture_weights``. With labels, ``nll`` is the censored
negative log-likelihood and ``loss`` adds the balancing penalty.
``mode=None`` is intentional: use a custom survival training loop and
censoring-aware metrics rather than classification scoring.

Example and ablation
--------------------

From a PyHealth checkout with this contribution installed::

    pip install SurvivalEVAL
    python examples/support2_survival_personalized_moe.py --synthetic --epochs 2

This uses synthetic data to check execution. For SUPPORT2, replace
``--synthetic --epochs 2`` with ``--root data/support2``; missing data are
downloaded automatically. Preprocessing is fitted on training data only.

The example compares dropout rates 0, 0.1, 0.3 and 0.5 using five paired seeds
and a fixed data split. Results are written to ``personalized_moe_results.json``
(override with ``--output``). Setup, reproduction results and prediction-setting
limitations are documented in ``examples/support2_survival_personalized_moe.py``.

API
---

.. autoclass:: pyhealth.models.PersonalizedSurvivalMoE
    :members:
    :undoc-members:
    :show-inheritance:
