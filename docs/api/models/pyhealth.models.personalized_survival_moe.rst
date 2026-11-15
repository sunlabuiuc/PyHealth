pyhealth.models.PersonalizedSurvivalMoE
=======================================

Personalized mixture-of-experts model for discrete-time survival prediction.

Inputs use configurable ``feature_key`` (default ``features``) and
``categories_key`` (default ``categories``, required only for embeddings).
Labels use ``time_bin_key`` (default ``time_bin``) and ``event_key`` (default
``event``). Dataset input/output schemas must contain the configured keys;
construction raises ``ValueError`` otherwise. Omit both labels for inference.

The model returns event-time probabilities, survival curves and routing weights,
plus loss when labeled. See ``examples/support2_survival_personalized_moe.py``
for preprocessing, training and evaluation.

.. autoclass:: pyhealth.models.PersonalizedSurvivalMoE
    :members:
    :undoc-members:
    :show-inheritance:
