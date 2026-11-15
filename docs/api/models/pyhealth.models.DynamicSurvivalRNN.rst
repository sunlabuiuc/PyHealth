pyhealth.models.DynamicSurvivalRNN
==================================

Dynamic survival analysis for early event prediction (Yeche et al., CHIL 2024):
an :class:`~pyhealth.models.RNN_V2` encoder with a hazard head, a survival loss
truncated to the prediction horizon, and hazard-bias initialization.

.. autoclass:: pyhealth.models.DynamicSurvivalRNN
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.dynamic_survival_rnn.survival_loss

.. autofunction:: pyhealth.models.dynamic_survival_rnn.hazards_to_risk

.. automodule:: pyhealth.models.dynamic_survival
    :members:
