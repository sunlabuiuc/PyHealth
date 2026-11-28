pyhealth.models.RealTimeTDCNNLSTM
=================================

CNN-LSTM value network over irregular measurement tuples, trained with
temporal-difference (TD) learning against a soft-updated target network, or
with supervised targets as a baseline. Reproduces Frost, Li & Harris,
`Robust Real-Time Mortality Prediction in the Intensive Care Unit using
Temporal Difference Learning <https://proceedings.mlr.press/v259/frost25a.html>`_
(ML4H 2024). Use it with :class:`pyhealth.tasks.RealTimeICUMortalityMIMIC4`
(or the eICU and PhysioNet 2012 variants) and the stock
:class:`pyhealth.trainer.Trainer`; see
``examples/mimic4_realtime_icu_mortality_td_cnn_lstm.py``.

:class:`pyhealth.models.TDCNNLSTM` (:doc:`pyhealth.models.td_cnn_lstm`) is an
independent implementation of the same paper's network, which needs a custom
training loop. This model trains with the stock ``Trainer`` (the target-network
update runs inside ``forward``) and also covers the paper's class-balanced
supervised baselines and λ-mixed targets.

.. autoclass:: pyhealth.models.RealTimeTDCNNLSTMEncoder
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.RealTimeTDCNNLSTM
    :members:
    :undoc-members:
    :show-inheritance:
