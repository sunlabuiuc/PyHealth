pyhealth.tasks.RealTimeICUMortalityMIMIC4
=========================================

Real-time ICU mortality prediction on MIMIC-IV: one sample per measurement
time ("state marker"), with the irregular measurement tuples of the current
and the next state, a terminal flag, and mortality labels at 1/3/7/14/28
days. This is the data framing of Frost, Li & Harris,
`Robust Real-Time Mortality Prediction in the Intensive Care Unit using
Temporal Difference Learning <https://proceedings.mlr.press/v259/frost25a.html>`_
(ML4H 2024), designed for :class:`pyhealth.models.RealTimeTDCNNLSTM`.

.. autoclass:: pyhealth.tasks.realtime_icu_mortality_mimic4.RealTimeICUMortalityMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:
