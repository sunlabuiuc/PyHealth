pyhealth.models.td_cnn_lstm
===================================

Temporal-difference CNN-LSTM for real-time ICU mortality prediction, from
Frost et al., `Robust Real-Time Mortality Prediction in the Intensive Care
Unit using Temporal Difference Learning <https://arxiv.org/abs/2411.04285>`_.

Each sample is one ICU measurement-time state: up to ``L`` irregular events,
each with five aligned channels (``timepoints``, ``values``, ``features``,
``delta_time`` and ``delta_value``). In TD mode, a frozen target copy of the
predictor scores the successor state (the matching ``next_*`` channels), and
its prediction replaces the observed label as the training target. States
without a successor fall back to the observed 28-day mortality label.

The standard PyHealth ``Trainer`` does not update the target network, so TD
training uses a short custom loop: ``loss.backward()``, ``optimizer.step()``,
then ``model.update_target()``. Set ``objective="supervised"`` to train
against observed labels only, and ``use_delta_embeddings=False`` for the
no-delta ablation. A complete synthetic example, including the ablation, is
in ``examples/mimic4_mortality_td_cnn_lstm.py``.

.. autoclass:: pyhealth.models.TDCNNLSTM
    :members:
    :undoc-members:
    :show-inheritance:
