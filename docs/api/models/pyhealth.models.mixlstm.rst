pyhealth.models.MixLSTM
=======================

The MixLSTM model from Oh et al. 2020, "Relaxed Parameter Sharing:
Effectively Modeling Time-Varying Relationships in Clinical Time-Series"
(https://arxiv.org/abs/1906.02898).

MixLSTM addresses the problem of *temporal conditional shift* in clinical
time-series, i.e., settings in which the relationship between input features
and outcomes changes over the course of a patient's hospital stay. Instead
of sharing a single set of LSTM parameters across all time steps, MixLSTM
maintains ``K`` independent LSTM cells and, at every time step, computes a
learned convex combination of their parameters using mixing coefficients.
This enables smooth transitions between different temporal dynamics without
requiring hard segment boundaries.

.. autoclass:: pyhealth.models.MixLSTM
   :members:
   :undoc-members:
   :show-inheritance:
