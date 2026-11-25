pyhealth.tasks.MIMIC3Forecasting
================================

Irregular multivariate time-series forecasting on MIMIC-III in the
GRU-ODE-Bayes and APN benchmark setup: aggregate 96 laboratory, treatment,
output and prescription variables into half-hour bins, observe the first 36
hours, and forecast the next three observed time bins.

Paper: `Rethinking Irregular Time Series Forecasting: A Simple yet Effective
Baseline <https://arxiv.org/abs/2505.11250>`_.

.. autoclass:: pyhealth.tasks.MIMIC3Forecasting
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.mimic3_forecasting_statistics
