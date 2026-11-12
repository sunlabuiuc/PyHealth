pyhealth.models.APN
===================

APN: a simple, effective baseline for irregular multivariate time-series
forecasting. Time-Aware Patch Aggregation pools each variable's irregular
observations into adaptive patches with learnable boundaries, a query module
summarizes the patches, and a shallow MLP forecasts the query times
(`Liu et al., AAAI 2026 <https://arxiv.org/abs/2505.11250>`_).

See ``examples/apn_physionet2012_forecasting.ipynb`` for a reproduction of
the paper's PhysioNet 2012 result.

.. autoclass:: pyhealth.models.APN
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TimeAwarePatchAggregation
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.LearnableTimeEncoding
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.masked_mse
