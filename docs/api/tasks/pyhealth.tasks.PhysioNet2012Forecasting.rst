pyhealth.tasks.PhysioNet2012Forecasting
=======================================

Irregular multivariate time-series forecasting on PhysioNet 2012, in the
setup of the forecasting benchmarks (GraFITi, Warpformer, APN): from the
first 36 hours of an ICU stay, forecast the values of 36 variables observed
at the next three observation times. Values are hourly means, z-scored, with
outliers beyond 5 standard deviations removed.

.. autoclass:: pyhealth.tasks.PhysioNet2012Forecasting
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.physionet2012_statistics

.. autofunction:: pyhealth.tasks.physionet2012_benchmark_split
