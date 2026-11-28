pyhealth.tasks.SleepEDFForecasting
==================================

Univariate EEG forecasting on Sleep-EDF, in the setup of the MIRA evaluation
(`Li et al., NeurIPS 2025 <https://arxiv.org/abs/2506.07584>`_). Windows are
taken from the sleep period of each night, as marked by the hypnogram, and
each channel is z-scored over that period. The samples have the fields of
:class:`~pyhealth.tasks.PhysioNet2012Forecasting`, so
:class:`~pyhealth.models.MIRA` and :class:`~pyhealth.models.APN` run on them
unchanged. See ``examples/mira_signal_forecasting.py``.

.. autoclass:: pyhealth.tasks.SleepEDFForecasting
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: pyhealth.tasks.SLEEPEDF_CHANNELS
