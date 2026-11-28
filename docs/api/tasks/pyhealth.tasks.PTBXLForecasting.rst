pyhealth.tasks.PTBXLForecasting
===============================

Univariate ECG forecasting on PTB-XL, in the setup of the MIRA evaluation
(`Li et al., NeurIPS 2025 <https://arxiv.org/abs/2506.07584>`_;
``model_eval.py`` in the `authors' code <https://github.com/microsoft/MIRA>`_).
Each lead is z-scored over its 10-second record, and the model forecasts the
``horizon`` samples that follow a ``context_length`` window. The samples have
the fields of :class:`~pyhealth.tasks.PhysioNet2012Forecasting`, so
:class:`~pyhealth.models.MIRA` and :class:`~pyhealth.models.APN` run on them
unchanged. See ``examples/mira_signal_forecasting.py``.

.. autoclass:: pyhealth.tasks.PTBXLForecasting
    :members:
    :undoc-members:
    :show-inheritance:

.. autodata:: pyhealth.tasks.PTBXL_LEADS
