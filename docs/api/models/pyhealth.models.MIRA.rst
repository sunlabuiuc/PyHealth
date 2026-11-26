pyhealth.models.MIRA
====================

MIRA: a medical time-series foundation model for irregularly sampled data.
A decoder-only transformer with continuous-time rotary positions (CT-RoPE), a
frequency-specialized mixture of experts and a neural-ODE extrapolation block
forecasts the next value of a univariate series at arbitrary times
(`Li et al., NeurIPS 2025 <https://arxiv.org/abs/2506.07584>`_,
`authors' code <https://github.com/microsoft/MIRA>`_).

``MIRA(dataset, pretrained=True)`` loads the published MIRA-Large checkpoint
(`MIRA-Mode/MIRA <https://huggingface.co/MIRA-Mode/MIRA>`_, MIT License) with
every weight matched, for zero-shot forecasting or fine-tuning. The model runs
channel-independently on the inputs of
:class:`~pyhealth.tasks.PhysioNet2012Forecasting`, like
:class:`~pyhealth.models.APN`.

See ``examples/mira_physionet2012_forecasting.ipynb`` for zero-shot MIRA on
PhysioNet 2012 next to APN and last observation carried forward.

.. autoclass:: pyhealth.models.MIRA
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MIRABackbone
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MIRAConfig
    :members:
    :undoc-members:

.. autofunction:: pyhealth.models.load_mira_checkpoint

.. autofunction:: pyhealth.models.mira.normalize_time_for_ctrope
