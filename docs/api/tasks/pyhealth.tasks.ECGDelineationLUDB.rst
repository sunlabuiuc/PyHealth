pyhealth.tasks.ECGDelineationLUDB
=================================

P / QRS / T delineation on LUDB, following Park et al., CHIL 2025
(https://proceedings.mlr.press/v287/park25a.html).

Each lead is one sample: ``signal`` is the filtered, z-scored lead,
shape ``(1, 5000)``, and ``mask`` marks the P wave, QRS complex and T wave
at each time step, shape ``(3, 5000)``.

Task parameters:

- filter_mode: ``"bandpass"`` (paper), ``"highpass"``, ``"lowpass"`` or
  ``"none"``
- low_cut_hz: float, default=0.05
- high_cut_hz: float, default=150
- filter_order: int, default=2
- normalize: bool, default=True
- leads: list of LUDB lead names, or None for all 12

.. autoclass:: pyhealth.tasks.ECGDelineationLUDB
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ecg_delineation_ludb.filter_ecg

.. autofunction:: pyhealth.tasks.ecg_delineation_ludb.segments_to_mask
