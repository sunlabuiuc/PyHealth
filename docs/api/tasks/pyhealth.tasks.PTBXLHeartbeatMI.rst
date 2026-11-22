pyhealth.tasks.PTBXLHeartbeatMI
===============================

Single-heartbeat myocardial infarction detection on PTB-XL (MI vs healthy), the
heartbeat setup TarDiff evaluates on PTB, with the segmentation of Medformer's PTB
preprocessing. ``examples/ecg/ptbxl/tardiff_ptbxl_tstr.py`` uses it to evaluate
:class:`~pyhealth.models.TarDiff`.

.. autoclass:: pyhealth.tasks.PTBXLHeartbeatMI
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.ptbxl_heartbeat.detect_r_peaks

.. autofunction:: pyhealth.tasks.ptbxl_heartbeat.segment_heartbeats
