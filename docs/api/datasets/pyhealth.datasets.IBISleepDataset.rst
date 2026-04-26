pyhealth.datasets.IBISleepDataset
=================================

Dataset for IBI-based sleep staging from DREAMT, SHHS, and MESA recordings.
Each subject's overnight recording is stored as a pre-processed NPZ file containing
a 25 Hz inter-beat-interval (IBI) time series and per-sample sleep stage labels.

See ``examples/preprocess_dreamt_to_ibi.py``, ``examples/preprocess_shhs_to_ibi.py``, and
``examples/preprocess_mesa_to_ibi.py`` for scripts that convert raw EDF recordings to the
NPZ format expected by this dataset.

.. autoclass:: pyhealth.datasets.IBISleepDataset
    :members:
    :undoc-members:
    :show-inheritance:
