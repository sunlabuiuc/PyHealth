pyhealth.datasets.LUDBDataset
===================================

Dataset is available at https://physionet.org/content/ludb/1.0.1/

The Lobachevsky University Electrocardiography Database (LUDB) has 200
ten-second, 12-lead ECGs at 500 Hz, with P, QRS and T onsets and offsets
marked on every lead (Kalyakulina et al., IEEE Access 2020,
doi: 10.1109/ACCESS.2020.3029211). The dataset writes one metadata row per
record; signals and wave boundaries are loaded with
:func:`pyhealth.datasets.ludb.load_ludb_record`.

.. autoclass:: pyhealth.datasets.LUDBDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.datasets.ludb.load_ludb_record

.. autofunction:: pyhealth.datasets.ludb.parse_wave_annotations
