pyhealth.datasets.PTBXLDataset
==============================

Overview
--------

PTB-XL is a large publicly available 12-lead ECG dataset from PhysioNet
(version 1.0.3). It contains **21,799** clinical ECG records of 10 seconds
from **18,869** patients, with multi-label SCP-ECG statements, official
stratified folds (``strat_fold``), and demographic / site / device metadata
suited to shift-aware evaluation. Only v1.0.3 is supported (v1.0.1 figures
of 21,837 records / 18,885 patients come from Wagner et al., Scientific
Data 2020, and must not be mixed with the PhysioNet v1.0.3 counts).

For more information see `PhysioNet PTB-XL v1.0.3
<https://physionet.org/content/ptb-xl/1.0.3/>`_ and Wagner et al.,
`Scientific Data 2020 <https://www.nature.com/articles/s41597-020-0495-6>`_.

Optional dependency: install waveform I/O with ``pip install 'pyhealth[ptbxl]'``.

API Reference
-------------

.. autoclass:: pyhealth.datasets.PTBXLDataset
    :members:
    :undoc-members:
    :show-inheritance:
