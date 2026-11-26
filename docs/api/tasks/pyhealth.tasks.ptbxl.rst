pyhealth.tasks.ptbxl
====================

Overview
--------

Task helpers for PTB-XL multi-label ECG diagnosis. This module currently
implements the official **5-diagnostic-superclass** classification task
(``NORM``, ``MI``, ``STTC``, ``CD``, ``HYP``) by aggregating diagnostic SCP
statements via ``scp_statements.csv``.

Empty superclass label sets (≈400 records on v1.0.1, Wagner et al.
Table 9; mainly pacemaker ECGs) are dropped by default. On the full
v1.0.3 corpus, ``MultiLabelProcessor`` emits labels in alphabetical
order: ``CD, HYP, MI, NORM, STTC``. Official fold splitting is provided
by :func:`pyhealth.datasets.split_by_strat_fold`.

API Reference
-------------

.. automodule:: pyhealth.tasks.ptbxl
    :members:
    :undoc-members:
    :show-inheritance:
