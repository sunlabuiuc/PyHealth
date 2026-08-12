pyhealth.tasks.ptbxl
====================

Overview
--------

Task helpers for PTB-XL multi-label ECG diagnosis. This module currently
implements the official **5-diagnostic-superclass** classification task
(``NORM``, ``MI``, ``STTC``, ``CD``, ``HYP``) by aggregating diagnostic SCP
statements via ``scp_statements.csv``.

Empty superclass label sets (407 / 21,799 records after aggregation; mainly
pacemaker ECGs) are dropped by default. Official fold splitting is provided
by :func:`pyhealth.datasets.split_by_strat_fold`.

API Reference
-------------

.. automodule:: pyhealth.tasks.ptbxl
    :members:
    :undoc-members:
    :show-inheritance:
