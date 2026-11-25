pyhealth.tasks.tpc_day_one
==============================

Day-one continuous remaining-LoS adapter for TPC on existing eICU and MIMIC-IV
YAIB feature extraction. Each sample covers the first 24 hours of an ICU stay
and is labelled with the remaining length of stay in days.

Use ``BaseDataset.set_task`` with ``PatientSplit`` to fit normalization on
training patients only. ``TPCDayOne`` preserves missing values and the original
observation mask. Configure TPC with ``observed_key="observed"``,
``decay_key=None``, ``static_key="static"``, and the task's ``lab_indices``.
The example runner is ``examples/eicu_los_tpc.py``. ``examples/tpc/README.md``
documents the protocol, commands, limitations, and experiment results.

.. autoclass:: pyhealth.tasks.tpc_day_one.TPCDayOne
    :members:
    :show-inheritance:
    :no-inherited-members:
