pyhealth.tasks.tpc_day_one
==============================

Day-one continuous remaining-LoS adapter for TPC on existing eICU and MIMIC-IV
YAIB feature extraction. This helper supports the model contribution; it is
not a claim of full-pipeline clinical reproduction.

**Clinical dataset access, experiments, and results are pending.**
The adapter and runner are covered only by synthetic unit tests so far.

Use ``BaseDataset.set_task`` with ``PatientSplit`` to fit normalization on
training patients only. ``TPCDayOne`` preserves missing values and the original
observation mask. Configure TPC with ``observed_key="observed"``,
``decay_key=None``, ``static_key="static"``, and the task's ``lab_indices``.
The example runner is ``examples/eicu_los_tpc.py``; its default mode only
writes a plan and PENDING tables. ``examples/tpc/README.md`` documents the
protocol, commands, limitations, and clinical result placeholders.

.. autoclass:: pyhealth.tasks.tpc_day_one.TPCDayOne
    :members:
    :show-inheritance:
    :no-inherited-members:
