pyhealth.tasks.tpc_day_one
==============================

Day-one continuous remaining-LoS adapter for TPC on existing eICU and MIMIC-IV
YAIB feature extraction. This helper supports the model contribution; it is
not a claim of full-pipeline clinical reproduction.

The adapter and runner have completed a three-seed, 45-configuration sweep on
the **MIMIC-IV Clinical Database Demo 2.2**, documented in
``examples/tpc/README.md``. This small-demo experiment is not paper benchmark
reproduction. Full-dataset eICU/MIMIC-IV experiments and results remain pending.
Unit tests continue to use only synthetic fixtures, not demo records.

Use ``BaseDataset.set_task`` with ``PatientSplit`` to fit normalization on
training patients only. ``TPCDayOne`` preserves missing values and the original
observation mask. Configure TPC with ``observed_key="observed"``,
``decay_key=None``, ``static_key="static"``, and the task's ``lab_indices``.
The example runner is ``examples/eicu_los_tpc.py``; its default mode only
writes a plan and PENDING tables. ``examples/tpc/README.md`` documents the
protocol, commands, limitations, measured demo results, and pending full-data work.

.. autoclass:: pyhealth.tasks.tpc_day_one.TPCDayOne
    :members:
    :show-inheritance:
    :no-inherited-members:
