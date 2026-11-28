pyhealth.tasks.decompensation_mimic3
=======================================

MIMIC-III hourly 24-hour decompensation prediction task.

Dataset Configuration
---------------------

The default MIMIC-III dataset configuration remains unchanged
for backward compatibility with existing PyHealth tasks.

For ward-level cohort filtering, this contribution provides a
dedicated configuration, ``mimic3_decompensation.yaml``, which
includes the ``first_wardid`` and ``last_wardid`` attributes.

Example usage from the repository root:

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import DecompensationMIMIC3

    dataset = MIMIC3Dataset(
        root="/path/to/mimic-iii/1.4",
        tables=["chartevents", "labevents"],
        dataset_name="mimic3_decompensation",
        config_path=(
            "pyhealth/datasets/configs/"
            "mimic3_decompensation.yaml"
        ),
    )

    sample_dataset = dataset.set_task(
        DecompensationMIMIC3()
    )

The dedicated configuration requires source data containing
both ward ID columns.

Cohort Filtering
----------------

The task excludes ICU stays with different first and last
ward IDs or different first and last care units.

If both ward IDs are unavailable, the task can still process
the stay subject to the remaining cohort criteria. However,
ward ID consistency cannot be verified in this case.

If only one ward ID is available, the stay is excluded.

The dedicated configuration preserves the ward ID comparison
used in the reproduction while avoiding changes to the
default MIMIC-III dataset configuration.

.. autoclass:: pyhealth.tasks.decompensation_mimic3.DecompensationMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:
