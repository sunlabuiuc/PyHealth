pyhealth.tasks.vitals_generation_eicu
===========================================

A task that turns eICU into static vital-sign vectors for generating new
patients: each patient's first ICU stay becomes ``[age, heart rate,
respiration rate, diastolic BP, mean BP]``, with each vital averaged over the
stay's first 24 hours. The values are continuous, so the task pairs with
:class:`~pyhealth.models.GAN`'s continuous mode; see
``examples/gan_tutorial.ipynb``.

The task reads the ``vitalperiodic`` and ``vitalaperiodic`` tables:

.. code-block:: python

    from pyhealth.datasets import eICUDataset
    from pyhealth.tasks import CARTVitalsGenerationEICU

    dataset = eICUDataset(
        root="https://physionet.org/files/eicu-crd-demo/2.0.1",
        tables=["vitalperiodic", "vitalaperiodic"],
    )
    samples = dataset.set_task(CARTVitalsGenerationEICU())

.. autoclass:: pyhealth.tasks.vitals_generation_eicu.CARTVitalsGenerationEICU
    :members:
    :undoc-members:
    :show-inheritance:
