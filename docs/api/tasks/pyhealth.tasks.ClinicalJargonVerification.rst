pyhealth.tasks.ClinicalJargonVerification
=========================================

Binary candidate-verification task for the public MedLingo and CASI clinical
jargon benchmarks.

Example
-------

.. code-block:: python

    from pyhealth.datasets import ClinicalJargonDataset
    from pyhealth.tasks import ClinicalJargonVerification

    dataset = ClinicalJargonDataset(root="/tmp/clinical_jargon", download=True)
    task = ClinicalJargonVerification(
        benchmark="medlingo",
        medlingo_distractors=1,
    )
    samples = dataset.set_task(task)

.. autoclass:: pyhealth.tasks.ClinicalJargonVerification
    :members:
    :undoc-members:
    :show-inheritance:
