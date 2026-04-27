pyhealth.datasets.ClinicalJargonDataset
=======================================

Public clinical jargon benchmark dataset backed by the released MedLingo and
CASI assets from Jia et al. (CHIL 2025).

Example
-------

.. code-block:: python

    from pyhealth.datasets import ClinicalJargonDataset

    dataset = ClinicalJargonDataset(root="/tmp/clinical_jargon", download=True)
    task = dataset.default_task
    samples = dataset.set_task(task)

.. autoclass:: pyhealth.datasets.ClinicalJargonDataset
    :members:
    :undoc-members:
    :show-inheritance:
