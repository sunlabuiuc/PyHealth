pyhealth.datasets.GoldTestSplit
===============================

:class:`GoldTestSplit` follows the split of DiPro: subjects with manually
validated labels all go to the test part, which random subjects fill up to 20%
of all subjects, and an eighth of the rest goes to validation. Pass it to
``set_task`` like any other :class:`~pyhealth.datasets.Split`.

.. code-block:: python

    from pyhealth.datasets import GoldTestSplit
    from pyhealth.tasks import DiseaseProgressionMIMIC4

    task = DiseaseProgressionMIMIC4()
    split = GoldTestSplit(task.gold_subjects(dataset), seed=0)
    train, val, test = dataset.set_task(task, split=split)

.. autoclass:: pyhealth.datasets.gold_test_split.GoldTestSplit
    :members:
    :undoc-members:
    :show-inheritance:
