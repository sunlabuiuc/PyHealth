pyhealth.tasks.TaskCard
=======================================

What a task predicts, for whom and when, with the reason for each choice, and
a dictionary of its features. Tasks provide one through the optional
:meth:`~pyhealth.tasks.BaseTask.task_card` hook; :mod:`pyhealth.autoimprove`
reports use it to explain models in clinical terms and to check that the
direction of each learned effect is plausible.

.. autoclass:: pyhealth.tasks.task_card.TaskCard
    :members:

.. autoclass:: pyhealth.tasks.task_card.FeatureInfo
    :members:

.. autodata:: pyhealth.tasks.task_card.DIRECTIONS
