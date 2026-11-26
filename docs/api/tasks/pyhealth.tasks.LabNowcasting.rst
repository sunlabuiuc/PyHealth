pyhealth.tasks.LabNowcasting
============================

Intra-visit laboratory event nowcasting, the task of TRACE (Liang et al.,
WWW 2025 Companion): from the events so far in a hospital admission
(laboratory results with their flag, medications), predict which laboratory
events, test and flag, the next blood draw will contain. Laboratory results
are flagged high/low/normal from the reference range on MIMIC-IV and
abnormal/normal on MIMIC-III. Pairs with :class:`~pyhealth.models.TRACE`;
score it with ``multilabel_metrics_fn`` (``pr_auc_micro``, ``f1_micro``,
``precision@5``, ``ndcg@5``).

.. autoclass:: pyhealth.tasks.LabNowcastingMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.LabNowcastingMIMIC3
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.LabNowcasting
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.tasks.lab_flag
