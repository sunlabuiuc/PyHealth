pyhealth.models.CAML
===================================

Convolutional Attention for Multi-Label classification (CAML) for automated
medical coding, reproduced from Mullenbach et al., "Explainable Prediction of
Medical Codes from Clinical Text" (NAACL 2018,
https://arxiv.org/abs/1802.05695).

Word embeddings of the clinical note pass through one convolutional layer,
and every medical code attends over all positions with its own attention
vector. The per-code attention weights are returned with every prediction,
so the model can be explained with attention-based methods such as
AttInGrad, in the same way as :class:`~pyhealth.models.PLMCA`. Use it with a
task that has one text input and one multilabel output, for example
:class:`~pyhealth.tasks.MIMIC3ICD9Coding`.

.. autoclass:: pyhealth.models.CAML
    :members:
    :undoc-members:
    :show-inheritance:
