pyhealth.models.PLMCA
===================================

Pretrained language model with class-wise cross-attention (PLM-CA) for
automated medical coding, reproduced from Edin et al., "An Unsupervised
Approach to Achieve Supervised-Level Explainability in Healthcare Records"
(EMNLP 2024, https://arxiv.org/abs/2406.08958).

A pretrained transformer encodes the clinical note in chunks, and a
cross-attention decoder gives each medical code its own query over all
tokens. The per-code attention weights are returned with every prediction,
so the model can be explained with attention-based methods such as
AttInGrad. Use it with a task that has one text input and one multilabel
output, for example :class:`~pyhealth.tasks.MIMIC3ICD9Coding`.

.. autoclass:: pyhealth.models.PLMCA
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ClassWiseCrossAttention
    :members:
    :undoc-members:
    :show-inheritance:
