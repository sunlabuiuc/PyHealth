pyhealth.models.SentenceKDTransformer
=====================================

Sentence-level knowledge-distillation transformer from
Kim et al., "Integrating ChatGPT into Secure Hospital Networks:
A Case Study on Improving Radiology Report Analysis", CHIL 2024
(`paper <https://proceedings.mlr.press/v248/kim24a.html>`_).

A pretrained BERT-family encoder (default ``StanfordAIMI/RadBERT``) maps
each input text to a ``[CLS]`` hidden state, a linear head produces class
logits, and the training loss combines cross-entropy with the supervised
contrastive loss of Khosla et al. (2020) weighted by ``lam`` — this is
paper Eq. 5. The ``[CLS]`` hidden state is used directly as the
contrastive feature (paper Sec. 4.5).

A :meth:`~pyhealth.models.SentenceKDTransformer.document_predict` helper
aggregates per-sentence predictions into a document-level probability of
being abnormal, following paper Eq. 4 (``max``) and two novel ablation
modes (``topk_mean``, ``attn``).

.. autoclass:: pyhealth.models.SentenceKDTransformer
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.supervised_contrastive_loss
