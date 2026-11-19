pyhealth.models.HFEHRFoundationModel
====================================

The pretrained EHR foundation models of Wornow et al., "Context Clues:
Evaluating Long Context Models for Clinical Prediction Tasks on EHRs" (2024),
released as gated ``StanfordShahLab/*-clmbr`` Hugging Face checkpoints (gpt,
llama, mamba and hyena), with a linear head. The model tokenizes raw event
timelines from :class:`~pyhealth.tasks.EHRFoundationModelTask` with a
reimplementation of hf_ehr's CLMBR tokenizer in ``forward()``, so tasks stay
model-agnostic. ``mode="frozen"`` is a linear probe on cached embeddings;
``mode="finetune"`` trains the backbone too.

Access: request access on each model's Hugging Face page (CC-BY-NC-4.0, a
current CITI certificate and the EHRSHOT Credentialed Health Data License),
then ``huggingface-cli login``. The EHRSHOT data is on Redivis under its
data-use agreement (:class:`~pyhealth.datasets.EHRShotDataset`).

.. autoclass:: pyhealth.models.HFEHRFoundationModel
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.HFEHRTokenizer
    :members:

.. autoclass:: pyhealth.models.EHRFoundationModelBase
    :members: coverage_report, tokenize, encode

.. autofunction:: pyhealth.models.ehr_fm.as_value
