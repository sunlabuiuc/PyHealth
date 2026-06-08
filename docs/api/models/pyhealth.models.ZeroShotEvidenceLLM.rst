pyhealth.models.ZeroShotEvidenceLLM
=====================================

Zero-shot LLM pipeline for retrieving and summarising clinically relevant
evidence from unstructured EHR notes.  Implements the two-step prompting
strategy from `Ahsan et al. (2024)
<https://arxiv.org/abs/2309.04550>`_ (CHIL 2024, PMLR 248:489-505).

The model requires no task-specific training — it uses instruction-tuned
models (e.g. Flan-T5 XXL or Mistral-7B-Instruct) in a zero-shot setting.
A Clinical-BERT dense-retrieval baseline is also available via
``use_cbert_baseline=True``.

.. autoclass:: pyhealth.models.ZeroShotEvidenceLLM
    :members:
    :undoc-members:
    :show-inheritance:
