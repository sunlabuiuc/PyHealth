pyhealth.models.DOSSIERPipeline
================================

DOSSIER: Fact Checking in Electronic Health Records while Preserving Patient
Privacy (Zhang et al., MLHC 2024).

DOSSIER is a zero-shot LLM-based pipeline for verifying natural language
claims about a patient's ICU stay.  It translates claims into SQL queries
executed against structured MIMIC-III evidence tables, optionally augmented
with a global biomedical knowledge graph (SemMedDB).

Unlike standard trainable models, ``DOSSIERPipeline`` is not a subclass of
``BaseModel`` — it has no learnable weights and provides a
``predict`` / ``evaluate`` interface instead of a ``forward`` pass.

.. autoclass:: pyhealth.models.DOSSIERPipeline
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.DOSSIERPromptGenerator
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.SQLExecutor
    :members:
    :undoc-members:
    :show-inheritance:
