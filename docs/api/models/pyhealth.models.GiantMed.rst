pyhealth.models.GiantMed
========================

GiantMed refines the boundary medication probabilities of a trained model.
See the MIMIC-III and MIMIC-IV examples for the training and inference workflow.

.. autoclass:: pyhealth.models.GiantMed
    :members:

.. autoclass:: pyhealth.models.MediEvidenceFilter
    :members:

.. autoclass:: pyhealth.models.HuggingFaceGiantMedGenerator
    :members:

.. autofunction:: pyhealth.experiment_tracking.record_experiment

MEDI evidence supports ICD-9-CM and ICD-10-CM indications. MIMIC-IV diagnosis
codes use ``9_`` and ``10_`` prefixes; unprefixed codes use ICD-9-CM for
compatibility with MIMIC-III. Matching stays within the original code system.
See the MIMIC-IV example for a local, deterministic pipeline smoke test.

LLM backend interfaces
----------------------

GiantMed accepts a ``generate(prompt) -> str`` callable. Its optional
Hugging Face adapter implements this interface; the OpenRouter adapter lives
in the example script. AutoImprove backends accept ``llm(system, prompt) -> str``
and can be used through a wrapper. The wrapper's messages, generation settings
and token-usage behavior can differ from the OpenRouter example. See the
GiantMed example README for an adapter and data-handling considerations.
