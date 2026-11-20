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
