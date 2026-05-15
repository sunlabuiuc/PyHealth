pyhealth.models.EHRMambaCEHR
===================================

EHRMambaCEHR applies CEHR-style embeddings (:class:`~pyhealth.models.cehr_embeddings.MambaEmbeddingsForCEHR`)
and a stack of :class:`~pyhealth.models.MambaBlock` layers to a single FHIR token stream, for use with
:class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask` and
:class:`~pyhealth.datasets.mimic4_fhir.MIMIC4FHIRDataset`.

.. autoclass:: pyhealth.models.EHRMambaCEHR
    :members:
    :undoc-members:
    :show-inheritance:
