pyhealth.datasets.MIMIC4FHIRDataset
=====================================

`MIMIC-IV on FHIR <https://physionet.org/content/mimic-iv-fhir/>`_ NDJSON ingest
for CEHR-style token sequences used with
:class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask` and
:class:`~pyhealth.models.EHRMambaCEHR`.

YAML defaults live in ``pyhealth/datasets/configs/mimic4_fhir.yaml`` (e.g.
``glob_pattern``). The loader subclasses :class:`~pyhealth.datasets.BaseDataset`
but does not build a Polars ``global_event_df``; use :meth:`MIMIC4FHIRDataset.gather_samples`
or :meth:`MIMIC4FHIRDataset.set_task`.

.. autoclass:: pyhealth.datasets.MIMIC4FHIRDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.ConceptVocab
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.FHIRPatient
    :members:
    :undoc-members:
    :show-inheritance:
