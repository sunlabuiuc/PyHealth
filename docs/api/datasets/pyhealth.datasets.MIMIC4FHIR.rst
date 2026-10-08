pyhealth.datasets.MIMIC4FHIR
============================

A pre-bundled :class:`~pyhealth.datasets.FHIRDataset` for the PhysioNet
`MIMIC-IV on FHIR <https://physionet.org/content/mimic-iv-fhir/>`_ export
(R4, demo 2.1.0 and full release). All ingest logic — file globs, per-resource
projection, downstream event schema — is described by the bundled YAML at
``pyhealth/datasets/fhir/configs/mimic4fhir.yaml``; this class only points at
that path.

For everything outside the MIMIC-specific defaults (transform registry,
``Col`` / ``ResourceSpec`` syntax, the three-tier cache story), see the parent
page: :doc:`pyhealth.datasets.FHIRDataset`.

Quick start
-----------

.. code-block:: python

    from pyhealth.datasets import MIMIC4FHIR
    from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask

    def main():
        ds = MIMIC4FHIR(root="/data/mimic-iv-fhir")
        sample_ds = ds.set_task(MPFClinicalPredictionTask(), num_workers=1)
        # ... split / dataloader / model / trainer ...

    if __name__ == "__main__":
        main()

For the full end-to-end demo (training EHR-Mamba on MPF samples) see
``examples/mimic4fhir_mpf_ehrmamba.py``.

Resource coverage
-----------------

The bundled config flattens six FHIR resource types out of the PhysioNet
export:

==========================  ============================  ===============================
FHIR resourceType           Output table                  Key columns
==========================  ============================  ===============================
``Patient``                 ``patient.parquet``           ``patient_id``, ``birth_date``, ``gender``, ``deceased_*``
``Encounter``               ``encounter.parquet``         ``patient_id``, ``encounter_id``, ``event_time``, ``encounter_class``
``Condition``               ``condition.parquet``         ``patient_id``, ``encounter_id``, ``event_time``, ``concept_key``
``Observation``             ``observation.parquet``       ``patient_id``, ``encounter_id``, ``event_time``, ``concept_key``
``MedicationRequest``       ``medication_request.parquet`` ``patient_id``, ``encounter_id``, ``event_time``, ``concept_key``
``Procedure``               ``procedure.parquet``         ``patient_id``, ``encounter_id``, ``event_time``, ``concept_key``
==========================  ============================  ===============================

PhysioNet shards that contain only other resource types
(``MedicationAdministration``, ``Specimen``, ``Organization``, …) are skipped
at the file level by the bundled ``glob_patterns``. To include them, override
``glob_patterns=`` at the constructor and add a ``resource_specs:`` entry plus
matching ``tables:`` entry in a copy of the YAML.

Customising
-----------

The bundled config is the easiest starting point for authoring a similar ingest
for other FHIR exports. Copy
``pyhealth/datasets/fhir/configs/mimic4fhir.yaml``, edit the
``resource_specs:`` and ``tables:`` blocks for the resources you care about,
and either:

* pass ``config_path=...`` directly to ``FHIRDataset(root=..., config_path=...)``, or
* subclass ``FHIRDataset`` and set ``DEFAULT_CONFIG_PATH`` on the subclass.

See the "Customising for a non-MIMIC FHIR export" section of
:doc:`pyhealth.datasets.FHIRDataset` for the step-by-step.

API reference
-------------

.. autoclass:: pyhealth.datasets.MIMIC4FHIR
    :members:
    :undoc-members:
    :show-inheritance:
