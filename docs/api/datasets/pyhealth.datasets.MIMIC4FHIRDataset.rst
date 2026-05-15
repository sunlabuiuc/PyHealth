pyhealth.datasets.MIMIC4FHIRDataset
=====================================

`MIMIC-IV on FHIR <https://physionet.org/content/mimic-iv-fhir/>`_ NDJSON ingest
for CEHR-style token sequences used with
:class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask` and
:class:`~pyhealth.models.EHRMambaCEHR`.

YAML defaults live in ``pyhealth/datasets/configs/mimic4_fhir.yaml``. Unlike the
earlier nested-object approach, the YAML now declares a normal ``tables:``
schema for flattened FHIR resources (``patient``, ``encounter``, ``condition``,
``observation``, ``medication_request``, ``procedure``). The class subclasses
:class:`~pyhealth.datasets.BaseDataset` and builds a standard Polars
``global_event_df`` backed by cached Parquet (``global_event_df.parquet/part-*.parquet``),
same tabular path as other datasets: :meth:`~pyhealth.datasets.BaseDataset.set_task`,
:meth:`iter_patients`, :meth:`get_patient`, etc.

**Ingest (out-of-core).** Matching ``*.ndjson`` / ``*.ndjson.gz`` files are read
**line by line**; each resource is normalized into a flattened per-resource
Parquet table under ``cache/flattened_tables/``. Those tables are then fed
through the regular YAML-driven :class:`~pyhealth.datasets.BaseDataset` loader to
materialize ``global_event_df``. This keeps FHIR aligned with PyHealth's usual
table-first pipeline instead of reparsing nested JSON per patient downstream.

**``max_patients``.** When set, the loader selects the first *N* patient ids after
a **sorted** ``unique`` over the flattened patient table, filters every
normalized table to that cohort, and then builds ``global_event_df`` from the
filtered tables. Ingest still scans all matching NDJSON once unless you also
override ``glob_patterns`` / ``glob_pattern`` (defaults skip non-flattened PhysioNet shards).

**Downstream memory (still important).** Streaming ingest avoids loading the
entire NDJSON corpus into RAM at once, but other steps can still be heavy on
large cohorts: ``global_event_df`` materialization, MPF vocabulary warmup, and
:meth:`set_task` still walk patients and samples; training needs RAM/VRAM for the
model and batches. For a **full** PhysioNet tree, plan for **large disk**
(flattened tables plus event cache), **comfortable system RAM** for Polars/PyArrow
and task pipelines, and restrict ``glob_patterns`` / ``glob_pattern`` or ``max_patients`` when
prototyping on a laptop.

**Recommended hardware (informal)**

Order-of-magnitude guides, not guarantees. Ingest footprint is **much smaller**
than “load everything into Python”; wall time still grows with **decompressed
NDJSON volume** and the amount of flattened table data produced.

* **Smoke / CI**  
  Small on-disk fixtures (see tests and ``examples/mimic4fhir_mpf_ehrmamba.py``):
  a recent laptop is sufficient.

* **Laptop-scale real FHIR subset**  
  A **narrow** ``glob_patterns`` / ``glob_pattern`` and/or ``max_patients`` in the hundreds keeps
  cache and task passes manageable. **≥ 16 GB** system RAM is a practical
  comfort target for Polars + trainer + OS; validate GPU **VRAM** for your
  ``max_len`` and batch size.

* **Full default globs on a complete export**  
  Favor **workstations or servers** with **fast SSD**, **large disk**, and
  **ample RAM** for downstream steps—not because NDJSON is fully buffered in
  memory during ingest, but because total work and caches still scale with the
  full dataset.

.. autoclass:: pyhealth.datasets.MIMIC4FHIRDataset
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.datasets.ConceptVocab
    :members:
    :undoc-members:
    :show-inheritance:
