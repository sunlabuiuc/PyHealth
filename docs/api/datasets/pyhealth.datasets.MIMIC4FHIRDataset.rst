pyhealth.datasets.MIMIC4FHIRDataset
=====================================

`MIMIC-IV on FHIR <https://physionet.org/content/mimic-iv-fhir/>`_ NDJSON ingest
for CEHR-style token sequences used with
:class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask` and
:class:`~pyhealth.models.EHRMambaCEHR`.

YAML defaults live in ``pyhealth/datasets/configs/mimic4_fhir.yaml`` (e.g.
``glob_pattern`` for ``**/*.ndjson.gz`` on PhysioNet exports, or ``**/*.ndjson``
if uncompressed). The class subclasses :class:`~pyhealth.datasets.BaseDataset`
and builds a standard Polars ``global_event_df`` backed by cached Parquet
(``global_event_df.parquet/part-*.parquet``), same tabular path as other
datasets: :meth:`~pyhealth.datasets.BaseDataset.set_task`, :meth:`iter_patients`,
:meth:`get_patient`, etc.

**Ingest (out-of-core).** Matching ``*.ndjson`` / ``*.ndjson.gz`` files are read
**line by line**; each FHIR resource row is written to hash-partitioned Parquet
shards (``patient_id`` → stable shard via CRC32). That bounds **peak ingest RAM**
to on-disk batch buffers and shard writers (see constructor / YAML
``ingest_num_shards``), instead of materializing the full export in Python lists.
Shards are finalized into ``part-*.parquet`` under the dataset cache; there is no
full-table ``(patient_id, timestamp)`` sort on disk—per-event time order for
:class:`~pyhealth.data.Patient` comes from ``data_source.sort("timestamp")`` when
a patient slice is loaded.

**``max_patients``.** When set, the loader selects the first *N* patient ids after
a **sorted** ``unique`` over the cached table (then filters shards). That caps
stored patients and speeds downstream iteration for subsets; ingest still scans
all matching NDJSON once to populate shards unless you also narrow
``glob_pattern``.

**Downstream memory (still important).** Streaming ingest avoids loading the
entire NDJSON corpus into RAM at once, but other steps can still be heavy on large
cohorts: building :class:`~pyhealth.data.Patient` / :class:`~pyhealth.datasets.FHIRPatient`
from ``fhir/resource_json`` parses JSON per patient; MPF vocabulary warmup and
:meth:`set_task` walk patients/samples; training needs RAM/VRAM for the model and
batches. For a **full** PhysioNet tree, plan for **large disk** (Parquet cache),
**comfortable system RAM** for Polars/PyArrow and task pipelines, and restrict
``glob_pattern`` or ``max_patients`` when prototyping on a laptop.

**Recommended hardware (informal)**

Order-of-magnitude guides, not guarantees. Ingest footprint is **much smaller**
than “load everything into Python”; wall time still grows with **decompressed
NDJSON volume** and shard/batch settings.

* **Smoke / CI**  
  Small on-disk fixtures (see tests and ``examples/mimic4fhir_mpf_ehrmamba.py``):
  a recent laptop is sufficient.

* **Laptop-scale real FHIR subset**  
  A **narrow** ``glob_pattern`` and/or ``max_patients`` in the hundreds keeps
  cache and task passes manageable. **≥ 16 GB** system RAM is a practical
  comfort target for Polars + trainer + OS; validate GPU **VRAM** for your
  ``max_len`` and batch size.

* **Full default glob on a complete export**  
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

.. autoclass:: pyhealth.datasets.FHIRPatient
    :members:
    :undoc-members:
    :show-inheritance:
