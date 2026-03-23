pyhealth.datasets.MIMIC4FHIRDataset
=====================================

`MIMIC-IV on FHIR <https://physionet.org/content/mimic-iv-fhir/>`_ NDJSON ingest
for CEHR-style token sequences used with
:class:`~pyhealth.tasks.mpf_clinical_prediction.MPFClinicalPredictionTask` and
:class:`~pyhealth.models.EHRMambaCEHR`.

YAML defaults live in ``pyhealth/datasets/configs/mimic4_fhir.yaml`` (e.g.
``glob_pattern`` for ``**/*.ndjson.gz`` on PhysioNet exports, or ``**/*.ndjson``
if uncompressed). The loader subclasses :class:`~pyhealth.datasets.BaseDataset`
but does not build a Polars ``global_event_df``; use :meth:`MIMIC4FHIRDataset.gather_samples`
or :meth:`MIMIC4FHIRDataset.set_task`.

**Limitations.** Ingest is **non-streaming**: all lines from
every glob-matched file are materialized in RAM before patient grouping. A full
PhysioNet MIMIC-IV FHIR download can exceed **tens–hundreds of GB** and dominate
runtime. Use a **narrow** ``glob_pattern`` for development, or extend the loader
with chunked/streaming IO in a follow-up. ``max_patients`` does not reduce peak
load memory—it only caps how many grouped patients are kept **after** the full
parse.

**Recommended hardware (informal)**

These are order-of-magnitude guides, not guarantees. Peak RAM is dominated by the
**parse phase** (all lines from matched ``.ndjson.gz`` / ``.ndjson`` files held in
memory before ``max_patients`` trims grouped patients). **GPU** (CUDA/MPS) speeds
**training and inference** once tensors are built but does **not** remove that parse
cost—plan system RAM from the ingest footprint first.

* **Smoke / CI — synthetic**  
  ``examples/mimic4fhir_mpf_ehrmamba.py --quick-test`` on CPU: ~**600–700 MiB**
  peak RSS, **~10–15 s** wall for two short epochs (two-patient fixture). Any
  recent laptop is sufficient.

* **Laptop-scale real FHIR subset**  
  A **narrow** ``glob_pattern`` (e.g. Patient + Encounter + Condition
  ``.ndjson.gz`` only), ``max_patients`` in the hundreds, and multi-epoch training
  via ``examples/mimic4fhir_mpf_ehrmamba.py`` on **CPU**: expect **several GiB**
  peak RSS and **many minutes** wall time per run. Treat **≥ 16 GB** system RAM as
  a practical minimum with OS headroom; **8 GB** is often too tight. Use a
  **GPU** for longer training (set device / ``CUDA_VISIBLE_DEVICES`` in your
  runner); **VRAM** must fit model + batch—validate for your ``max_len`` and batch
  size (small CEHR-style configs often use **8+ GB** GPU memory).

* **Scaling**  
  Peak RSS and wall time **grow** with **decompressed NDJSON volume** (wider
  ``glob_pattern`` or more files before ``max_patients``) and with **larger**
  ``max_len``, vocabulary, batch size, and ``hidden_dim``.

* **Full PhysioNet tree**  
  Default ``**/*.ndjson.gz`` over the **entire** export is aimed at **workstations
  or servers** with **very large RAM** (often **64 GB+**) and fast **SSD** I/O, or
  at workflows that pass a **narrow** ``glob_pattern``. Not recommended on **16 GB**
  laptops without subsetting.

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
