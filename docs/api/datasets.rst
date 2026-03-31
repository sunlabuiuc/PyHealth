Datasets
===============

Getting Started
---------------

New to PyHealth datasets? Start here:

- **Tutorial**: `Introduction to pyhealth.datasets <https://colab.research.google.com/drive/1voSx7wEfzXfEf2sIfW6b-8p1KqMyuWxK?usp=sharing>`_ | `Video (PyHealth 1.6) <https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3>`_

This tutorial covers:

- How to load and work with any PyHealth dataset (MIMIC-III, MIMIC-IV, eICU, OMOP, and many more)
- Understanding the ``BaseDataset`` structure and patient representation
- Parsing raw EHR data into standardized PyHealth format
- Accessing patient records, visits, and clinical events
- Dataset splitting for train/validation/test sets

**Data Access**: If you're new and need help accessing MIMIC datasets, check the :doc:`../how_to_contribute` guide's "Data Access for Testing" section for information on:

- Getting MIMIC credentialing through PhysioNet
- Using openly available demo datasets (MIMIC-III Demo, MIMIC-IV Demo)
- Working with synthetic data for testing

How PyHealth Loads Data
------------------------

When you initialise a dataset, PyHealth reads the raw CSV or Parquet files
using Polars, joins the tables according to a YAML schema, and writes a
compact ``global_event_df.parquet`` cache to disk. On subsequent runs with
the same configuration it reads from cache rather than re-parsing the source
files, so startup is fast.

The result is a :class:`~pyhealth.datasets.BaseDataset` — a structured
patient→event tree. It is different from a PyTorch Dataset: it has no integer
length and you cannot index into it with ``dataset[i]``. Think of it as a
queryable dictionary of patient records. To turn it into something a model
can train on, you call ``dataset.set_task()`` (see :doc:`tasks`), which
returns a :class:`~pyhealth.datasets.SampleDataset` that *is* indexable and
DataLoader-ready.

From BaseDataset to SampleDataset
-----------------------------------

``BaseDataset`` and ``SampleDataset`` serve different roles and are not
interchangeable:

- **BaseDataset** is a queryable patient registry. It holds the raw
  patient→visit→event tree loaded from disk. You cannot index into it like a
  list — it has no integer length and is not DataLoader-ready.
- **SampleDataset** is a PyTorch-compatible streaming dataset returned by
  ``dataset.set_task()``. Each element is a fully processed feature
  dictionary that a model can consume directly.

The conversion happens in one call:

.. code-block:: python

    import torch
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import MortalityPredictionMIMIC3

    dataset = MIMIC3Dataset(root="...", tables=["diagnoses_icd"])
    samples = dataset.set_task(MortalityPredictionMIMIC3())
    # `samples` is a SampleDataset — pass it straight to a DataLoader
    loader = torch.utils.data.DataLoader(samples, batch_size=32)

Under the hood, ``set_task()`` runs a ``SampleBuilder`` that fits feature
processors (tokenisers, label encoders, etc.) across the full dataset, then
writes compressed, chunked sample files to disk via
`litdata <https://github.com/Lightning-AI/litdata>`_. A companion
``schema.pkl`` stores the fitted processors so the dataset can be reloaded in
future runs without re-fitting.

``SampleDataset`` also exposes two convenience lookups built during fitting:

- ``samples.patient_to_index`` — maps a patient ID to all sample indices for
  that patient.
- ``samples.record_to_index`` — maps a visit/record ID to the sample indices
  for that visit.

For testing or small cohorts you can skip the disk step entirely using
``InMemorySampleDataset``, which holds all processed samples in RAM and is
returned by default from ``create_sample_dataset()``.

.. note::
   Building a custom dataset or bringing your own data?
   See :doc:`../tutorials` (Tutorial 1) for a step-by-step walkthrough, and
   the `config.yaml for Custom Datasets`_ section below for the schema format.

Native Datasets vs Custom Datasets
------------------------------------

PyHealth includes native support for many standard healthcare databases — including
MIMIC-III, MIMIC-IV, eICU, OMOP, and many others (see the full list in `Available Datasets`_
below). All of these come with built-in schema definitions so you
can load them with just a root path and a list of tables:

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset

    if __name__ == '__main__':
        dataset = MIMIC3Dataset(
            root="/data/physionet.org/files/mimiciii/1.4",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=".cache",
            dev=True,   # use 1 000 patients while exploring
        )

For any other data source — a custom patient registry, an institutional cohort,
or a non-EHR dataset — you create a subclass of ``BaseDataset`` and provide a
``config.yaml`` file that describes your table structure.

Initialization Parameters
--------------------------

- **root** — path to the directory containing the raw data files. For MIMIC-IV
  specifically, use ``ehr_root`` instead of ``root``.
- **tables** — the table names you want to load, e.g.
  ``["diagnoses_icd", "labevents"]``. Only these tables will be accessible in
  patient queries downstream.
- **config_path** — path to your ``config.yaml``; needed for custom datasets.
  Native datasets have this built in and ignore the parameter.
- **cache_dir** — where to store the cached Parquet and LitData files. PyHealth
  appends a UUID derived from your configuration, so different setups never
  overwrite each other.
- **num_workers** — parallel processes for data loading. Increasing this can
  speed up ``set_task()`` on large datasets.
- **dev** — when ``True``, PyHealth caps the dataset at 1 000 patients. This
  is very useful during development because it makes each iteration complete in
  seconds rather than minutes. Switch to ``dev=False`` for your final training
  run.

Cache Directory Behavior
-------------------------

PyHealth derives the dataset cache UUID from the combination of:

- ``root``
- ``tables`` (sorted before hashing)
- ``dataset_name``
- ``dev``

This applies even when you pass ``cache_dir`` explicitly. In that case,
PyHealth treats your ``cache_dir`` as the parent folder and still appends a
configuration-specific UUID, so multiple dataset variants do not overwrite
each other.

Task caches live inside ``<cache_dir>/<dataset_uuid>/tasks/``. Their cache
key includes the task name, task class/module, task instance attributes
(``vars(task)``), plus the task input/output schemas. The final processed
sample cache additionally includes the input/output processor configuration.
If you change task code without changing those cache inputs, clear the stale
task cache before re-running.

Source and Compatibility Matrix
--------------------------------

The most common built-in datasets expect the following roots and files. For
custom datasets, the authoritative source of truth is your own
``config.yaml``: PyHealth will only look for the files referenced by
``file_path`` there.

.. list-table::
   :header-rows: 1
   :widths: 20 24 56

   * - Dataset class
     - Expected root
     - Minimum compatible files / notes
   * - ``MIMIC3Dataset``
     - PhysioNet MIMIC-III v1.4 root
     - Always requires ``PATIENTS.csv.gz``, ``ADMISSIONS.csv.gz``, and
       ``ICUSTAYS.csv.gz``. Optional tables map directly to
       ``DIAGNOSES_ICD.csv.gz``, ``PROCEDURES_ICD.csv.gz``,
       ``PRESCRIPTIONS.csv.gz``, ``LABEVENTS.csv.gz`` plus
       ``D_LABITEMS.csv.gz``, and ``NOTEEVENTS.csv.gz``.
   * - ``MIMIC4EHRDataset``
     - MIMIC-IV v2.2 EHR root
     - Requires the ``hosp/`` and ``icu/`` subdirectories. Core files are
       ``hosp/patients.csv.gz``, ``hosp/admissions.csv.gz``, and
       ``icu/icustays.csv.gz``. Optional tables include
       ``hosp/diagnoses_icd.csv.gz``, ``hosp/procedures_icd.csv.gz``,
       ``hosp/prescriptions.csv.gz``, ``hosp/labevents.csv.gz`` plus
       ``hosp/d_labitems.csv.gz``, and ``hosp/hcpcsevents.csv.gz``.
   * - ``MIMIC4NoteDataset``
     - MIMIC-IV-Note v2.2 root
     - Requires the ``note/`` subdirectory. Supported tables are
       ``note/discharge.csv.gz``, ``note/discharge_detail.csv.gz``,
       ``note/radiology.csv.gz``, and ``note/radiology_detail.csv.gz``.
       Use this with a notes root, not the EHR root.
   * - ``MIMIC4CXRDataset``
     - MIMIC-CXR-JPG root
     - Requires ``mimic-cxr-2.0.0-metadata.csv.gz`` and the JPEG ``files/``
       tree so PyHealth can build ``image_path`` entries. Optional labels and
       splits come from ``mimic-cxr-2.0.0-chexpert.csv.gz``,
       ``mimic-cxr-2.0.0-negbio.csv.gz``,
       ``mimic-cxr-2.1.0-test-set-labeled.csv``, and
       ``mimic-cxr-2.0.0-split.csv.gz``. PyHealth generates
       ``mimic-cxr-2.0.0-metadata-pyhealth.csv`` on first load. DICOM-only
       MIMIC-CXR exports are not sufficient for this loader.
   * - ``MIMIC4Dataset``
     - Combined MIMIC-IV roots
     - Pass any combination of ``ehr_root``, ``note_root``, and ``cxr_root``.
       Each supplied root must satisfy the compatibility requirements of the
       corresponding child dataset above.
   * - ``eICUDataset``
     - eICU-CRD v2.0 root
     - Requires ``patient.csv``. Additional supported files are
       ``hospital.csv``, ``diagnosis.csv``, ``medication.csv``,
       ``treatment.csv``, ``lab.csv``, ``physicalExam.csv``, and
       ``admissionDx.csv``.
   * - ``OMOPDataset``
     - OMOP CDM v5.3 export root
     - Requires ``person.csv``, ``visit_occurrence.csv``, and ``death.csv``.
       Additional supported files are ``condition_occurrence.csv``,
       ``procedure_occurrence.csv``, ``drug_exposure.csv``, and
       ``measurement.csv``.

config.yaml for Custom Datasets
---------------------------------

If you are bringing your own data, the YAML file tells PyHealth which column
is the patient identifier, which column is the timestamp, and which other
columns to include as event attributes:

.. code-block:: yaml

    tables:
      my_table:
        file_path: relative/path/to/file.csv
        patient_id: subject_id
        timestamp: charttime
        timestamp_format: "%Y-%m-%d %H:%M:%S"
        attributes:
          - icd_code
          - value
          - itemid
        join: []   # optional table joins

All attribute column names are lowercased internally, so ``ICD_CODE`` in
your CSV becomes ``icd_code`` in your code.

Querying Patients and Events
-----------------------------

Once a dataset is loaded, you can explore it using these methods:

.. code-block:: python

    dataset.unique_patient_ids          # all patient IDs as a list of strings
    dataset.get_patient("p001")         # retrieve one Patient object
    dataset.iter_patients()             # iterate over all patients
    dataset.stats()                     # print patient and event counts

Patient records are accessed through ``get_events()``, which supports
temporal filtering and attribute-level filters:

.. code-block:: python

    events = patient.get_events(
        event_type="diagnoses_icd",             # table name from your config
        start=datetime(2020, 1, 1),             # optional: exclude earlier events
        end=datetime(2020, 6, 1),               # optional: exclude later events
        filters=[("icd_code", "==", "250.00")], # optional: attribute conditions
    )

Each event in the returned list has:

- ``event.timestamp`` — a Python ``datetime`` object. PyHealth normalises all
  timestamp columns (``charttime``, ``admittime``, etc.) into this single
  property, so this is what you should use regardless of what the original
  column was called.
- ``event.icd_code``, ``event["icd_code"]``, ``event.attr_dict`` — different
  ways to access the other attributes. All attribute names are lowercase.

Things to Watch Out For
------------------------

A few patterns that commonly trip up new users:

**BaseDataset vs SampleDataset.** Models expect a ``SampleDataset`` (the
output of ``set_task()``), not the raw ``BaseDataset``. Passing the wrong one
will raise an error. If you see an ``AttributeError`` about ``input_schema``
or ``output_schema``, this is likely the cause.

**Timestamp attribute names.** Writing ``event.charttime`` will raise an
``AttributeError`` because PyHealth remaps that column to ``event.timestamp``.
The same applies to ``admittime``, ``starttime``, or whatever the original
column was called.

**Column name casing.** PyHealth lowercases all column names at load time.
Even if your source CSV has ``ICD_CODE``, you access it as ``event.icd_code``.

**dev=True in production.** The ``dev`` flag is great for exploring data but
it caps the dataset at 1 000 patients. Remember to switch to ``dev=False``
before running a full training job.

**Multiprocessing guard.** Scripts that call ``set_task()`` should wrap their
top-level code in ``if __name__ == '__main__':``. See :doc:`tasks` for details.

Available Datasets
------------------

.. toctree::
    :maxdepth: 3

    datasets/pyhealth.datasets.BaseDataset
    datasets/pyhealth.datasets.SampleDataset
    datasets/pyhealth.datasets.MIMIC3Dataset
    datasets/pyhealth.datasets.MIMIC4Dataset
    datasets/pyhealth.datasets.MedicalTranscriptionsDataset
    datasets/pyhealth.datasets.CardiologyDataset
    datasets/pyhealth.datasets.eICUDataset
    datasets/pyhealth.datasets.ISRUCDataset
    datasets/pyhealth.datasets.MIMICExtractDataset
    datasets/pyhealth.datasets.OMOPDataset
    datasets/pyhealth.datasets.DREAMTDataset
    datasets/pyhealth.datasets.SHHSDataset
    datasets/pyhealth.datasets.SleepEDFDataset
    datasets/pyhealth.datasets.EHRShotDataset
    datasets/pyhealth.datasets.Support2Dataset
    datasets/pyhealth.datasets.BMDHSDataset
    datasets/pyhealth.datasets.COVID19CXRDataset
    datasets/pyhealth.datasets.ChestXray14Dataset
    datasets/pyhealth.datasets.TUABDataset
    datasets/pyhealth.datasets.TUEVDataset
    datasets/pyhealth.datasets.ClinVarDataset
    datasets/pyhealth.datasets.COSMICDataset
    datasets/pyhealth.datasets.TCGAPRADDataset
    datasets/pyhealth.datasets.splitter
    datasets/pyhealth.datasets.utils
