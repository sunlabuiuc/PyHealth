Datasets
===============

Getting Started
---------------

New to PyHealth datasets? Start here:

- **Tutorial**: `Introduction to pyhealth.datasets <https://colab.research.google.com/drive/1voSx7wEfzXfEf2sIfW6b-8p1KqMyuWxK?usp=sharing>`_ | `Video (PyHealth 1.6) <https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3>`_

This tutorial covers:

- How to load and work with different healthcare datasets (MIMIC-III, MIMIC-IV, eICU, etc.)
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

Native Datasets vs Custom Datasets
------------------------------------

PyHealth includes native support for several standard EHR databases — MIMIC-III,
MIMIC-IV, eICU, and OMOP. These come with built-in schema definitions so you
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
