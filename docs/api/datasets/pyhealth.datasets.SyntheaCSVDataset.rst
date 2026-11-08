pyhealth.datasets.SyntheaCSVDataset
===================================

Loads the CSV export of `Synthea <https://github.com/synthetichealth/synthea>`_,
a synthetic patient simulator, through the standard PyHealth dataset API.
Synthea data is fully synthetic, so it needs no credentialing and is useful for
prototyping pipelines before you have access to MIMIC or eICU.

Getting Data
------------

``root`` is a directory of Synthea CSV files. There are two ways to get one:

- **Generate it** with :class:`~pyhealth.models.Synthea` (requires Java 17+;
  see its Setup section). This lets you choose the population size, seed,
  location, and age range.
- **Download a pre-generated sample** from the
  `Synthea downloads page <https://synthea.mitre.org/downloads>`_ and unzip it.
  No Java is needed; pick the CSV variant.

.. code-block:: python

    from pyhealth.datasets import SyntheaCSVDataset
    from pyhealth.models import Synthea

    csv_dir = Synthea("./synthea-output").generate(population=100, seed=42)
    # or: csv_dir = "/path/to/unzipped/sample/csv"  # directory with patients.csv

    dataset = SyntheaCSVDataset(
        csv_dir,
        tables=["conditions", "medications", "observations"],
    )
    dataset.stats()

How Synthea Differs From Other PyHealth Datasets
------------------------------------------------

**What Synthea writes is very different from MIMIC; what PyHealth gives you
back is the same.** The differences to watch for are in the raw files and in
the medical codes, not in the dataset object.

Raw output: many wide tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Synthea run writes about 18 CSV files. Each has many columns, every row is
keyed by a UUID, and the files include synthetic billing and insurance data.
For example, the first rows of ``conditions.csv`` and ``observations.csv``:

.. code-block:: text

    START,STOP,PATIENT,ENCOUNTER,SYSTEM,CODE,DESCRIPTION
    2004-07-17,,ba419d35-...-eebf02485a56,ba419d35-...-4d3d0f870810,SNOMED-CT,224299000,Received higher education (finding)

    DATE,PATIENT,ENCOUNTER,CATEGORY,CODE,DESCRIPTION,VALUE,UNITS,TYPE
    2023-08-12T20:03:25Z,ba419d35-...-eebf02485a56,ba419d35-...-b4779d5cf14b,laboratory,4548-4,Hemoglobin A1c/Hemoglobin.total in Blood,6.3,%,numeric

Not every file can be loaded, because PyHealth organizes events by patient:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Files
     - Loaded?
   * - ``patients``, ``encounters``, ``conditions``, ``medications``,
       ``observations``, ``procedures``, ``immunizations``, ``careplans``,
       ``allergies``, ``devices``, ``imaging_studies``, ``supplies``,
       ``payer_transitions``
     - Yes, by default. Files Synthea did not emit (no rows) are skipped.
   * - ``claims``, ``claims_transactions``
     - Only when listed in ``tables``. They are large and mostly billing
       detail.
   * - ``organizations``, ``providers``, ``payers``
     - No. They describe institutions, not patients, so they have no patient
       column.

After loading: the standard PyHealth event table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Like every PyHealth dataset, ``SyntheaCSVDataset`` turns each row of each table
into one **event** with three fixed columns, ``patient_id``, ``event_type`` (the
table name), and ``timestamp``, plus one ``<table>/<column>`` column per kept
attribute. Columns belonging to other tables are null for that event:

.. code-block:: text

    patient_id     event_type    timestamp            conditions/code  conditions/description  observations/code  observations/value  ...
    ba419d35-...   conditions    2004-07-17 00:00:00  224299000        Received higher ...     null               null
    ba419d35-...   observations  2023-08-12 20:03:25  null             null                    4548-4             6.3

Through the patient API, the table prefix is dropped:

.. code-block:: python

    patient = dataset.get_patient(dataset.unique_patient_ids[0])
    for event in patient.get_events(event_type="conditions"):
        print(event.timestamp, event.code, event.description)

    numeric_obs = patient.get_events(
        event_type="observations",
        filters=[("type", "==", "numeric")],
    )

The columns kept per table are listed in
``pyhealth/datasets/configs/synthea_csv.yaml``. Columns that identify a person
(name, SSN, street address, coordinates) are not loaded.

Medical codes: SNOMED-CT, RxNorm, LOINC, CVX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This difference matters most. MIMIC codes diagnoses with ICD-9/ICD-10 and drugs
with NDC; Synthea uses different vocabularies:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Table
     - Code system
     - Example ``code``
   * - ``conditions``
     - SNOMED-CT
     - ``714628002`` (Prediabetes)
   * - ``procedures``, ``encounters``, ``careplans``, ``devices``,
       ``supplies``
     - SNOMED-CT
     - ``252160004`` (Standard pregnancy test)
   * - ``medications``
     - RxNorm
     - ``849574`` (Naproxen sodium 220 MG Oral Tablet)
   * - ``observations``
     - LOINC
     - ``4548-4`` (Hemoglobin A1c)
   * - ``immunizations``
     - CVX
     - ``140`` (Influenza, split virus, trivalent, PF)
   * - ``allergies``
     - Mostly SNOMED-CT; drug allergies use RxNorm. The ``system`` column is
       often ``Unknown``, so do not rely on it.
     - ``84489001`` (Mold)

As a result:

- Built-in tasks written for MIMIC (for example
  :class:`~pyhealth.tasks.MortalityPredictionMIMIC3`) read ICD columns and do
  not work on Synthea. Write a task for Synthea instead (see :doc:`../tasks`).
- ICD-based code mappings in :mod:`pyhealth.medcode` (such as ICD to CCS) do
  not apply. RxNorm codes are supported by ``pyhealth.medcode.RxNorm``.
- Every code comes with a human-readable ``description`` column.

Other differences
^^^^^^^^^^^^^^^^^

- **Each patient has a ``patients`` event at their birth date.** Demographics
  (``gender``, ``race``, ``deathdate``, ...) are attributes of that event.
- **Visits are not a separate level.** Each clinical event has an
  ``encounter`` attribute holding the UUID of its encounter; group events by it
  to rebuild visits.
- **Lab values are strings.** ``observations/value`` holds numbers and text
  answers in one column; ``observations/type`` (``numeric`` or ``text``) says
  which.
- **Timestamps mix dates and date-times across tables** (for example
  ``conditions`` has dates, ``observations`` has date-times). PyHealth parses
  both; date-only events get a midnight timestamp.

.. autoclass:: pyhealth.datasets.SyntheaCSVDataset
    :members:
    :undoc-members:
    :show-inheritance:
