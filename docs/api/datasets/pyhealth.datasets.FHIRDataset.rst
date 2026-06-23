pyhealth.datasets.FHIRDataset
=====================================

A generic, config-driven NDJSON ingest for `HL7 FHIR
<https://www.hl7.org/fhir/>`_ datasets. The whole pipeline is described by **a
single YAML config** with three top-level sections — what files to read, how to
turn each FHIR resource into a flat row, and how those rows appear as events
downstream. A custom FHIR ingest is "point at a YAML" — no Python required.

The bundled :class:`~pyhealth.datasets.MIMIC4FHIR` subclass uses this engine
with the ``pyhealth/datasets/fhir/configs/mimic4fhir.yaml`` config tuned for
PhysioNet's MIMIC-IV on FHIR export. See the sub-page below for the quick-start.

.. contents:: On this page
   :local:
   :depth: 1


Quick start
-----------

.. code-block:: python

    from pyhealth.datasets import MIMIC4FHIR, get_dataloader, split_by_patient
    from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask
    from pyhealth.models import EHRMambaCEHR
    from pyhealth.trainer import Trainer

    def main():
        ds = MIMIC4FHIR(root="/data/mimic-iv-fhir")
        sample_ds = ds.set_task(MPFClinicalPredictionTask(), num_workers=1)
        train, val, test = split_by_patient(sample_ds, [0.7, 0.1, 0.2])
        vocab_size = sample_ds.input_processors["concept_ids"].vocab.vocab_size
        model = EHRMambaCEHR(dataset=sample_ds, vocab_size=vocab_size)
        Trainer(model=model).train(
            train_dataloader=get_dataloader(train, batch_size=8, shuffle=True),
            val_dataloader=get_dataloader(val, batch_size=8),
            epochs=2,
        )

    if __name__ == "__main__":
        main()

(``if __name__ == "__main__":`` matters — :meth:`~pyhealth.datasets.BaseDataset.set_task`
forks Dask workers; without the guard the workers re-import and re-spawn.)


Pipeline at a glance
--------------------

::

    NDJSON shards on disk
        |
        |  (Phase A) — stream line by line, route by resourceType,
        |             project via the YAML's resource_specs
        v
    flattened_tables/<table>.parquet         <- cache #1
        |
        |  (Phase B) — load_table, dd.concat, sort by patient_id (Dask)
        v
    global_event_df.parquet/part-*.parquet   <- cache #2
        |
        |  (Phase C) — task_transform per-patient sample emit
        v
    task_df.ld/        <- cache #3a
        |
        |  fit CehrProcessor vocab via SampleBuilder.fit(dataset)
        |  proc_transform per-sample tensorisation
        v
    samples_*.ld/      <- cache #3b   ──>   SampleDataset

Each of the three cache tiers has its own existence check; re-running with
identical inputs skips every phase. Cache identity hashes the YAML byte digest,
glob patterns, ``max_patients``, and engine schema version — any meaningful
config change invalidates everything below it. See
:class:`~pyhealth.datasets.BaseDataset` for the Phase B/C internals that are
shared with all other PyHealth datasets.


The unified YAML config
-----------------------

A FHIR ingest YAML has three top-level sections. The bundled
``mimic4fhir.yaml`` is the canonical worked example; what follows is the
section-by-section reference.

Section 1: ``glob_patterns:`` (which files to read)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    glob_patterns:
      - "**/MimicPatient*.ndjson.gz"
      - "**/MimicEncounter*.ndjson.gz"
      # ... one pattern per resource-type shard family

Defaults to ``["**/*.ndjson.gz"]`` when omitted. Only worth setting when your
export has a per-resource-type file-naming convention you want to exploit for
speed — PhysioNet MIMIC-IV FHIR ships shards as ``MimicPatient*.ndjson.gz``,
``MimicEncounter*.ndjson.gz``, etc., and filtering at the file level avoids
decompressing ~10% of the export that contains only unconfigured resource
types. For a generic export where everything is in ``bundles.ndjson.gz``, omit
this block and the streamer will filter by ``resourceType`` after parsing.

Override at runtime via ``MIMIC4FHIR(glob_pattern=...)`` or
``MIMIC4FHIR(glob_patterns=[...])``.

Section 2: ``resource_specs:`` (how to project JSON into rows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keys are FHIR ``resourceType`` strings. For each, declare a ``table`` name and
an ordered ``columns`` mapping:

.. code-block:: yaml

    resource_specs:

      Patient:
        table: patient
        columns:
          patient_id:        { locate: ["id"], required: true }
          birth_date:        { locate: ["birthDate"] }
          gender:            { locate: ["gender"] }
          deceased_boolean:  { locate: ["deceasedBoolean"], transform: bool_norm }

      Observation:
        table: observation
        columns:
          patient_id:    { locate: ["subject.reference"], transform: ref_id, required: true }
          resource_id:   { locate: ["id"] }
          encounter_id:  { locate: ["encounter.reference"], transform: ref_id }
          event_time:    { locate: ["effectiveDateTime", "effectivePeriod.start", "issued"] }
          concept_key:   { locate: ["code"], transform: coding_key }

Each column entry has three fields:

``locate`` *(required, list of dotted paths)*
    Ordered JSON paths into the resource; the first that resolves to a non-null
    value wins. This is how FHIR choice-types (``onset[x]``, ``effective[x]``,
    ``performed[x]``, …) are handled — list every variant explicitly. A single
    string is accepted as shorthand for a one-element list.

``transform`` *(optional, name of a built-in transform, default ``identity``)*
    Maps the located leaf to a flat scalar string. See the registry below.

``required`` *(optional, bool, default false)*
    When ``true``, a resource whose ``locate`` cannot be resolved is **dropped**
    (and logged) rather than emitted with a null. Use this on the patient
    reference column so events without a discoverable patient never reach the
    global event frame.

Transform registry
^^^^^^^^^^^^^^^^^^

Available transforms (defined in
``pyhealth/datasets/fhir/utils.py`` ``TRANSFORMS`` dict):

==================  ===========================================================
``identity``        Pass the value through. Stringifies non-string scalars.
``ref_id``          Reference object or ``"Patient/p1"`` -> ``"p1"``.
``coding_key``      CodeableConcept -> ``"system|code"`` of its first coding.
``bool_norm``       JSON boolean / ``"true"``/``"false"`` -> ``"true"``/``"false"``/None.
``med_concept``     MedicationRequest medication[x] -> codeable-concept or
                    ``"MedicationRequest/reference|<id>"`` fallback.
==================  ===========================================================

Adding a new transform is a one-liner: register a callable in ``TRANSFORMS``
in ``utils.py`` and reference it by name from the YAML.

Section 3: ``tables:`` (how rows are exposed as events)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keys here must match the ``table:`` values from Section 2. Each entry tells
:meth:`~pyhealth.datasets.BaseDataset.load_table` how to read the flat parquet:

.. code-block:: yaml

    tables:
      patient:
        file_path: "patient.parquet"
        patient_id: "patient_id"
        timestamp: "birth_date"
        attributes: ["birth_date", "gender", "deceased_boolean"]

      observation:
        file_path: "observation.parquet"
        patient_id: "patient_id"
        timestamp: "event_time"
        attributes: ["resource_id", "encounter_id", "event_time", "concept_key"]

``file_path`` is the parquet filename inside the cached
``flattened_tables/`` directory. ``patient_id`` and ``timestamp`` name the
columns to surface as the normalised ``patient_id`` and ``timestamp`` on each
event. ``attributes`` is the list of columns surfaced as event attributes — in
the global event frame they're renamed to ``{table}/{attr}`` and later show up
on ``patient.get_events(event_type=...).attr_name``.

Cross-section validation
~~~~~~~~~~~~~~~~~~~~~~~~

At load time the dataset checks that every ``table:`` value declared in
Section 2 has a matching ``tables.<name>`` block in Section 3. Typos surface
as a config error at startup, not silent empty parquets.


Customising for a non-MIMIC FHIR export
---------------------------------------

Step 1 — write your YAML.
~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``pyhealth/datasets/fhir/configs/mimic4fhir.yaml`` and adapt the
``resource_specs:`` and ``tables:`` blocks for the resources you care about.
For an export that adds Immunizations:

.. code-block:: yaml

    resource_specs:
      Patient:
        table: patient
        columns:
          patient_id: { locate: ["id"], required: true }
          birth_date: { locate: ["birthDate"] }
      Immunization:
        table: immunization
        columns:
          patient_id:   { locate: ["patient.reference"], transform: ref_id, required: true }
          resource_id:  { locate: ["id"] }
          event_time:   { locate: ["occurrenceDateTime", "recorded"] }
          concept_key:  { locate: ["vaccineCode"], transform: coding_key }

    tables:
      patient:
        file_path: "patient.parquet"
        patient_id: "patient_id"
        timestamp: "birth_date"
        attributes: ["birth_date"]
      immunization:
        file_path: "immunization.parquet"
        patient_id: "patient_id"
        timestamp: "event_time"
        attributes: ["resource_id", "event_time", "concept_key"]

Step 2 — instantiate
~~~~~~~~~~~~~~~~~~~~

Either pass ``config_path=...`` directly:

.. code-block:: python

    from pyhealth.datasets import FHIRDataset

    ds = FHIRDataset(
        root="/data/my_fhir_export",
        config_path="/path/to/my_export.yaml",
    )

or write a 3-line subclass that bundles your config:

.. code-block:: python

    from pyhealth.datasets import FHIRDataset

    class MyFHIR(FHIRDataset):
        DEFAULT_CONFIG_PATH = "/path/to/my_export.yaml"

    ds = MyFHIR(root="/data/my_fhir_export")

Step 3 — that's it.
~~~~~~~~~~~~~~~~~~~

Everything downstream — :meth:`~pyhealth.datasets.BaseDataset.set_task`,
:meth:`~pyhealth.datasets.BaseDataset.iter_patients`,
:meth:`~pyhealth.datasets.BaseDataset.get_patient` — works the same as for any
other PyHealth dataset.


Notes on resource use
---------------------

Streaming ingest avoids loading the whole NDJSON corpus into RAM, but downstream
steps still scale with cohort size. For a **smoke run** the bundled example
fixtures fit on any laptop. For a **laptop-scale real subset**, set
``max_patients=`` and/or narrow ``glob_patterns`` to keep cache and task passes
manageable; ≥16 GB system RAM is a comfort target for Polars + the trainer.
For the **full PhysioNet export**, prefer fast SSD, large disk, and plenty of
RAM — total work scales with the corpus size even if RAM ingest is bounded.


Bundled FHIR datasets
---------------------

.. toctree::
   :maxdepth: 1

   pyhealth.datasets.MIMIC4FHIR


API reference
-------------

.. autoclass:: pyhealth.datasets.FHIRDataset
    :members:
    :undoc-members:
    :show-inheritance:
