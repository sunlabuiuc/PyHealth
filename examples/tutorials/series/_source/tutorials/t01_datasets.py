from nbkit import code, footer, header, md

FILENAME = "01_datasets.ipynb"


def cells():
    return [
        *header(
            "01",
            "Datasets: patients, events and your own data",
            [
                "The difference between a dataset and a task, and why PyHealth keeps them apart",
                "How to explore patients and their events, filter by type, time or admission",
                "What the YAML config does, and how caching works",
                "How to load your own CSV files with a short config, no new class needed",
            ],
            20,
            "Tutorial 00 (Quickstart) is helpful but not required.",
        ),
        md("""
        ## A dataset is not a task

        ![Datasets feed many tasks](https://drive.google.com/uc?export=view&id=1hHJcavXqisH9JEMqEVtE4TqEg_E489l5)

        A **dataset** in PyHealth is a pool of raw patient data: who the
        patients are and everything that happened to them, in time order. It
        does not know what you want to predict.

        A **task** (Tutorial 02) reads that pool and builds labelled examples
        for one question. The same MIMIC-III dataset serves mortality
        prediction, readmission prediction, length of stay, drug
        recommendation and medical coding, each a different task.

        Keeping them apart means you load and clean the data once, then ask
        many questions of it. When you contribute to PyHealth, a new data
        *source* belongs in `pyhealth.datasets`; a new *question* about an
        existing source (say, a new MIMIC-III outcome) belongs in
        `pyhealth.tasks`.
        """),
        md("""
        ## Loading a built-in dataset

        PyHealth has loaders for many sources, including `MIMIC3Dataset`,
        `MIMIC4Dataset`, `eICUDataset`, `OMOPDataset` (any OMOP-CDM database),
        and imaging, signal and text datasets. They all share the same
        arguments:

        | Argument | Meaning |
        |---|---|
        | `root` | Folder or URL with the raw files |
        | `tables` | Which tables to load (core tables such as patients and admissions are always added) |
        | `cache_dir` | Where the processed copy is stored; reused on the next run |
        | `dev` | `True` keeps only 1,000 patients, for quick experiments |

        We use the public synthetic MIMIC-III, with four tables. If you ran
        Tutorial 00 in the same runtime, this loads from the cache in seconds.
        PyHealth logs each step as it goes; the long log is normal.
        """),
        code("""
        from pyhealth.datasets import MIMIC3Dataset

        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
            tables=["diagnoses_icd", "procedures_icd", "prescriptions", "noteevents"],
            cache_dir="pyhealth_cache",
        )
        dataset.stats()
        """),
        md("""
        ## Patients and events

        Everything in a dataset is an **event**: one row of one table, tied to
        a patient and (usually) a time. An event has:

        - `event_type`: the table it came from (`"admissions"`, `"diagnoses_icd"`, ...)
        - `timestamp`: when it happened (`None` for timeless rows such as demographics)
        - attributes: the table columns listed in the config, readable as
          `event.icd9_code` or `event["icd9_code"]`

        A **patient** is that patient's events in time order.
        """),
        code("""
        patient_ids = dataset.unique_patient_ids
        patient = dataset.get_patient(patient_ids[0])

        events = patient.get_events()
        print(f"patient {patient.patient_id} has {len(events)} events")
        for event in events[:3]:
            print(event.event_type, event.timestamp, dict(event.attr_dict))
        """),
        md("""
        ### Picking out the events you need

        `get_events` takes three kinds of filters, which you can combine:

        - `event_type="diagnoses_icd"`: events from one table
        - `start=` / `end=`: events within a time window
        - `filters=[("hadm_id", "==", "...")]`: events whose attribute matches,
          for example everything recorded during one hospital admission
        """),
        code("""
        # every diagnosis code this patient ever received
        diagnoses = patient.get_events(event_type="diagnoses_icd")
        print("diagnosis codes:", [e.icd9_code for e in diagnoses][:10])

        # the patient's admissions, in time order
        admissions = patient.get_events(event_type="admissions")
        for a in admissions:
            print(f"admission {a.hadm_id}: {a.timestamp} -> {a.dischtime}, died in hospital: {a.hospital_expire_flag}")
        """),
        code("""
        # what happened during the first admission only
        first = admissions[0]
        in_first = patient.get_events(event_type="prescriptions", filters=[("hadm_id", "==", first.hadm_id)])
        print(f"{len(in_first)} drug orders in admission {first.hadm_id}:",
              sorted({e.drug for e in in_first})[:8])

        # and everything from that admission's start onwards
        later = patient.get_events(start=first.timestamp)
        print(f"{len(later)} events from {first.timestamp} on")
        """),
        md("""
        For analysis, `return_df=True` gives the events as a
        [polars](https://pola.rs) DataFrame instead of a list. Attribute
        columns are named `<table>/<attribute>`; the frame has a column for
        every attribute of every loaded table, so select the ones you need:
        """),
        code("""
        df = patient.get_events(event_type="diagnoses_icd", return_df=True)
        df.select(["timestamp", "diagnoses_icd/hadm_id", "diagnoses_icd/icd9_code", "diagnoses_icd/seq_num"])
        """),
        md("""
        Looping over many patients is the same idea. Here we count how many
        patients have more than one admission, which is what the mortality task
        in Tutorial 00 needs:
        """),
        code("""
        from itertools import islice

        multi = 0
        checked = 0
        for pid in islice(patient_ids, 2000):
            checked += 1
            if len(dataset.get_patient(pid).get_events(event_type="admissions")) > 1:
                multi += 1
        print(f"{multi} of the first {checked} patients have 2+ admissions")
        """),
        md("""
        ## Under the hood: the YAML config

        How does PyHealth know which column is the patient id, which is the
        time, and which columns to keep? Each dataset has a YAML config. Here
        is part of the one used above, read straight from the installed
        package:
        """),
        code("""
        from pathlib import Path
        import pyhealth.datasets

        config = Path(pyhealth.datasets.__file__).parent / "configs" / "mimic3.yaml"
        text = config.read_text()
        start = text.index("  diagnoses_icd:")
        print(text[:text.index("  admissions:")])
        print(text[start:text.index("\\n\\n", start)])
        """),
        md("""
        Every table entry has:

        | Field | Meaning |
        |---|---|
        | `file_path` | The file under `root` |
        | `patient_id` | The column that identifies the patient |
        | `timestamp` | The column with the event time (`null` for timeless tables) |
        | `attributes` | The columns to keep; they become event attributes |
        | `join` (optional) | Columns to pull in from another file. `diagnoses_icd` has no time of its own, so it borrows `dischtime` from `ADMISSIONS` |

        ## Caching

        Reading and joining the raw tables is the slow part, so PyHealth
        stores the result under `cache_dir` and reuses it. The cache entry is
        keyed on the dataset's settings and on the config and source files:
        change a table or edit the config, and you get a fresh entry instead
        of stale data. Task outputs (Tutorial 02) are cached the same way.

        Two practical notes:
        - Without `cache_dir`, PyHealth uses a default folder in your home
          directory and prints a warning saying where. With real patient
          data, always pass a `cache_dir` inside storage you control.
        - To make that mandatory (for example on a shared server), set the
          environment variable `PYHEALTH_REQUIRE_CACHE_DIR=1`; loading a
          dataset without `cache_dir` then raises an error.
        """),
        md("""
        ## Your own data: CSV files plus a config

        You do not need to write a Python class to use your own data. Put each
        table in a CSV file, describe them in a YAML config, and load them
        with `BaseDataset`. Here we create a tiny clinic dataset with a
        demographics table and a visits table.
        """),
        code("""
        from pathlib import Path

        root = Path("my_clinic")
        root.mkdir(exist_ok=True)

        (root / "patients.csv").write_text(
            "patient_id,birth_year,sex\\n"
            "p1,1956,F\\n"
            "p2,1971,M\\n"
            "p3,1990,F\\n"
        )
        (root / "visits.csv").write_text(
            "patient_id,visit_time,diagnosis,systolic_bp\\n"
            "p1,2023-01-05 09:30,I10,152\\n"
            "p1,2023-06-12 14:00,E11.9,148\\n"
            "p1,2024-02-01 10:15,I10,139\\n"
            "p2,2023-03-20 08:45,J45.909,121\\n"
            "p3,2023-11-02 16:20,I10,135\\n"
            "p3,2024-04-18 11:05,N18.3,141\\n"
        )
        (root / "my_clinic.yaml").write_text('''
        version: "1.0"
        tables:
          patients:
            file_path: "patients.csv"
            patient_id: "patient_id"
            timestamp: null
            attributes: ["birth_year", "sex"]
          visits:
            file_path: "visits.csv"
            patient_id: "patient_id"
            timestamp: "visit_time"
            timestamp_format: "%Y-%m-%d %H:%M"
            attributes: ["diagnosis", "systolic_bp"]
        ''')
        print(sorted(p.name for p in root.iterdir()))
        """),
        code("""
        from pyhealth.datasets import BaseDataset

        clinic = BaseDataset(
            root=str(root),
            tables=["patients", "visits"],
            dataset_name="my_clinic",
            config_path=str(root / "my_clinic.yaml"),
            cache_dir="pyhealth_cache",
        )
        clinic.stats()

        for event in clinic.get_patient("p1").get_events(event_type="visits"):
            print(event.timestamp, event.diagnosis, event.systolic_bp)
        """),
        md("""
        That is all it takes: your data now works with every PyHealth task,
        model and evaluation tool, exactly like MIMIC-III. Values are read as
        text, so convert them in your task (for example
        `float(event.systolic_bp)`).

        If you later want to share the loader with others, wrap it in a small
        `BaseDataset` subclass and contribute it; Tutorial 08 walks through
        that, using PyHealth's own MIMIC-III and COVID-19 X-ray loaders as
        examples.

        ## Summary

        - A dataset holds patients; a patient is a time-ordered list of events.
        - `get_events(event_type=..., start=..., end=..., filters=...)` selects
          what you need; `return_df=True` gives a DataFrame.
        - A YAML config maps raw files to events; `cache_dir` keeps the
          processed copy.
        - Your own CSVs load with `BaseDataset` and a config.
        """),
        *footer("02"),
    ]
