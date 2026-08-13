---
name: pyhealth-use-a-dataset
description: Pick and load a built-in PyHealth dataset (MIMIC-III/IV, eICU, OMOP, EHRShot, signal and imaging sets) — which class, which tables, which column becomes event.timestamp versus a named attribute, and how to filter events to a visit. Use when the data is already a supported dataset.
---

# Use a built-in PyHealth dataset

Explains how PyHealth loads tables and
which columns become `event.timestamp` vs named attributes on `Event` objects.

---

## Critical: `if __name__ == '__main__':` Guard

PyHealth uses multiprocessing internally. Any dataset operation at module-level (outside
a function or the `__main__` guard) will crash worker processes. **All dataset code must
be inside the guard.**

```python
# WRONG — crashes worker processes
dataset = MIMIC4Dataset(ehr_root=..., ehr_tables=..., dev=True)
patient = list(dataset.patients.values())[0]

# CORRECT — always use this pattern in exploration scripts
if __name__ == '__main__':
    dataset = MIMIC4Dataset(ehr_root=..., ehr_tables=..., dev=True)
    patient = list(dataset.patients.values())[0]
```

This guard is **required** in every dataset exploration script, including any exploration script.

---

## How PyHealth Datasets Work

Each table in the dataset YAML has three important fields:

```yaml
  <table_name>:
    patient_id: "<column>"   # identifies the patient; stored as event.patient_id
    timestamp:  "<column>"   # ← THIS COLUMN BECOMES event.timestamp
                             #   it is NOT accessible as event.<column_name>
    attributes:              # these columns become event.<column_name> attributes
      - "col_a"
      - "col_b"
```

### The Timestamp Rule (critical)

The column named under `timestamp:` is mapped to `event.timestamp` (a Python
`datetime` object). It is **not** stored as a named attribute — so
`getattr(event, "admittime", None)` always returns `None` even though `admittime`
is what the YAML says.

```python
# WRONG — always None, will silently skip all samples
admit_time = getattr(visit, "admittime", None)

# CORRECT — always works
admit_time = visit.timestamp
```

Tables with `timestamp: null` produce events where `event.timestamp` is `NaT`
(not useful for time-ordering or timeseries).

### Attributes

Every column listed under `attributes:` is accessible as a named attribute via
`event.<column_name>` (e.g. `event.hadm_id`, `event.icd_code`, `event.valuenum`).

### The Case Rule (surprising, and it fails silently)

`tables=` is case-insensitive; `event_type=` is **not**.

```python
# Both fine — load_data() lowercases the list (pyhealth/datasets/base_dataset.py:656)
MIMIC3Dataset(root=..., tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"])
MIMIC3Dataset(root=..., tables=["diagnoses_icd", "prescriptions"])

# Silently returns [] — no error, no warning, and your task yields zero samples
patient.get_events(event_type="DIAGNOSES_ICD")

# Correct — event_type is always the lowercase config key
patient.get_events(event_type="diagnoses_icd")
```

Events are stamped `event_type=table_name` from the already-lowercased name
(`base_dataset.py:754`), so the stored type is always lowercase. Worse, when you also pass
`filters`, `get_events` builds the column `"DIAGNOSES_ICD/hadm_id"`
(`pyhealth/data/data.py:216`), which does not exist — raising a polars `ColumnNotFoundError`
far from the actual mistake.

**If a task returns zero samples, check this first.**

### Reading the full YAML

To see every attribute available for a specific table, read the full YAML.
All configs live in `pyhealth/datasets/configs/`:

```
`pyhealth/datasets/configs/mimic4_ehr.yaml`  # MIMIC-IV EHR
`pyhealth/datasets/configs/mimic3.yaml`      # MIMIC-III
`pyhealth/datasets/configs/eicu.yaml`        # eICU
`pyhealth/datasets/configs/omop.yaml`        # OMOP CDM
`pyhealth/datasets/configs/mimic4_cxr.yaml`  # MIMIC-IV CXR
`pyhealth/datasets/configs/mimic4_note.yaml` # MIMIC-IV Notes
`pyhealth/datasets/configs/ehrshot.yaml`     # EHRSHOT
```

---

## MIMIC-IV EHR (`mimic4_ehr.yaml`)

**Dataset class**: `MIMIC4Dataset` (via `MIMIC4EHRDataset` internally)

| Table | `event.timestamp` comes from | Key attributes (partial) |
|-------|------------------------------|--------------------------|
| `patients` | `null` (NaT) | `gender`, `anchor_age`, `dod` |
| `admissions` | `admittime` | `hadm_id`, `admission_type`, `race`, `hospital_expire_flag`, `dischtime` |
| `icustays` | `intime` | `hadm_id`, `stay_id`, `first_careunit`, `last_careunit`, `outtime` |
| `diagnoses_icd` | `dischtime` | `hadm_id`, `icd_code`, `icd_version` |
| `procedures_icd` | `dischtime` | `hadm_id`, `icd_code`, `icd_version` |
| `prescriptions` | `starttime` | `hadm_id`, `drug`, `route`, `dose_val_rx` |
| `labevents` | `charttime` | `hadm_id`, `itemid`, `label`, `valuenum`, `valueuom` |
| `hcpcsevents` | `chartdate` | `hcpcs_cd`, `short_description` |

---

## MIMIC-III (`mimic3.yaml`)

**Dataset class**: `MIMIC3Dataset`

| Table | `event.timestamp` comes from | Key attributes (partial) |
|-------|------------------------------|--------------------------|
| `patients` | `null` (NaT) | `gender`, `dob`, `dod` |
| `admissions` | `admittime` | `hadm_id`, `admission_type`, `ethnicity`, `hospital_expire_flag`, `dischtime` |
| `icustays` | `intime` | `hadm_id`, `first_careunit`, `last_careunit`, `outtime` |
| `diagnoses_icd` | `dischtime` | `hadm_id`, `icd9_code` |
| `procedures_icd` | `dischtime` | `hadm_id`, `icd9_code` |
| `prescriptions` | `startdate` | `hadm_id`, `drug`, `route`, `dose_val_rx` |
| `labevents` | `charttime` | `hadm_id`, `itemid`, `valuenum`, `valueuom` |
| `noteevents` | `null` (NaT) | `hadm_id`, `category`, `text` |

---

## eICU (`eicu.yaml`)

**Dataset class**: `eICUDataset`

| Table | `event.timestamp` comes from | Key attributes (partial) |
|-------|------------------------------|--------------------------|
| `patient` | `null` (NaT) | `patienthealthsystemstayid`, `unittype`, `hospitaldischargestatus` |
| `diagnosis` | `null` (NaT) | `patienthealthsystemstayid`, `diagnosisstring`, `icd9code` |
| `medication` | `null` (NaT) | `patienthealthsystemstayid`, `drugname`, `dosage`, `routeadmin` |
| `treatment` | `null` (NaT) | `patienthealthsystemstayid`, `treatmentstring` |
| `lab` | `null` (NaT) | `patienthealthsystemstayid`, `labname`, `labresult` |
| `physicalexam` | `null` (NaT) | `patienthealthsystemstayid`, `physicalexampath`, `physicalexamvalue` |
| `admissiondx` | `null` (NaT) | `patienthealthsystemstayid`, `admitdxpath` |

> **Note**: Most eICU tables have `timestamp: null`. Use attribute columns for
> relative time offsets (e.g. `unitdischargeoffset`) if time-ordering is needed.

---

## OMOP (`omop.yaml`)

**Dataset class**: `OMOPDataset`

| Table | `event.timestamp` comes from | Key attributes (partial) |
|-------|------------------------------|--------------------------|
| `person` | `null` (NaT) | `gender_concept_id`, `year_of_birth`, `race_concept_id` |
| `visit_occurrence` | `visit_start_datetime` | `visit_concept_id`, `visit_end_datetime`, `discharge_to_concept_id` |
| `death` | `death_datetime` | `cause_concept_id` |
| `condition_occurrence` | `condition_start_datetime` | `visit_occurrence_id`, `condition_concept_id` |
| `procedure_occurrence` | `procedure_datetime` | `visit_occurrence_id`, `procedure_concept_id` |
| `drug_exposure` | `drug_exposure_start_datetime` | `visit_occurrence_id`, `drug_concept_id`, `quantity` |
| `measurement` | `measurement_datetime` | `visit_occurrence_id`, `measurement_concept_id`, `value_as_number` |

---

## Filtering events by visit key

When filtering events to a specific visit, use the shared key column as an
attribute (e.g. `hadm_id` in MIMIC, `visit_occurrence_id` in OMOP):

```python
# MIMIC-IV/III
diag_events = patient.get_events(
    event_type="diagnoses_icd",
    filters=[("hadm_id", "==", getattr(visit, "hadm_id"))],
)

# OMOP
cond_events = patient.get_events(
    event_type="condition_occurrence",
    filters=[("visit_occurrence_id", "==", getattr(visit, "visit_occurrence_id"))],
)
```

Note: `hadm_id` is an **attribute** column (listed under `attributes:` in the YAML),
so `getattr(visit, "hadm_id")` works correctly — unlike `admittime` which is the
`timestamp:` column and is only accessible as `visit.timestamp`.
