---
name: pyhealth-bring-your-own-data
description: Turn raw patient files (CSV, TSV, Parquet) into a PyHealth dataset — inventory the tables, map each to patient_id / timestamp / attributes, write the YAML table config, load it with BaseDataset, and validate. Use when the user's data is not one of PyHealth's built-in datasets.
---

# Bring your own data — raw files → PyHealth

Use this when the user has data that is **not** one of PyHealth's built-in
datasets: a folder of CSV exports, a warehouse dump, a lab-local EHR extract.

This is the hardest step for a newcomer and the one where silent failure is most
likely, so work through it out loud and show output at every stage.

**Who does what.** You write the YAML, not the user. Never ask them to supply a
config file, and never ask a question phrased in YAML — no "what should
`timestamp:` be for this table?". The user owns facts about their data that no
amount of inspection can recover:

- which column means "patient" when three plausible ones exist
- whether a date column is the event time or an administrative entry time
- which columns are clinically meaningful enough to keep
- whether the data is identifiable, and where it is allowed to be written

You own everything else: reading the files, inferring the obvious mappings,
proposing the rest as a plain-language table with recommended defaults, writing
the YAML, and running it. Ask about the ambiguities you actually hit — not the
full field list — and carry a default into every question so the user can say
"yes, all of that" and move on.

---

## What PyHealth needs

PyHealth does not care what your tables are called or how they are shaped. It
needs, for each table, an answer to three questions:

1. **Which column identifies the patient?** (`patient_id`)
2. **Which column says when this happened?** (`timestamp`) — or `null` for static
   tables like demographics.
3. **Which other columns should be kept?** (`attributes`)

Those answers go in a YAML file that you write. PyHealth reads it, melts every
table into one long event table keyed by patient and time, and hands back
`Patient` objects with their events in chronological order.

The canonical example to model the config on is
`pyhealth/datasets/configs/mimic4_ehr.yaml`. Read it before writing one.

Supported file formats: CSV and TSV (optionally `.gz`), and Parquet — a single
file, a glob, or a directory of shards. See `BaseDataset._scan_table` in
`pyhealth/datasets/base_dataset.py`.

---

## Step 1 — Inventory the files

Do this with code. Do not ask the user to describe their own schema from memory;
they will get it slightly wrong and you will debug the wrong thing for an hour.

```python
from pathlib import Path
import pandas as pd

root = Path("<DATA_ROOT>")
for f in sorted(root.rglob("*")):
    name = f.name.lower()
    if not name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet", ".pq")):
        continue
    if name.endswith((".parquet", ".pq")):
        df = pd.read_parquet(f)
    else:
        # A .tsv read with the default comma separator parses as one wide
        # column and the schema you print back is fiction.
        sep = "\t" if ".tsv" in name else ","
        df = pd.read_csv(f, sep=sep, nrows=200)
    print(f"\n=== {f.relative_to(root)} ===")
    print(f"columns: {list(df.columns)}")
    print(df.dtypes.to_string())
    print(df.head(3).to_string())
```

Then say back to the user, in plain language, what you found: "You have five
files. `patients.csv` looks like one row per person; `admissions.csv` and
`labs.csv` look like events. `labs.csv` has no patient column of its own — it
joins through `admissions` on `hadm_id`."

---

## Step 2 — Propose the three roles, get them confirmed

Fill this table in **yourself**, from the inventory, then show it to the user as
a proposal. Every cell is your best inference; the user's job is to correct it,
not to author it.

| File | patient_id column | timestamp column | keep as attributes | notes |
|---|---|---|---|---|
| patients.csv | `subject_id` | `null` (static) | gender, birth_year | one row per patient |
| admissions.csv | `subject_id` | `admittime` | hadm_id, dischtime, expired | one row per visit |
| diagnoses.csv | `subject_id` | — needs join → `admissions.dischtime` | hadm_id, icd_code | no time of its own |

Then ask only about what you could not settle by looking, in the user's
vocabulary rather than the config's, each with a default attached:

> "I'm treating `admittime` as when the admission event happened, and keeping
> `dischtime` as a regular field — that's the usual choice. Two things I can't
> tell from the data:
>
> 1. `labs.csv` has both `charttime` and `storetime`. I'd use `charttime`, since
>    that's when the sample was taken rather than when it was entered. Right?
> 2. `patients.csv` has 31 columns. I'd keep `gender` and `birth_year` and drop
>    the rest — anything else you want available to the model?
>
> Say 'go' and I'll take those defaults."

Do not enumerate every table and column back at the user for approval. Confirm
the decisions that change the science — what counts as the event time, what the
cohort key is, which fields are in scope — and infer the rest.

**The timestamp rule.** The column you name under `timestamp:` becomes
`event.timestamp` and is **not** available as a named attribute. If you write
`timestamp: "admittime"`, then `event.admittime` is `None` forever, and
`event.timestamp` is what you want. This trips up nearly everyone once.

**When a table has no time of its own** — very common for diagnosis and procedure
tables — pull one in with a `join`. In MIMIC-IV, `diagnoses_icd` has no timestamp,
so the config joins `admissions` on `hadm_id` and uses `dischtime`. That join
also encodes a real clinical fact: those codes were assigned at *discharge*, not
at admission, so they cannot be admission-time features. Say that out loud.

**Keep the join key as an attribute.** If you want to filter events down to one
visit later, the visit key (`hadm_id` or equivalent) must appear in
`attributes:`, or you will not be able to reach it.

---

## Step 3 — Write the YAML yourself

Once the table in Step 2 is confirmed, generate the config from it. The user
should never open a text editor in this step, and should never be handed a
template to fill in.

**Before writing the file, say where it goes and get an explicit yes.** It is a
new file in the user's project, and the data it points at may be identifiable:

> "I'll write `configs/my_ehr.yaml` in the repo — it holds column names and the
> dataset root path, no patient data. OK?"

If the answer is no, or the root path is itself sensitive, put it wherever they
say and carry that path forward. Show the finished YAML in the conversation
either way — it is the artifact they will have to reason about later, and a
config the user has never seen is one nobody can debug.

```yaml
version: "1.0"
tables:
  patients:
    file_path: "patients.csv"       # relative to the dataset root
    patient_id: "subject_id"
    timestamp: null                  # static table — no event time
    attributes:
      - "gender"
      - "birth_year"

  admissions:
    file_path: "admissions.csv"
    patient_id: "subject_id"
    timestamp: "admittime"           # ← becomes event.timestamp, NOT event.admittime
    attributes:
      - "hadm_id"                    # keep the join key so you can filter by visit
      - "dischtime"
      - "expired"

  diagnoses:
    file_path: "diagnoses.csv"
    patient_id: "subject_id"
    join:                            # this table has no time of its own
      - file_path: "admissions.csv"
        "on": "hadm_id"
        how: "inner"
        columns:
          - "dischtime"
    timestamp: "dischtime"           # brought in by the join above
    attributes:
      - "hadm_id"
      - "icd_code"
```

Config schema, authoritative: `pyhealth/datasets/configs/config.py`
(`DatasetConfig`, `TableConfig`, `JoinConfig`). Fields available on a table:

| Key | Required | Meaning |
|---|---|---|
| `file_path` | yes | path relative to `root` |
| `patient_id` | no | patient column; if `null`, the row index is used |
| `timestamp` | no | column name, or a **list** of columns concatenated in order |
| `timestamp_format` | no | strftime format for the (concatenated) timestamp |
| `attributes` | yes | columns to keep as `event.<name>` |
| `join` | no | list of `{file_path, on, how, columns}` |

Note `"on"` must be quoted in YAML — bare `on` parses as the boolean `true`.

Split date and time columns? Use the list form:
`timestamp: ["date_col", "time_col"]` with a matching `timestamp_format`.

---

## Step 4 — Load and validate

`BaseDataset` is directly usable — you do not need a subclass to get started:

```python
from pyhealth.datasets import BaseDataset

if __name__ == "__main__":            # required: PyHealth uses multiprocessing
    dataset = BaseDataset(
        root="<DATA_ROOT>",
        tables=["patients", "admissions", "diagnoses"],   # keys from your YAML
        dataset_name="MyEHR",
        config_path="my_ehr.yaml",
        cache_dir="~/.cache/pyhealth",
        dev=True,                     # start small; flip to False once it works
    )
    dataset.stats()                   # patient count + event count

    pid = dataset.unique_patient_ids[0]
    patient = dataset.get_patient(pid)
    for e in patient.get_events()[:20]:
        print(e.event_type, e.timestamp, e.attr_dict)
```

**Do not move on until you have shown the user:**
- the patient count and event count from `stats()`, and whether they are
  plausible given the raw row counts
- one real patient's events, printed, in time order
- the per-table event counts — a table contributing zero events is a broken
  mapping, not an empty table

Write the same `dataset_name` and `config_path` into the spec so later steps
reproduce this exactly.

Once it works with `dev=True`, re-run with `dev=False`. Only then do you have the
real cohort size.

---

## Common breakages

**Patient ids with inconsistent dtypes.** `subject_id` read as `int64` from one
file and `str` from another means the join silently produces zero rows. CSV
sources are coerced to strings; Parquet keeps its native schema — so a mixed
CSV/Parquet layout is the usual culprit. Cast at the source.

**Timestamps that will not parse.** Symptom: events exist but every timestamp is
null, and any time-ordered task returns no samples. Fix with an explicit
`timestamp_format`. Check for mixed formats within one column, and for timezone
suffixes on some rows and not others.

**A static table given a timestamp.** Demographics rows are not events. Set
`timestamp: null`; giving them a fake time pollutes the event ordering.

**A table with zero events after loading.** Almost always a wrong `file_path`, a
misspelled column in `attributes`, or a join that matched nothing. Check the join
first: `how: "inner"` drops every unmatched row without complaint.

**Forgetting the visit key in `attributes`.** You will discover this at task time,
when you cannot filter events to a visit. Add it now.

**Shared cache across incompatible configs.** The cache path includes a UUID
derived from root, tables, name, and dev flag — but *not* the YAML contents. If
you edit the YAML, delete the corresponding cache directory before re-running, or
you will keep loading the old parse.

**Real PHI.** If this is identifiable patient data, say so before writing
anything to a shared cache directory or a location that gets logged.

---

## When to graduate to a subclass

`BaseDataset` with a `config_path` is enough for most bring-your-own-data cases.
Write a subclass (see `pyhealth/datasets/mimic4.py`) when you want to ship the
config alongside the class, set a `default_task`, or add dataset-specific
convenience properties. It is not a prerequisite — do not let it block the first
working pipeline.
