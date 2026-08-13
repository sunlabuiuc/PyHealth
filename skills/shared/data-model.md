# PyHealth's data model

Every dataset, task, and processor speaks this vocabulary. Read it once; the guides assume it.

Source of truth: `pyhealth/data/data.py`, `pyhealth/datasets/base_dataset.py`,
`pyhealth/datasets/sample_dataset.py`.

---

## The chain

```
raw files
   │  YAML table config (file_path / patient_id / timestamp / attributes / join)
   ▼
global event table          one long table: patient_id, event_type, timestamp, <type>/<attr>...
   │  partition by patient
   ▼
Patient                     all of one person's events, time-sorted
   │  task.__call__(patient)
   ▼
sample dicts                one per training example
   │  processors fit + transform  (input_schema / output_schema)
   ▼
SampleDataset               tensors, ready to split and load
```

Stages 1–4 all happen inside `dataset.set_task(task)`. That one call parses (or loads cached)
raw files, runs your task over every patient, fits the processors, and writes the result to
cache.

---

## `Event`

One timestamped row from one source table (`pyhealth/data/data.py:12`).

```python
event.event_type   # str — the table name, ALWAYS lowercase
event.timestamp    # datetime, or NaT for static tables
event.attr_dict    # dict of everything listed under `attributes:` in the YAML
event.icd_code     # attribute access proxies into attr_dict (data.py:98)
```

Two things that trip people up:

- **The `timestamp:` column is not an attribute.** If the YAML says `timestamp: "admittime"`,
  then `event.admittime` is `None` forever and `event.timestamp` is the value. There is no
  warning.
- **`event_type` is always the lowercase config key**, even when you passed the table name in
  uppercase to the constructor. See the case rule in
  [use-a-dataset](../guides/use-a-dataset/SKILL.md).

## `Patient`

All of one person's events, sorted by timestamp, partitioned by event type for O(1) type lookup
(`data.py:126`). This is what your task's `__call__` receives.

```python
patient.patient_id
patient.get_events(event_type=None, start=None, end=None, filters=None, return_df=False)
```

`get_events` (`data.py:173`) applies, in order: event-type filter → time-range filter (binary
search on the sorted timestamps) → attribute filters.

**Attribute filters** are `[(attr, op, value), ...]`, AND-ed together, with `op` one of
`== != < <= > >=`:

```python
patient.get_events(
    event_type="diagnoses_icd",
    filters=[("hadm_id", "==", visit.hadm_id)],
)
```

Two rules the source enforces (`data.py:206`, `data.py:216`):

- `filters` requires `event_type` — it asserts otherwise, because the filter column is built as
  `f"{event_type}/{attr}"`.
- That prefixing is also why an attribute you want to filter on **must** be listed under
  `attributes:` in the YAML. If it isn't, the column doesn't exist and polars raises
  `ColumnNotFoundError`.

`return_df=True` gives you the raw polars frame with `"<event_type>/<attr>"` columns instead of
`Event` objects — much faster when you are pulling one column across many rows.

There is **no** `patient.visits`. Visits are just events of the admissions table; get them with
`get_events(event_type="admissions")` and key off `hadm_id`.

## Sample dict

What `task.__call__` returns — a list of them, one per training example. Flat `dict` containing:

- `patient_id`
- a visit or record id (`visit_id`, `hadm_id`, `record_id` — whatever identifies the row)
- exactly one key per entry in `input_schema`
- exactly one key per entry in `output_schema`

No extra keys, no missing keys. The schemas and the dict must match exactly.

Return `[]` for a patient who does not qualify (too few visits, missing required data). That is
the normal way to express cohort exclusion.

## `input_schema` / `output_schema`

Class attributes on the task mapping each sample-dict key to a **processor name** from
`PROCESSOR_REGISTRY`. See [choose-processors](../guides/choose-processors/SKILL.md) for all 24.

```python
class MyTask(BaseTask):
    task_name: str = "MyTask"
    input_schema  = {"conditions": "sequence", "labs": "tensor"}
    output_schema = {"label": "binary"}
```

## `SampleDataset`

The output of `set_task` (`pyhealth/datasets/sample_dataset.py:255`) — a
`litdata.StreamingDataset` holding the processed samples plus the fitted processors. It is what
you split, load, and hand to a model; models read their input dimensions from it, which is why
`RNN(dataset=samples, ...)` needs the sample dataset rather than a raw config.

Because it streams, it manages its own ordering: use `get_dataloader(...)`, never a raw
`DataLoader(..., shuffle=True)`.

## Caching

`set_task` caches on `task_name` **plus** the schemas. Two consequences worth memorizing:

- Rewrite task logic while keeping the same `task_name` and schemas → you silently get the old
  samples back. Bump `task_name` (`_v2`).
- Change only hyperparameters → nothing about the samples changed, so leave `task_name` alone;
  mutating it forces a needless full rebuild.

The dataset-level cache path also embeds a UUID over root, table list, dataset name, and the
`dev` flag — but **not** the YAML contents. Edit the YAML, delete that cache directory.

## The `__main__` guard

PyHealth uses multiprocessing. Task classes must be defined at **module level** so they pickle;
dataset construction and training must run under `if __name__ == "__main__":`. A task class
defined inside a function or under the guard kills the workers with an opaque error.
