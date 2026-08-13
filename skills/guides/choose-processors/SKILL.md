---
name: pyhealth-choose-processors
description: Choose the PyHealth processor for each feature and label — the 24 registered names, what value type each expects in the sample dict, the missing-data sentinels, and the empty-list rules that crash training. Use when filling in input_schema or output_schema, or when debugging a processor error.
---

# Choose processors

Processors turn a sample-dict value into a tensor, learning any vocabulary or statistics they need from the training data. Choosing a processor is choosing a representation.

Every string in a task's `input_schema` and `output_schema` is a processor name looked up in
`PROCESSOR_REGISTRY` (`pyhealth/processors/__init__.py:1`). A name that is not registered raises
at `set_task` — **do not guess names**. Your `__call__` must emit exactly the value type the
chosen processor expects.

`input_schema` and `output_schema` draw from the *same* registry. The four label processors
below are simply the ones that make sense on the output side.

---

## The full registry — all 24 names

Generated from the `@register_processor` decorators; this is the complete set.

### Feature processors — categorical

| Name | Class | Source | Sample-dict value |
|---|---|---|---|
| `sequence` | `SequenceProcessor` | `sequence_processor.py` | `List[str]` — ordered codes |
| `multi_hot` | `MultiHotProcessor` | `multi_hot_processor.py` | `List[str]` — unordered unique set |
| `nested_sequence` | `NestedSequenceProcessor` | `nested_sequence_processor.py` | `List[List[str]]` — one inner list per visit |
| `deep_nested_sequence` | `DeepNestedSequenceProcessor` | `deep_nested_sequence_processor.py` | `List[List[List[str]]]` — one more level of nesting |

### Feature processors — numeric

| Name | Class | Source | Sample-dict value |
|---|---|---|---|
| `tensor` | `TensorProcessor` | `tensor_processor.py` | `List[float]`, fixed length |
| `nested_sequence_floats` | `NestedFloatsProcessor` | `nested_sequence_processor.py:189` | `List[List[float]]` |
| `deep_nested_sequence_floats` | `DeepNestedFloatsProcessor` | `deep_nested_sequence_processor.py` | `List[List[List[float]]]` |
| `timeseries` | `TimeseriesProcessor` | `timeseries_processor.py` | `(times, np.ndarray (N, F))` |
| `temporal_timeseries` | `TemporalTimeseriesProcessor` | `temporal_timeseries_processor.py` | temporal variant emitting a dict |

⚠️ The floats processor is named **`nested_sequence_floats`**, not `nested_floats`. The class is
`NestedFloatsProcessor` but the *registered name* is what goes in the schema.

### Feature processors — other modalities

| Name | Class | Source | Sample-dict value |
|---|---|---|---|
| `text` | `TextProcessor` | `text_processor.py` | `str` — clinical notes, reports |
| `tuple_time_text` | `TupleTimeTextProcessor` | `tuple_time_text_processor.py` | `(times, texts)` — timestamped notes |
| `image` | `ImageProcessor` | `image_processor.py` | image path or array |
| `time_image` | `TimeImageProcessor` | `time_image_processor.py` | `(times, images)` |
| `audio` | `AudioProcessor` | `audio_processor.py` | audio path or waveform |
| `graph` | `GraphProcessor` | `graph_processor.py` | graph structure |

### Feature processors — model-specific and pass-through

| Name | Class | Source | Use when |
|---|---|---|---|
| `stagenet` | `StageNetProcessor` | `stagenet_processor.py` | required by `StageNet` / `StageAttentionNet` |
| `stagenet_tensor` | `StageNetTensorProcessor` | `stagenet_processor.py` | numeric input to the same models |
| `cehr` | `CehrProcessor` | `cehr_processor.py` | CEHR-style tokenized patient sequences |
| `raw` | `RawProcessor` | `raw_processor.py` | pass the value through untouched |
| `ignore` | `IgnoreProcessor` | `ignore_processor.py` | keep a field in the sample dict but feed nothing to the model (ids, metadata) |

### Label processors — for `output_schema`

| Name | Class | Sample-dict value | Metrics fn |
|---|---|---|---|
| `binary` | `BinaryLabelProcessor` | `0` or `1` | `binary_metrics_fn` |
| `multiclass` | `MultiClassLabelProcessor` | one class index / label | `multiclass_metrics_fn` |
| `multilabel` | `MultiLabelProcessor` | `List[str]` — the set of positive labels | `multilabel_metrics_fn` |
| `regression` | `RegressionLabelProcessor` | `float` | `regression_metrics_fn` |

All four live in `pyhealth/processors/label_processor.py`. The choice here determines the
model's loss **and** which metrics you get — see
[metrics-and-reporting.md](../train-and-evaluate/references/metrics-and-reporting.md).

---

## The six you will actually use most

Full detail on formats, sentinels, and crash modes.

### 1. `sequence` (SequenceProcessor)
**Implementation:** `pyhealth/processors/sequence_processor.py`
**Used for:** Sequential categorical events (e.g., diagnosis codes, procedure codes).
**Output Format:** A contiguous 1D list of strings.
**Requirements:**
- MUST NOT be empty. If missing, return `["<missing>"]`.

```python
# input_schema declaration
input_schema = {"conditions": "sequence"}

# sample dict value — 1D list of strings
sample["conditions"] = ["I10", "E11.9", "Z87.39"]   # normal
sample["conditions"] = ["<missing>"]                  # no events
```

### 2. `multi_hot` (MultiHotProcessor)
**Implementation:** `pyhealth/processors/multi_hot_processor.py`
**Used for:** Unordered sets of categorical events or static demographics (e.g., gender, race, set of medications without specific timestamps).
**Output Format:** A 1D list of unique strings.
**Requirements:**
- MUST NOT be empty. If missing, return `["<missing>"]`.
- Order does not matter, but typically you should remove duplicates.

```python
# input_schema declaration
input_schema = {"demographics": "multi_hot"}

# sample dict value — 1D list of unique strings (order irrelevant)
sample["demographics"] = ["M", "White", "65-74"]   # normal
sample["demographics"] = ["<missing>"]               # no data
```

### 3. `tensor` (TensorProcessor)
**Implementation:** `pyhealth/processors/tensor_processor.py`
**Used for:** A fixed-size array of numerical values (e.g., specific lab values at admission).
**Output Format:** A 1D list of floats.
**Requirements:**
- MUST match the exact length of `selected_ids` specified in the config.
- If missing, return `[-1.0] * len(selected_ids)` (use -1.0, NOT 0.0 — 0.0 is reserved for collation padding).

```python
# input_schema declaration
input_schema = {"labs": "tensor"}

# selected_ids come from your pipeline spec (fixed order — do not change)
selected_ids = ["50912", "50971", "50983"]  # e.g. creatinine, potassium, sodium

# sample dict value — 1D list of floats, length == len(selected_ids)
sample["labs"] = [1.2, 4.1, 138.0]              # normal — one float per selected_id slot
sample["labs"] = [-1.0, -1.0, -1.0]             # all missing (-1.0 sentinel, NOT 0.0)
sample["labs"] = [1.2, -1.0, 138.0]             # partial — -1.0 for unmeasured slots
```

### 4. `timeseries` (TimeseriesProcessor)
**Implementation:** `pyhealth/processors/timeseries_processor.py`
**Used for:** Numerical values attached to specific timestamps (e.g., longitudinal lab results throughout a visit).
**Output Format:** A tuple of two lists/arrays: `(times, values)`
- `times`: List of numeric or time objects `[t1, t2, ...]`.
- `values`: 2D `np.ndarray` of shape `(num_events, num_features)`. Typically `(N, 1)` for a single feature tracked over time.
**CRITICAL REQUIREMENTS (DO NOT IGNORE):**
- **Cannot be empty!** If there are no data points, `TimeseriesProcessor` will crash with `ValueError: Timestamps list is empty.`
- If a patient or visit has no timeseries data points for a required feature, you **MUST skip/discard that sample entirely** during your `Task.__call__` logic. Do not emit empty timeseries.

```python
# input_schema declaration
input_schema = {"heart_rate": "timeseries"}

# sample dict value — (list_of_timestamps, 2D_ndarray)
# times: list of datetime objects (always use e.timestamp, never getattr(e, "charttime"))
# values: shape (N, 1) for single feature, (N, F) for F features measured together
import numpy as np
times  = [datetime(2020,1,1,8,0), datetime(2020,1,1,12,0), datetime(2020,1,1,16,0)]
values = np.array([[72.0], [75.0], [68.0]])   # shape (3, 1)
sample["heart_rate"] = (times, values)

# ❌ NEVER emit empty — drop the sample instead:
# sample["heart_rate"] = ([], np.zeros((0, 1)))  → ValueError: Timestamps list is empty
```

### 5. `nested_sequence` (NestedSequenceProcessor)
**Implementation:** `pyhealth/processors/nested_sequence_processor.py`
**Used for:** Sequences of sequences (e.g., a patient has multiple visits, and each visit has a sequence of diagnoses).
**Output Format:** A 2D list of strings `[[visit1_codes], [visit2_codes], ...]`.
**Requirements:**
- Inner lists MUST NOT be empty. If a visit has no events, use `["<missing>"]`; for leakage-guard slots use `["<pad>"]`.

```python
# input_schema declaration
input_schema = {"conditions": "nested_sequence"}

# sample dict value — 2D list of strings, one inner list per visit in history
sample["conditions"] = [["I10", "E11"], ["I10"], ["Z87.39", "I10", "N18"]]  # 3-visit history
sample["conditions"] = [["<missing>"]]       # single visit, no events
sample["conditions"] = [["I10"], ["<pad>"]]  # leakage guard: current-visit slot blanked

# ❌ NEVER use empty inner list:
# sample["conditions"] = [["I10"], []]  → RuntimeError: Length of all samples has to be > 0
```

### 6. `nested_sequence_floats` (NestedFloatsProcessor)
**Implementation:** `pyhealth/processors/nested_sequence_processor.py`
**Used for:** Sequences of float vectors (e.g., multiple visits, each with numerical lab values).
**Output Format:** A 2D list of floats `[[f0, f1, ...], [f0, f1, ...], ...]` — one inner list per visit.
**CRITICAL REQUIREMENTS:**
- Inner float lists MUST NOT be empty. If no data is available for a visit, use `[-1.0]` (or `[-1.0] * len(selected_ids)` to match the expected vector length).
- **Empty inner lists cause `pack_padded_sequence` to crash** with `"Length of all samples has to be greater than 0"` at training time — same failure mode as empty inner lists in `nested_sequence`.
- If `selected_ids` is empty (no features configured for this field), omit the feature from `input_schema` entirely rather than emitting `[[], [], ...]`.

```python
# input_schema declaration
input_schema = {"labs_summary": "nested_sequence_floats"}

# selected_ids come from your pipeline spec (e.g. 3 lab item IDs → inner lists of length 3)
selected_ids = ["50912", "50971", "50983"]

# sample dict value — 2D list of floats, one inner list per visit in history
sample["labs_summary"] = [[1.2, 4.1, 138.0], [1.5, -1.0, 140.0]]  # 2-visit history
sample["labs_summary"] = [[-1.0, -1.0, -1.0]]                       # single visit, all missing
sample["labs_summary"] = [[-1.0]]                                    # scalar sentinel if len(selected_ids)==1

# ❌ NEVER use empty inner list:
# sample["labs_summary"] = [[1.2, 4.1], []]  → RuntimeError: Length of all samples has to be > 0
# ❌ NEVER emit if selected_ids is empty — omit the feature from input_schema entirely
```

## Common Crash Fixes
- `ValueError: Timestamps list is empty.` -> You returned `([], np.zeros((0, 1)))` for a timeseries. You MUST NOT emit empty timeseries; drop the sample instead.
- `KeyError: '<missing>'` -> Usually acceptable as the tokenizer will handle it, but make sure to use exact string `"<missing>"` for empty sequences/multi_hot.
- `RuntimeError: Length of all samples has to be greater than 0` → An empty inner list exists in a `nested_sequence` or `nested_sequence_floats` feature. For `nested_sequence` use `["<pad>"]`; for `nested_sequence_floats` use `[-1.0]` (or `[-1.0] * len(selected_ids)`). Never emit `[]` as an inner list for either processor type.
