---
name: pyhealth-define-a-task
description: Write a PyHealth BaseTask — cohort filtering, temporal structure (same-visit, next-visit, patient-cumulative), label computation, and the input_schema/output_schema that turn one Patient into sample dicts. Includes seven runnable example task classes. Use when defining what the model predicts and from what.
---

# Define a task

A task turns one `Patient` into zero or more sample dicts. This is where cohort, temporal window, and label live. The examples below each show:
- Task class definition with input_schema and output_schema
- Dataset initialization (MIMIC3Dataset or MIMIC4Dataset)
- set_task() call for sample generation
- Basic sample verification

Examples are organized by task type and granularity level (patient vs visit).

## Schema ↔ Sample Dict Quick Reference

Each entry in `input_schema` maps a feature name (key in the sample dict) to a processor type.
The table below shows what value type `__call__` must put in the sample dict for each processor.

| `input_schema` value | Sample dict value type | Missing sentinel | Empty inner list? |
|---|---|---|---|
| `"sequence"` | `List[str]` (1D) | `["<missing>"]` | — |
| `"multi_hot"` | `List[str]` (1D, unique) | `["<missing>"]` | — |
| `"tensor"` | `List[float]`, len == `len(selected_ids)` | `[-1.0] * len(selected_ids)` | — |
| `"timeseries"` | `(List[datetime], np.ndarray shape (N, F))` | drop sample entirely | — |
| `"nested_sequence"` | `List[List[str]]` (2D) | inner: `["<missing>"]` or `["<pad>"]` | **NEVER** `[]` → crash |
| `"nested_sequence_floats"` | `List[List[float]]` (2D) | inner: `[-1.0]` or `[-1.0]*len(ids)` | **NEVER** `[]` → crash |

```python
# Concrete example — task with every processor type
input_schema = {
    "conditions":   "sequence",         # → sample["conditions"]   = ["I10", "E11.9"]
    "demographics": "multi_hot",        # → sample["demographics"] = ["M", "White"]
    "labs":         "tensor",           # → sample["labs"]         = [1.2, -1.0, 138.0]
    "heart_rate":   "timeseries",       # → sample["heart_rate"]   = (times, np.ndarray)
    "cond_hist":    "nested_sequence",  # → sample["cond_hist"]    = [["I10"], ["E11"]]
    "labs_hist":    "nested_sequence_floats",    # → sample["labs_hist"]    = [[1.2, 4.1], [-1.0, -1.0]]
}
output_schema = {"label": "binary"}    # → sample["label"]        = 0 or 1
```

**Rules that apply across all processor types:**
- `input_schema` keys MUST exactly match the keys in every sample dict — no extra, no missing
- For `tensor`: use `-1.0` (not `0.0`) as the missing sentinel — `0.0` is reserved for collation padding
- For `nested_sequence` / `nested_sequence_floats`: the outer list length = number of visits in the history window; inner lists MUST have ≥ 1 element
- For `timeseries`: skip the sample entirely if there are no events — never emit an empty tuple

---

## Temporal Patterns

The loop structure in `__call__` depends on the temporal relationship between features and label. Read each example carefully — pay attention to which visit provides the features and which provides the label.

**Next-Visit Pattern** (`for i in range(len(visits) - 1)`):
- Features from visit `i`, label from visit `i+1`
- Requires at least 2 visits: `if len(visits) <= 1: return []`
- Lower leakage risk — prediction target is always in the future
- Examples: mortality_visit_mimic4.py, mortality_patient_mimic3.py

**Previous-Visit History Pattern** (`for i in range(1, len(visits))`):
- Features from visit `i` + history from visit `i-1`, label from visit `i`
- Requires at least 2 visits: `if len(visits) <= 1: return []`
- Safe when label is independent of history features (e.g. drug recommendation)
- Examples: drug_rec_patient_mimic3.py

**Cumulative Nested History Pattern** (two-pass: collect then accumulate):
- ⭐ Use when your pipeline spec has **`nested_sequence`** processor entries
- Each feature is a 2D list `[[v0_codes], [v1_codes], ..., [vN_codes]]` — ALL prior visits
- Pass 1: collect per-visit flat code lists; Pass 2: accumulate into 2D nested lists
- Apply leakage guard if history field overlaps with label field (e.g., `drugs_hist[i] = []`)
- Requires at least 2 valid visits; `input_schema` MUST use `"nested_sequence"` (not `"sequence"`)
- Examples: drug_rec_nested_mimic4.py ← see for complete reference implementation

**Visit-Level Pattern** (`for visit in visits`):
- Features and label both from the same visit
- Higher leakage risk — only safe when label is structurally independent of features (e.g. length of stay from admit/discharge timestamps, or label is computed before any treatment features)
- Examples: los_visit_mimic4.py

---

## Mortality Prediction

### Patient-Level
- **File**: [mortality_prediction/patient/mortality_patient_mimic3.py](examples/mortality_prediction/patient/mortality_patient_mimic3.py)
- **Dataset**: MIMIC-III
- **Description**: Predict mortality at next visit based on current visit data
- **Pattern**: Requires multiple visits, iterates over visit pairs (i, i+1)
- **Label Source**: Next visit's `hospital_expire_flag`
- **Features**: Current visit's diagnoses, procedures, prescriptions

### Visit-Level (Next-Visit Pattern)
- **File**: [mortality_prediction/visit/mortality_visit_mimic4.py](examples/mortality_prediction/visit/mortality_visit_mimic4.py)
- **Dataset**: MIMIC-IV
- **Description**: Predict mortality in the NEXT visit using current visit features
- **Pattern**: Iterates over visit pairs (i, i+1) — requires ≥2 visits per patient
- **Label Source**: NEXT visit's `hospital_expire_flag`
- **Features**: Current visit's diagnoses, procedures, prescriptions

- **File**: [mortality_prediction/visit/mortality_visit_mimic4_with_labs.py](examples/mortality_prediction/visit/mortality_visit_mimic4_with_labs.py)
- **Dataset**: MIMIC-IV
- **Description**: Predict mortality in the NEXT visit using codes + lab values
- **Pattern**: Iterates over visit pairs (i, i+1) with semantic features (filtered lab itemids)
- **Label Source**: NEXT visit's `hospital_expire_flag`
- **Features**: Current visit's codes + aggregated lab values (tensor)
- **Special**: Demonstrates lab filtering with `SELECTED_LAB_ITEMS`

- **File**: [mortality_prediction/visit/mortality_visit_mimic4_with_medcode.py](examples/mortality_prediction/visit/mortality_visit_mimic4_with_medcode.py)
- **Dataset**: MIMIC-IV
- **Description**: Predict mortality in the NEXT visit using codes normalized via `CrossMap`
- **Pattern**: Iterates over visit pairs (i, i+1) — requires ≥2 visits per patient
- **Label Source**: NEXT visit's `hospital_expire_flag`
- **Features**: Current visit's codes converted to higher-level categories
- **Special**: Demonstrates `CrossMap` for code normalization — reduces vocabulary and groups clinically related codes:
  - ICD-9/10-CM → CCS-CM  (~15k diagnosis codes → ~285 categories)
  - ICD-9/10-PCS → CCS-PROC (~4k procedure codes → ~231 categories)
  - NDC → ATC level-3 (~100k drug codes → ~800 therapeutic subgroups)
  - Routes ICD-9 vs ICD-10 to separate CrossMaps using `icd_version` field
  - Unmappable codes are silently dropped via `_safe_map()` helper

---

## Drug Recommendation

### Visit-Level (Cumulative Nested History Pattern) — ⭐ USE THIS FOR `nested_sequence` TASKS
- **File**: [drug_recommendation/visit/drug_rec_nested_mimic4.py](examples/drug_recommendation/visit/drug_rec_nested_mimic4.py)
- **Dataset**: MIMIC-IV
- **Description**: Recommend drugs using the **full cumulative history** across ALL prior visits.
  Each feature field (`conditions`, `procedures`, `drugs_hist`) is a 2D nested list
  `[[v0_codes], [v1_codes], ..., [vN_codes]]` consumed by the `nested_sequence` processor.
- **Pattern**: Two-pass — collect per-visit flat lists, then accumulate into 2D nested lists
- **Label Source**: Current visit's prescriptions (multilabel)
- **Features**: `nested_sequence` for conditions, procedures, drugs_hist (prior visit history)
- **Special**:
  - `input_schema` uses `"nested_sequence"` (NOT `"sequence"`)
  - Historical accumulation loop builds 2D list across visits
  - Leakage guard: `raw[i]["drugs_hist"][i] = ["<pad>"]` blanks current-visit's history slot
    ⚠️ Use `["<pad>"]` **not** `[]` — empty inner lists cause `pack_padded_sequence` to crash
    with `"length <= 0"` at training time. The `nested_sequence` processor treats `["<pad>"]` as
    a single-token placeholder that gets masked out during batching.
  - **`nested_sequence_floats` (e.g., `labs_summary`)**: inner float lists also MUST NOT be empty.
    Use `[-1.0]` (or `[-1.0] * len(selected_ids)`) as sentinel when no data is available.
    Same `pack_padded_sequence` crash as empty `nested_sequence` inner lists.
    If `selected_ids` is empty, omit the feature from `input_schema` entirely.
  - **Model policy**: if no model is explicitly requested, default to
    `MultimodalRNN`. If a model is explicitly requested and incompatible, first
    prefers a compatible multimodal sibling model (e.g., `RNN`→`MultimodalRNN`) and only
    rewrites `task.py` if no compatible multimodal sibling exists.
  - Patients with < 2 valid visits return `[]`
  - **History-windowing variant**: after accumulation, slice to the last N visits:
    ```python
    WINDOW = 3
    for i in range(len(raw)):
        raw[i]["conditions"] = raw[i]["conditions"][-WINDOW:]
        raw[i]["procedures"] = raw[i]["procedures"][-WINDOW:]
        raw[i]["drugs_hist"] = raw[i]["drugs_hist"][-WINDOW:]
    # Then apply leakage guard on the last slot (not index i):
    for i in range(len(raw)):
        raw[i]["drugs_hist"][-1] = ["<pad>"]  # ⚠️ ["<pad>"] not [] — empty inner lists crash pack_padded_sequence
    ```
    The processor still receives a valid 2D list — just shorter. No schema changes needed.
- **Canonical PyHealth source**: `pyhealth/tasks/drug_recommendation.py`
  → `DrugRecommendationMIMIC4` (built-in; also available as `from pyhealth.tasks import DrugRecommendationMIMIC4`)

### Visit-Level (Single Previous-Visit Pattern)
- **File**: [drug_recommendation/visit/drug_rec_patient_mimic3.py](examples/drug_recommendation/visit/drug_rec_patient_mimic3.py)
- **Dataset**: MIMIC-III
- **Description**: Recommend drugs using only the *immediately preceding* visit's drug history
  (simpler than cumulative nested; uses flat `sequence` processor, not `nested_sequence`)
- **Pattern**: One sample per visit starting from visit 1 (`range(1, len(visits))`), requires ≥2 visits
- **Label Source**: Current visit's prescriptions (multilabel)
- **Features**: Current visit's conditions/procedures + **previous single visit's** drugs (`drugs_history`)
- **Special**: Cross-visit feature — drug history comes from visit `i-1` only (not all history)

---

## Length of Stay

### Visit-Level
- **File**: [length_of_stay/visit/los_visit_mimic4.py](examples/length_of_stay/visit/los_visit_mimic4.py)
- **Dataset**: MIMIC-IV
- **Description**: Predict length of stay category for each visit
- **Pattern**: Visit-level, each visit independent (`for visit in visits`)
- **Label Source**: Computed from same visit's `admittime` and `dischtime` using `categorize_los()`
- **Features**: Same visit's diagnoses and procedures
- **Label Scheme**: **MUST use the canonical 10-category scheme** — copy `categorize_los()` from the example verbatim, do NOT invent custom buckets:
  - `0`: < 1 day
  - `1–7`: one bucket per day of the first week
  - `8`: > 7 days and ≤ 14 days
  - `9`: > 14 days

---

## Optimization (only after a baseline works)

| Topic | Guide | What it covers |
|-------|-------|----------------|
| Task optimization | [task-engineering.md](../optimize-a-pipeline/SKILL.md) | Exploration strategy, axes decision table, full experiment template |
| Hyperparameter tuning | [tuning.md](../optimize-a-pipeline/references/hyperparameter-tuning.md) | Config sweep template, tuning axes, winner selection |

---

## Usage Tips

1. **Read the example code, not just the description**: The loop structure and which visit provides the label is what matters — copy it faithfully.

2. **Adapt the pattern**:
   - Read the closest matching example
   - Copy the `__call__` structure including the loop
   - Modify for your specific features (from your pipeline spec)
   - Adjust label computation logic

3. **Key implementation details**:
   - Next-visit pattern: Always check `len(visits) > 1` before iterating
   - Use `filters=[("hadm_id", "==", visit.hadm_id)]` to get visit-specific events
   - Handle missing data appropriately (skip samples or use defaults)

4. **Using CrossMap for code normalization** (`mortality_visit_mimic4_with_medcode.py`):
   - Import: `from pyhealth.medcode import CrossMap`
   - Load once in `__init__`: `self.cm = CrossMap.load("ICD9CM", "CCSCM")`
   - Map a code: `self.cm.map("428.0")` → `["108"]` (always returns a list)
   - NDC→ATC with level: `self.cm.map(ndc, target_kwargs={"level": 3})`
   - Maps are **cached to disk** after first download — subsequent runs are fast
   - MIMIC-IV has mixed ICD-9/ICD-10; check `event.icd_version` and route to the matching CrossMap
   - Available conversions: `ICD9CM→CCSCM`, `ICD10CM→CCSCM`, `ICD9PROC→CCSPROC`, `ICD10PROC→CCSPROC`, `NDC→ATC`, `NDC→RxNorm`

5. **Feature selection variant** (reduce input_schema features):
   - The simplest variant is to remove one or more feature keys from `input_schema` and the
     corresponding logic in `__call__`. For example, if the base task has:
     ```python
     input_schema = {"conditions": "sequence", "procedures": "sequence", "labs": "tensor"}
     ```
     a `feature_drop` variant could omit `labs` entirely:
     ```python
     input_schema = {"conditions": "sequence", "procedures": "sequence"}
     ```
     and simply not populate `labs` in the sample dict. This tests whether a feature adds
     signal or is just noise.
   - You can also keep the same schema but restrict `selected_ids` from your pipeline spec
     to a smaller high-confidence subset (e.g., only the top-K most frequent codes).

6. **Cache-safety when rewriting task logic (after a baseline works)**:
   - PyHealth task caches are keyed by task identity (`task_name`) plus schema.
   - If you rewrite task logic but keep the same `input_schema` and `output_schema`
     (e.g., ICD truncation, code filtering, history windowing, leakage-guard fixes),
     you MUST assign a NEW unique `task_name`.
   - Never reuse the previous `task_name` after a rewrite with unchanged schemas,
     or stale cached task outputs may be reused.
   - Practical pattern: append a short revision/hash suffix such as `_v2`, `_v3`,
     or `_rev_<hash>`.

---

## Canonical PyHealth Task Reference Files

When designing a `simplified` variant, read the built-in PyHealth implementation directly:

| Task | File | Key Class |
|------|------|-----------|
| Mortality (MIMIC-IV) | `pyhealth/tasks/mortality_prediction.py` | `MortalityPredictionMIMIC4` |
| Drug Recommendation (MIMIC-IV) | `pyhealth/tasks/drug_recommendation.py` | `DrugRecommendationMIMIC4` |
| Length of Stay (MIMIC-IV) | `pyhealth/tasks/length_of_stay_prediction.py` | `LengthOfStayPredictionMIMIC4` |

These canonical implementations are often simpler than agent-generated tasks — they skip lab
processing, use direct attribute access, and avoid over-engineering. Using one as
a variant tests whether the custom task actually outperforms the stock implementation.

---

## ICD Code Abstraction / Truncation (icd_trunc variant)

Reducing ICD code specificity groups clinically related conditions, shrinks vocabulary, and may
improve generalization. Two approaches:

### String Truncation (simple, no extra imports)
```python
# ICD-9 or ICD-10: drop decimal suffix (e.g., "428.0" → "428", "I10.9" → "I10")
trunc_code = code.split(".")[0] if "." in code else code

# ATC drug codes: first 3 chars = therapeutic subgroup (e.g., "A11CA" → "A11")
trunc_drug = drug[:3] if len(drug) >= 3 else drug
```

### Hierarchy-Aware via InnerMap
```python
from pyhealth.medcode import InnerMap

# Load once (cached to disk after first call)
icd9cm = InnerMap.load("ICD9CM")

def get_parent_icd9(code):
    try:
        ancestors = icd9cm.get_ancestors(code)
        return ancestors[0] if ancestors else code  # closest ancestor = immediate parent
    except Exception:
        return code  # fallback for codes absent from the graph
```

**InnerMap API** (`pyhealth/medcode/inner_map.py`):
- `InnerMap.load(vocabulary)` → loads code system (`ICD9CM`, `ICD10CM`, `ICD9PROC`, etc.)
- `.get_ancestors(code)` → list ordered closest → farthest (index 0 = immediate parent)
- `.get_descendants(code)` → list of child codes
- `.lookup(code, attribute="name")` → human-readable name
- Use `try/except` — codes absent from the graph raise `KeyError`
- MIMIC-IV has mixed ICD-9/ICD-10; check `event.icd_version` and route to `ICD9CM` or `ICD10CM`
