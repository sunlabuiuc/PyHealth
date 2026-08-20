---
name: pyhealth-map-medical-codes
description: Work with clinical vocabularies via pyhealth.medcode — look up code names, walk ICD/ATC hierarchies with InnerMap, and translate between systems with CrossMap (ICD→CCS, NDC→ATC, NDC→RxNorm). Use when codes need normalizing, grouping, or their vocabulary is too large, and when reducing a 15k-code vocabulary to a few hundred clinically meaningful groups.
---

# Map medical codes

Raw clinical vocabularies are enormous and sparse. ICD-9-CM has ~15,000 diagnosis codes; NDC has
~100,000 drug product codes. A model that sees each as a distinct token spends most of its
capacity on codes it encountered twice.

Mapping to a coarser vocabulary — ICD → CCS (~285 groups), NDC → ATC level 3 (~800) — is
routinely the single highest-value preprocessing step in an EHR pipeline. It shrinks the
vocabulary, groups clinically related codes, and improves generalization to codes the model
never saw.

Everything here comes from `pyhealth.medcode`. Maps download once and cache to disk.

---

## Two objects

**`InnerMap`** — *within* one vocabulary. Lookups, hierarchy walking.
**`CrossMap`** — *between* two vocabularies. Translation.

### `InnerMap` (`pyhealth/medcode/inner_map.py:17`)

```python
from pyhealth.medcode import InnerMap

icd9cm = InnerMap.load("ICD9CM")

icd9cm.lookup("428.0")            # human-readable name
icd9cm.get_ancestors("428.0")     # closest → farthest; [0] is the immediate parent
icd9cm.get_descendants("428")     # child codes
"428.0" in icd9cm                 # membership test
icd9cm.available_attributes()     # what lookup() can ask for
icd9cm.stat()                     # size of the vocabulary
```

Ten graph-backed vocabularies, each a subclass in `pyhealth/medcode/codes/`: `ICD9CM`,
`ICD10CM`, `ICD9PROC`, `ICD10PROC`, `CCSCM`, `CCSPROC`, `ATC`, `NDC`, `RxNorm`, `UMLS`.

Eight further grouper vocabularies — `CCSR`, `CCI`, `CCIR`, `ICD9CHAPTER`, `ICD10CHAPTER`,
`ICD10BLOCK`, `CCC`, `CCCSUB` — are `FlatMap` subclasses in
`pyhealth/medcode/codes/icd_groupers.py`. They are label systems with no ontology, so they
support `CrossMap` but not `lookup`/`get_ancestors`.

Codes absent from the graph raise `KeyError` — wrap every lookup:

```python
def parent_or_self(code):
    try:
        anc = icd9cm.get_ancestors(code)
        return anc[0] if anc else code
    except KeyError:
        return code
```

### `CrossMap` (`pyhealth/medcode/cross_map.py:14`)

```python
from pyhealth.medcode import CrossMap

cm = CrossMap.load("ICD9CM", "CCSCM")
cm.map("428.0")                                   # → ['108']  — ALWAYS a list

ndc_atc = CrossMap.load("NDC", "ATC")
ndc_atc.map("50090539100", target_kwargs={"level": 3})   # → ['A10A']
```

`.map()` always returns a **list** — a code may map to several targets, or to none. Handle the
empty case explicitly; do not index `[0]` blind.

Served from PyHealth's own tables, and always preferred: `ICD9CM→CCSCM`,
`ICD10CM→CCSCM`, `ICD9PROC→CCSPROC`, `ICD10PROC→CCSPROC`, `NDC→ATC`, `NDC→RxNorm` — each also
usable in reverse.

Served by the `icd-mappings` package, offline, for pairs PyHealth has no table for:
`ICD9CM↔ICD10CM`, `ICD10CM→CCSR`, `ICD9CM→CCI`, `ICD10CM→CCIR`, `ICD9CM→ICD9CHAPTER`,
`ICD10CM→ICD10CHAPTER`, `ICD10CM→ICD10BLOCK`, and `ICD9CM`/`ICD10CM→CCC`/`CCCSUB`.
`CrossMap.backend` reports which source was used.

**ICD9↔ICD10 is a primary-mapping approximation, not the full GEM relation.** It returns at
most one target, drops some codes entirely, and does not round-trip (`428.0`→`I50.9`→`428.9`).
Check `CrossMap.unmapped_codes` after a pass over a dataset. To unify mixed-vintage data,
prefer mapping *both* versions into a shared grouper over translating one into the other.

---

## Using it inside a task

**Load in `__init__`, never at module scope.** At module scope the map downloads on import and
re-downloads in every `set_task` worker process.

```python
class MyTask(BaseTask):
    task_name = "my_task_ccs"      # ← new name: the samples changed

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.icd9_to_ccs  = CrossMap.load("ICD9CM", "CCSCM")
        self.icd10_to_ccs = CrossMap.load("ICD10CM", "CCSCM")

    def _to_ccs(self, code, version):
        cm = self.icd9_to_ccs if str(version) == "9" else self.icd10_to_ccs
        try:
            mapped = cm.map(code)
        except Exception:
            return None
        return mapped[0] if mapped else None
```

**MIMIC-IV mixes ICD-9 and ICD-10.** Check `event.icd_version` on every code and route to the
matching map. Feeding an ICD-10 code to the ICD-9 map silently drops it, and you lose a chunk of
your features without any error.

Full worked example:
[mortality_visit_mimic4_with_medcode.py](../define-a-task/examples/mortality_prediction/visit/mortality_visit_mimic4_with_medcode.py).

---

## Choosing a mapping

| Situation | Try |
|---|---|
| Vocabulary too large, many rare codes | ICD → CCS |
| Want a cheap version with no download | string truncation: `code.split(".")[0]` |
| Drug codes (NDC) | NDC → ATC level 3 |
| Need a specific hierarchy level | `InnerMap.get_ancestors()` and pick the depth |
| Codes span ICD-9 and ICD-10 | map both to CCS — this is the main reason to bother |

The cheap version first, since it needs nothing:

```python
trunc_icd = code.split(".")[0] if "." in code else code   # "428.0" → "428"
trunc_atc = drug[:3] if len(drug) >= 3 else drug          # "A11CA" → "A11"
```

Truncation and CCS are the `icd_trunc` and `ccs_norm` axes in
[optimize-a-pipeline](../optimize-a-pipeline/SKILL.md). Truncation is nearly always worth
trying; CCS is the bigger reduction but needs the download.

---

## Cautions

- **Unmappable codes vanish silently.** Count them and report the fraction dropped. Losing 30%
  of your diagnosis codes to a failed map is not a preprocessing detail.
- **Mapping changes the samples**, so it needs a new `task_name` — otherwise `set_task` returns
  the pre-mapping cache. See [data-model.md](../../shared/data-model.md).
- **The first load downloads.** Warn the user rather than letting it look like a hang.
- **Grouping loses specificity.** CCS collapses distinct conditions into one category; if the
  clinical question turns on that distinction, do not group.
- **A mapping is an editorial choice.** Say which vocabulary and level you used, in the spec and
  in any reported result.
