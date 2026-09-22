# Synthea: bringing the new code in line with `pyhealth/datasets/`

Working checklist for the in-flight Synthea work on branch `dev`.

**What's currently uncommitted:**

```
 M pyhealth/datasets/__init__.py
?? pyhealth/datasets/configs/synthea.yaml
?? pyhealth/datasets/configs/synthea_properties.txt
?? pyhealth/datasets/synthea.py
?? pyhealth/datasets/synthea_generator.py
?? scripts/                        # entirely new — 3 Synthea dev scripts, nothing else
```

The code itself is close to the house style. Almost everything below is a
*missing companion artifact* rather than a defect in the loader or generator.

---

## Tier 1 — Hard gaps: artifacts every sibling dataset has and Synthea doesn't

### 1. No test in `tests/core/`

Every dataset in the folder has one — `test_tuab.py`, `test_eicu.py`,
`test_meds.py`, `test_support2.py`, and ~120 more, all
`unittest.TestCase`. Synthea has none.

What exists instead is an untracked `scripts/` directory containing three
hand-rolled harnesses, with module-global `PASS`/`FAIL` counters, custom
`check()` / `expect_raises()` helpers, and `argparse` phase selection:

| File | What it does |
|---|---|
| `scripts/synthea_generator_test.py` | offline + `--live` checks for `SyntheaGenerator` |
| `scripts/synthea_test.py` | loads real Synthea CSV through `SyntheaCSVDataset` |
| `scripts/synthea_sample.py` | downloads/generates a CSV sample; prototypes argv building |

**To do:**

- [ ] **Add `tests/core/test_synthea.py`** — `class TestSyntheaDataset(unittest.TestCase)`.
      Follow `tests/core/test_tuab.py`: `tempfile` plus written-out CSV fixtures,
      with `_DummyEvent` / `_DummyPatient` dataclasses where a real load is too
      heavy. Cover:
  - `preprocess_procedures` on both branches (`start` present vs. `date`-only)
  - default selection plus explicit-table deduplication
  - default config resolution when `config_path is None`
- [ ] **Add `tests/core/test_synthea_generator.py`** — port the `offline_checks()`
      body from `scripts/synthea_generator_test.py`: `build_argv` construction,
      `extra_config` allowlist and reserved-key rejection, every constructor
      `ValueError` / `TypeError`, and Java resolution order. Mechanical
      conversions:
  - `check(label, cond)` → `self.assertTrue(cond, label)`
  - `expect_raises(..., needle=...)` → `with self.assertRaisesRegex(...)`
  - `no_java_available()` and `empty_jar_cache()` port over unchanged — both are
    already `unittest.mock`-based context managers
- [ ] **Gate the live phase** with
      `@unittest.skipUnless(os.environ.get("PYHEALTH_SYNTHEA_LIVE"), "requires Java + ~201MB jar")`.
      That's the established pattern — `test_meds.py`, `test_eegbci.py`,
      `test_halo.py`, `test_promptehr.py` all use `skipUnless` / `skipIf` for
      expensive or dependency-gated paths.
- [ ] **Delete `scripts/` entirely.** The directory is untracked and holds
      nothing but this scaffolding, so removing the three files removes the
      directory. `synthea_sample.py` in particular is dead by its own admission —
      its docstring says it exists to prototype "the argv-building logic a future
      `SyntheaGenerator` would need," and `build_argv` now does that for real.
      Salvage the sample-data download URLs into the test fixture first if the
      live test wants them.

### 2. No docs pages

`docs/api/datasets/` has an `.rst` for all 28 exported datasets — including the
deprecated `CardiologyDataset` and `MIMICExtractDataset`. Synthea has none.

- [ ] **Add `docs/api/datasets/pyhealth.datasets.SyntheaCSVDataset.rst`.** Minimum
      viable form is `pyhealth.datasets.Support2Dataset.rst`: title with `=====`
      underline, short `Overview` section, then
      `.. autoclass:: pyhealth.datasets.SyntheaCSVDataset` with `:members:`,
      `:undoc-members:`, `:show-inheritance:`.
- [ ] **Add `docs/api/datasets/pyhealth.datasets.SyntheaGenerator.rst`.** This one
      warrants full documentation: `.. contents:: On this page`, a `Quick
      start` `code-block:: python`, and
      sections for the Java/jar acquisition order and the `PYHEALTH_NO_*` env-var
      switches. None of that is guessable from the signature.
- [ ] **Add both to the `toctree`** in `docs/api/datasets.rst` under
      `Available Datasets`. That list is grouped loosely by family rather than
      alphabetically — put `SyntheaCSVDataset` near the EHR loaders (after
      `OMOPDataset`) and `SyntheaGenerator` immediately after it.

### 3. `pyhealth[synthea]` extra doesn't exist

`pyhealth/datasets/synthea_generator.py:669` tells users to run
`pip install 'pyhealth[synthea]'`. `pyproject.toml:63` defines only `graph`,
`nlp`, and `lint`. **That instruction fails today.**

Pick one:

- [ ] **(recommended)** Add `synthea = ["jdk4py>=25"]` to
      `[project.optional-dependencies]`. Consistent with how `graph` / `nlp` gate
      optional heavy deps, and makes the message true. Derive the version in the
      message from the existing `JDK4PY_REQUIREMENT` constant rather than
      hardcoding it in two places.
- [ ] Or drop the extra from the message and point only at `pip install 'jdk4py>=25'`.

### 4. No CHANGELOG entry

`CHANGELOG.md` carries a `### New datasets` section per release, one bullet per
dataset, each ending in a PR link.

- [ ] Add a bullet under an unreleased / next-version heading covering both the
      loader and the generator.

---

## Tier 2 — Real code inconsistencies

### 5. Type-hint style

28 of 31 files in `pyhealth/datasets/` use `typing.Optional[...]` / `List[...]`.
Only `meds.py`, `synthea.py`, and `synthea_generator.py` use PEP 604
(`list[str] | None`). Even `base_dataset.py` runs 16 `Optional[...]` to 5
`| None`.

- [ ] Convert `synthea.py` to `Optional[List[str]]` / `Optional[str]` with
      `from typing import List, Optional`. It's a 121-line loader that should read
      exactly like its siblings.
- [ ] Leave `synthea_generator.py` alone. It isn't a `BaseDataset` subclass and
      nothing reads as inconsistent within it.

Worth noting this may be deliberate drift rather than a mistake:
`target-version = "py313"` makes PEP 604 perfectly legal, and the two newest
files in the folder both use it. If the project is moving that way, skip this
item.

### 6. Docstring `Args:` missing types in `SyntheaGenerator`

House convention is `root (str): The root directory...` — used in every loader,
and in the generator's own `run()` (`timeout (float | None):`) and
`_validated_path()`. But the class docstring's ~35 `Args:` entries omit types
entirely (`output_dir: Where Synthea writes its output.`). Inconsistent *within
the same file*.

- [ ] Add parenthesized types to all `Args:` entries in the class docstring.

### 7. `Attributes:` block is incomplete

`synthea_generator.py:295` documents only `output_dir` and `population`, but
`__init__` sets ~30 public attributes. Siblings list every public attribute they
set.

- [ ] Either list the meaningful attributes or cut the block.

### 8. `print()` for the install/download notices

`synthea_generator.py:674` and `:718`. Modern convention is `logger.*` —
`base_dataset.py` uses `logger.info` even for "no cache found, building" notices.

The counter-argument is real: these two announce a ~35MB pip install and a
~201MB download *before* they happen, `logging` is unconfigured by default in
most user scripts, so `logger.info` would be silent and a long run would look
hung. `stats()` in `base_dataset.py` also uses bare `print`.

- [ ] **Recommendation:** keep `print`, add a one-line comment explaining why, so
      the next reader doesn't "fix" it. If uniformity matters more,
      `logger.warning` is the compromise that survives default config.

---

## Tier 3 — Looks like a gap, isn't. Don't spend time here.

- **No `default_task` on `SyntheaCSVDataset`.** Only 11 of the loaders define one;
  `MIMIC3Dataset`, `eICUDataset`, `EHRShotDataset`, `Support2Dataset`, and
  `OMOPDataset` all omit it. Synthea has no canonical task, so omitting is
  correct — `set_task()` still works with an explicit task argument.
- **`import narwhals as pl`** in `synthea.py` — exactly matches `mimic3.py`,
  `omop.py`, `bmd_hs.py`. Correct.
- **Line length** — both files are fully under the 88-char ruff limit.
- **No `if __name__ == "__main__":` block** — that's a *legacy* marker
  (`cardiology.py`, `shhs.py`, `isruc.py`, all of which subclass the deprecated
  `BaseSignalDataset` stub). Not having one is right.
- **`SyntheaGenerator` not subclassing anything** — appropriate *for that class*.
  It's a generator, not a dataset; `run() -> Path` feeding
  `SyntheaCSVDataset(root=...)` is the correct seam, and that seam stays.
  **Superseded in part (2026-09-19):** a *third* class,
  `SyntheaGeneratorDataset`, will subclass `SyntheaCSVDataset` to offer a
  generate-then-load object. `SyntheaGenerator` itself remains standalone and
  unchanged. See Part II.
- **Both symbols already exported** in `pyhealth/datasets/__init__.py`, using the
  `as`-aliased re-export form. Done.

---

## Reference: the pattern being matched

`synthea.py` is a textbook instance of the canonical loader shape. For anything
added later, the four-step constructor is:

```python
def __init__(self, root, tables, dataset_name=None, config_path=None, **kwargs) -> None:
    # 1. resolve default config
    if config_path is None:
        logger.info("No config path provided, using default config")
        config_path = Path(__file__).parent / "configs" / "<name>.yaml"
    # 2. prepend mandatory tables
    default_tables = [...]
    tables = default_tables + tables
    # 3. delegate, defaulting the name inline
    super().__init__(
        root=root,
        tables=tables,
        dataset_name=dataset_name or "<lowercase_name>",
        config_path=config_path,
        **kwargs,
    )
```

`**kwargs` is the standard pass-through for `cache_dir` / `num_workers` / `dev`.
The data work itself is declarative — the YAML config drives
`BaseDataset.load_table()`, which lowercases all columns, calls
`self.preprocess_<table>` if it exists, applies joins, parses timestamps, and
emits the fixed `["patient_id", "event_type", "timestamp"] + ["{table}/{attr}", ...]`
schema.

There are only four extension hooks: `preprocess_<table>()`, the `default_task`
property, a `load_data()` override, and a `set_task()` wrapper. Synthea uses the
first and needs none of the others.

---

## Not verified

`ruff` isn't installed in `.venv`, so line length was checked by hand rather than
by running the linter. Do a real `ruff check` pass on both files before calling
this done — `pip install 'pyhealth[lint]'` gets the pinned `ruff~=0.15`.

---
---

# Part II — `SyntheaGeneratorDataset` (design, not yet implemented)

Decisions from the 2026-09-19 design pass. Everything marked *measured* was run
against `test-resources/core/mimic3demo` (100 patients / 12,894 events) and a
3-patient slice of MITRE's `latest` sample.

```
SyntheaGeneratorDataset          new
  └─ SyntheaCSVDataset           synthea.yaml + preprocess_procedures
      └─ BaseDataset             20 members, zero @abstractmethod
```

Holds `SyntheaGenerator` as `self._generator` — does not inherit from it.
**Overrides 4 of 20. Inherits 16.**

## II.1 Overrides

| # | Method | Line | MIMIC | Synthea | Why |
|---|---|---|---|---|---|
| 1 | `__init__` | 329 | override | **override** | build generator → fingerprint → derive `root` → delegate |
| 2 | `_init_cache_dir` | 375 | inherit | **override** | `root` doesn't identify the data (II.2) |
| 3 | `load_data` | 650 | inherit¹ | **override** | data doesn't exist yet (II.3) |
| 4 | `stats` | 831 | inherit | **override** (optional) | print population/seed/state/`SYNTHEA_VERSION` |

¹ `MIMIC4Dataset` does override it (`mimic4.py:334`), for a different reason:
its data lives in 3 sub-datasets. It *replaces* `super()`; Synthea *calls* it.

`__init__` ordering is mandatory — `BaseDataset.__init__` calls
`_init_cache_dir` on its last line (371), and the override reads the
fingerprint:

```
1. self._generator = SyntheaGenerator(...)
2. self._gen_fingerprint = ...          # must precede step 4
3. root = Path(output_dir).resolve() / "csv"
4. super().__init__(root=str(root), ...)
```

Same shape as `MIMIC4CXRDataset.prepare_metadata` (`mimic4.py:204`), which also
produces files into `root` before `super().__init__()`.

## II.2 `_init_cache_dir` — the one that matters

Base hashes `{root, sorted(tables), dataset_name, dev}`. MIMIC is fine: `root`
is where the data *is*. Synthea isn't: `root` is where output *goes*.

*Measured* — these two could be `seed=1` and `seed=2`:

```
SyntheaCSVDataset(root=".../out/csv")  ->  d3918a78-3720-5f72-a48a-c29fc01d5b83
SyntheaCSVDataset(root=".../out/csv")  ->  d3918a78-3720-5f72-a48a-c29fc01d5b83   IDENTICAL
```

The second silently reads the first's parquet. No warning at any layer.

Fix — add `"generator": self._gen_fingerprint` to the hashed dict:

```
argv = self._generator.build_argv(Path("java"), Path("synthea.jar"))
fingerprint = sha256("\x00".join(argv[3:])).hexdigest()
```

*Measured*, with `auto_install_java=False, auto_download_jar=False` — no JVM,
no download (`build_argv` is documented pure, `synthea_generator.py:752`):

```
argv tail seed=1 : [..., '-p', '100', '-s', '1']
argv tail seed=2 : [..., '-p', '100', '-s', '2']
fingerprint      : faf6f2a5feb6   vs   c8ff610688c0      DIFFER
same params twice: stable
```

Slice `argv[:3]` (`[java, "-jar", jar]`) so the jar path stays out of the key.
Derive from `build_argv`, never a hand-written param list.

`set_task` caches under `cache_dir/tasks/`, so this one fix partitions all three
cache stages.

## II.3 `load_data`

```
if not self._generated and (self._regenerate or not (root/"patients.csv").exists()):
    log loudly; self._generator.run(timeout=self._timeout); self._generated = True
return super().load_data()
```

Here and not `__init__` because `_event_transform` calls `load_data()` at line
572 and starts the Dask cluster at 573 — the JVM runs in a clean main process,
and only on a cache miss. Test on `patients.csv`, not "directory non-empty": a
failed run leaves a partial directory.

`timeout` moves from a `run()` argument to a constructor argument, since the
caller never invokes `run()` directly.

## II.4 Table selection

Three layers, same mechanism as MIMIC:

```
files on disk      →   declared in YAML   →   selected by tables=
```

| | Synthea | MIMIC-III |
|---|---|---|
| On disk | **18** CSVs | 6 in the demo dir |
| Declared in YAML | 8 today → **18** proposed (II.5) | 8 (incl. `labevents`, `noteevents` — absent from the demo dir, declared but unselected) |
| Forced defaults | `["patients","encounters"]` | `["patients","admissions","icustays"]` |

`load_data` is `[self.load_table(t.lower()) for t in self.tables]` (656).
Unselected files are never opened. *Measured:*

```
SyntheaCSVDataset(root=..., tables=["conditions"]).tables
  -> ['patients', 'encounters', 'conditions']
```

Declaring a table in YAML is a capability, not a requirement — a selected table
whose file is missing raises `FileNotFoundError` from `_csv_tsv_gz_path` (90).

## II.5 All 18 CSVs

Three groups. Group 3 is the reason this needs a design and not just YAML.

**Group 1 — event tables, 13.** 8 declared today + 5 to add. All timestamp
columns verified fully populated, so `errors="raise"` (742) is safe.

| Table | `patient_id` | `timestamp` | `timestamp_format` | `encounter`? | Status |
|---|---|---|---|---|---|
| `patients` | `id` | `birthdate` | `%Y-%m-%d` | — | declared |
| `encounters` | `patient` | `start` | `%Y-%m-%dT%H:%M:%SZ` | is the key | declared |
| `conditions` | `patient` | `start` | `%Y-%m-%d` | yes | declared |
| `medications` | `patient` | `start` | `%Y-%m-%dT%H:%M:%SZ` | yes | declared |
| `observations` | `patient` | `date` | `%Y-%m-%dT%H:%M:%SZ` | yes | declared |
| `procedures` | `patient` | `start` | `%Y-%m-%dT%H:%M:%SZ` | yes | declared, has hook |
| `immunizations` | `patient` | `date` | `%Y-%m-%dT%H:%M:%SZ` | yes | declared |
| `careplans` | `patient` | `start` | `%Y-%m-%d` | yes | declared |
| `allergies` | `patient` | `start` | `%Y-%m-%d` | yes | **add** |
| `devices` | `patient` | `start` | `%Y-%m-%dT%H:%M:%SZ` | yes | **add** |
| `imaging_studies` | `patient` | `date` | `%Y-%m-%dT%H:%M:%SZ` | yes | **add** |
| `supplies` | `patient` | `date` | `%Y-%m-%d` | yes | **add** |
| `payer_transitions` | `patient` | `start_date` | `%Y-%m-%dT%H:%M:%SZ` | **no** | **add** |

Date-only is the *minority* — 4 of 13. Per-table `timestamp_format` stays
mandatory.

**Group 2 — claims, 2.** Key on `PATIENTID`, not `PATIENT`. Declare with
`patient_id: "patientid"`, but keep **out of `DEFAULT_TABLES`** so they are
opt-in: `claims_transactions` is 85,047 rows for 108 patients (38 MB) against
3,518 for `conditions`, and would dominate every event frame and task scan.

**Group 3 — reference tables, 3.** `organizations` (279 rows), `payers` (11),
`providers` (279). **No patient column at all.**

## II.6 Why reference tables cannot be event tables

`patient_id: null` triggers the row-index fallback (754-762), assigning
`patient_id = "0", "1", "2", …`.

That matters because **`unique_patient_ids` is a union across every loaded
table**, not a read of `patients.csv`. *Measured* — deleted one patient's row
from `patients.csv`, left their 21 `conditions` rows:

```
unique_patient_ids: 3      <- orphan still present
Number of events:  1756    <- was 1757, exactly the one patients row
```

So declaring the 3 reference tables as event tables injects 569 fake patients
into every count, split and task.

Overriding "the patient functions" doesn't fix it cheaply — patient identity is
derived in **five** independent places, only one of which reads
`unique_patient_ids`:

| Line | Method | Derivation | Overridable |
|---|---|---|---|
| 584 | `_event_transform` | `df["patient_id"].unique().head(1000)` (dev filter) | yes, ~50 lines of Dask plumbing |
| 779 | `unique_patient_ids` | own `.unique()` | trivially |
| 819 | `iter_patients` | own `.unique()` | trivially |
| 835 | `stats` | `n_unique()` | trivially |
| 861 | `_task_transform` | own `.unique()` | yes, ~85 lines of spawn-pool code |
| **232** | `_task_transform_fn` | `partition_by("patient_id")` | **no** — module-level function |

**Rejected.** ~135 lines forked from base, and line 232 unreachable.

## II.7 Decision: reference tables load beside the event frame

Chosen 2026-09-19. Rejected alternative: YAML `join:` blocks onto `encounters`
(zero code, but denormalized and limited to fields named up front).

**Declaration** — new `reference_tables:` key in YAML, backed by a new
`ReferenceTableConfig` and a `reference_tables: Dict[str, ReferenceTableConfig]
= {}` field on `DatasetConfig` (`configs/config.py`). Defaulting to empty keeps
all 30 other dataset configs unaffected.

```yaml
reference_tables:
  organizations: {file_path: "organizations.csv"}
  payers:        {file_path: "payers.csv"}
  providers:     {file_path: "providers.csv"}
```

**Access** — three lazy cached properties returning `pl.LazyFrame`:

```
ds.organizations   # 279 rows
ds.payers          # 11 rows
ds.providers       # 279 rows
```

**No `load_data` override needed for this.** Reference tables live under
`reference_tables:`, not `tables:`, so `self.tables` never contains them and
`load_data` never sees them. `unique_patient_ids` stays correct by construction.

Loader should go through `self._scan_table` so it inherits the `.csv`/`.gz`
fallback and URL handling, then lowercase columns to match `load_table`.

Note this widens PR B into shared config code (`configs/config.py`), which
reviewers will scrutinise. Backward compatible, but call it out in the PR.

## II.8 `root` is always local

`_scan_csv_tsv_gz` supports URLs, but MITRE ships a **zip**, not addressable
CSVs. *Measured:*

```
404  .../synthea-sample-data/downloads/latest/patients.csv
200  .../downloads/latest/synthea_sample_data_csv_latest.zip   (application/x-zip-compressed)

200  https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/PATIENTS.csv
404  https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/PATIENTS.csv.gz
```

MIMIC's URL root works because each file is addressable — and that 404/200 pair
is exactly why `_csv_tsv_gz_path`'s fallback exists (`mimic3.yaml` declares
`.csv.gz`, the mirror serves `.csv`).

For `SyntheaGeneratorDataset` the question is moot: `root` is `output_dir/csv`,
a path the generator writes. Derive it internally; don't expose it as a
constructor param.

## II.9 The 16 inherited

Unchanged from MIMIC except where noted.

| Member | Line | Note |
|---|---|---|
| `create_tmpdir` / `clean_tmpdir` | 421 / 431 | `clean_tmpdir` rmtree's `cache_dir/tmp` — **generated CSVs must live under `output_dir`** |
| `_scan_table` | 437 | always routes to the CSV scanner |
| `_scan_parquet` | 458 | unreachable |
| `_scan_csv_tsv_gz` | 499 | all columns `string[pyarrow]`, `""` → `<NA>`. `.gz` fallback never fires |
| `_event_transform` | 570 | `load_data()` at 572, Dask at 573 — see II.3 |
| `global_event_df` | 621 | lazy trigger. *Measured:* 66 cols / 1,757 rows (Synthea) vs 49 / 12,894 (MIMIC-III) |
| `load_table` | 659 | lowercases headers; runs `preprocess_procedures`; **no joins declared** (MIMIC joins `ADMISSIONS` for `dischtime`) |
| `unique_patient_ids` | 771 | union across tables — II.6 |
| `get_patient` | 789 | only caller of `unique_patient_ids` |
| `iter_patients` | 810 | own derivation |
| `default_task` | 842 | `None`; bare `set_task()` raises. Part I Tier 3 |
| `_task_transform` | 851 | `num_workers` clamps to patient count |
| `_proc_transform` | 937 | `"zero samples"` = wrong `event_type` string in a task |
| `set_task` | 1004 | inherits the widened `cache_dir` |
| `_main_guard` | 1166 | `exit(1)`, not raise. Satisfied — generation is main-process. **Never move it into a worker** |

## II.10 Traps

| Trap | Line | Mitigation |
|---|---|---|
| Cache key uses raw `str(self.root)` before `clean_path` | 400 | resolve `output_dir` in `__init__` |
| Dedupe + cache key case-sensitive; only 656 lowercases | 356/401/656 | keep tables lowercase |
| `list(set(tables))` loses order nondeterministically | 358 | `SyntheaCSVDataset` already avoids it — don't regress |
| `patients` in defaults is for demographics, not ids | — | dropping it still gives the right patient count, but tasks reading `gender`/`birthdate` silently yield nothing |

## II.11 Work items

- [ ] `configs/config.py` — `ReferenceTableConfig` + `reference_tables` field
- [ ] `configs/synthea.yaml` — 5 new event tables, 2 claims, `reference_tables:` block
- [ ] `synthea.py` — extend `DEFAULT_TABLES` (13, claims excluded), 3 lazy properties
- [ ] new `synthea_generator_dataset.py` — the 4 overrides
- [ ] `tests/core/test_synthea_generator_dataset.py` — **cache-key discrimination first**; fingerprint stability/coverage; inert construction; CSV exporter enforcement; regenerate guard; live `population=5` gated on `PYHEALTH_SYNTHEA_LIVE`
- [ ] test that `ds.payers` does not appear in `unique_patient_ids`
- [ ] docs `.rst` + toctree, `__init__.py` export, CHANGELOG — per Part I Tier 1
