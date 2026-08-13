---
name: pyhealth-ship-a-contribution
description: Get a PyHealth change through CI — the enforced rule that any PR touching pyhealth/ must also add docs/ and examples/ files, ruff-clean changed lines, `>>>` docstring examples on new public APIs, and fast synthetic-data tests. Use when preparing, reviewing, or debugging a PR against PyHealth.
---

# Ship a contribution

PyHealth's CI enforces things most repos only ask for. A correct, well-tested module still fails
if it lands without documentation and an example. Work through this before opening the PR.

Authority: `.github/workflows/pr_contribution_rules.yml` runs
`python tools/check_pr_rules.py --base <sha> --head HEAD`. Read that script if a failure is
unclear — it is short and it is the actual rule.

---

## The three enforced rules

### 1. Docs and examples are mandatory

If the PR touches any `pyhealth/**/*.py`, it must **also** modify at least one file under
`docs/` and one under `examples/`. No exceptions in the checker.

This is the rule that fails good PRs. Plan the docs and example file at the same time as the
implementation.

### 2. Ruff-clean on changed lines

Only lines you added or modified are linted; pre-existing violations in the same file are left
alone. So you cannot inherit someone else's debt — and you cannot add to it.

```sh
ruff check pyhealth/<your_file>.py
```

### 3. `>>>` docstring examples on new public APIs

Every new or modified top-level public class or function in `pyhealth/**/*.py` needs a runnable
`>>>` example in its docstring. The checker parses the AST looking for it.

```python
def my_function(x, y):
    """One-line summary.

    Args:
        x: What x is.
        y: What y is.

    Returns:
        What comes back.

    Examples:
        >>> from pyhealth.module import my_function
        >>> my_function(1, 2)
        3
    """
```

Make the example real — an import line plus a call plus the actual output. A placeholder that
would not run is worse than none, because someone will copy it.

---

## Docs: two files, both required

**A per-class stub** at `docs/api/<subsystem>/pyhealth.<module>.<ClassName>.rst`:

```rst
pyhealth.datasets.NewDataset
=============================

Overview
--------

Brief description: what it is, where the data comes from, key characteristics.

API Reference
-------------

.. autoclass:: pyhealth.datasets.NewDataset
    :members:
    :undoc-members:
    :show-inheritance:
```

**A toctree entry** in the subsystem hub page — `docs/api/datasets.rst`, `tasks.rst`,
`models.rst`, `processors.rst`, `interpret.rst`, `metrics.rst`. A stub with no toctree entry
renders nowhere and Sphinx warns about an orphan.

Not every subsystem uses per-class stubs: `calib/` has module-level pages only, and `medcode`,
`tokenizer`, `trainer` are single flat pages. **Match the pattern of the directory you are
adding to** rather than applying the dataset pattern everywhere.

## Examples: one runnable script

Under `examples/`, in the topical subdirectory that fits — `mortality_prediction/`,
`drug_recommendation/`, `length_of_stay/`, `readmission/`, `interpretability/`,
`conformal_eeg/`, `cxr/`, `eeg/`, `clinical_tasks/`, `tutorials/` — or at the top level if none
does.

It must run. If it needs credentialed data, say so in a comment at the top and give the reader
the smallest path to trying it.

## Tests: fast, synthetic, `tests/core/`

`tests/core/test_<name>.py`. Flat directory, ~118 files. `tests/base.py` gives you
`BaseTestCase(unittest.TestCase)` with seeding and logging helpers.

**Never use a real dataset.** From `docs/how_to_contribute.rst`:

- Generate data programmatically, in memory
- Or tiny fixtures in `test-resources/` — a few KB
- **2–5 patients, 5–20 events**
- Build in `tempfile.mkdtemp()`, clean up after

```python
import tempfile, pandas as pd
from pathlib import Path

class TestNewThing(BaseTestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        pd.DataFrame({"subject_id": ["1", "2"], "gender": ["M", "F"]}) \
          .to_csv(Path(self.dir) / "patients.csv", index=False)
```

Run: `python -m unittest discover -t tests -s tests/core`

---

## Before you open the PR

- [ ] Public names exported from the subsystem `__init__.py`
- [ ] `>>>` docstring example on every new public class/function
- [ ] `docs/api/<subsystem>/pyhealth.<module>.<Name>.rst` written
- [ ] Toctree entry added to the subsystem hub page
- [ ] `examples/<topic>/<name>.py` written **and run**
- [ ] `tests/core/test_<name>.py` — synthetic data, runs in seconds
- [ ] `ruff check` clean on the lines you touched
- [ ] `python -m unittest discover -t tests -s tests/core` passes
- [ ] Branch targets `develop` unless told otherwise
- [ ] PEP 8, 88-char lines, Google docstrings, type hints on public methods

## When CI fails anyway

- **"must also modify docs/ and examples/"** — rule 1. You committed only `pyhealth/`.
- **A ruff error on a line you did not write** — you modified that line (even reformatting
  counts). Fix it or revert the incidental change.
- **"missing docstring example"** — a public class or function lacks `>>>`. Private names
  (leading underscore) and nested definitions are exempt; top-level public ones are not.
- **Sphinx orphan warning** — stub written, toctree entry forgotten.
- **A slow test** — you used a real dataset. Replace it with synthetic data.

Full conventions: [repo-conventions.md](../../shared/repo-conventions.md).
Implementation guidance per component kind: [add-a-component](../add-a-component/SKILL.md).
