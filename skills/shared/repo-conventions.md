# Repo conventions and the CI gate

Everything a contribution to `pyhealth/` must satisfy. Read this before writing code you intend
to upstream — the gate is machine-enforced, and it fails PRs that are otherwise fine.

Sources: `CONTRIBUTING.md`, `docs/how_to_contribute.rst`,
`.github/workflows/pr_contribution_rules.yml`, `tools/check_pr_rules.py`.

---

## The gate

`.github/workflows/pr_contribution_rules.yml` runs
`python tools/check_pr_rules.py --base <sha> --head HEAD` on every PR. If the PR touches any
`pyhealth/**/*.py`, three rules fire:

**1. Docs + examples are mandatory.** The PR must also modify at least one file under `docs/`
**and** at least one under `examples/`. Not "should" — the check fails otherwise. This is the
rule people trip on most, because a correct, well-tested module still fails without them.

**2. Ruff-clean on changed lines.** Only lines you added or modified are checked; pre-existing
violations elsewhere in the same file are left alone. So you cannot inherit someone else's
debt, and you cannot add to it.

**3. Docstring examples.** Every new or modified top-level public class or function in
`pyhealth/**/*.py` needs a `>>>` usage example in its docstring. `tools/check_pr_rules.py`
parses the AST and looks for it.

```python
def binary_metrics_fn(y_true, y_prob, metrics=None, threshold=0.5):
    """Computes metrics for binary classification.

    Args:
        y_true: True target values of shape (n_samples,).
        ...

    Examples:
        >>> from pyhealth.metrics import binary_metrics_fn
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        >>> binary_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
    """
```

**Plan for all three from the start.** Discovering the docs/examples rule after the
implementation is done means writing them under time pressure, badly.

---

## Docs: the `docs/api/` pattern

Two pieces per new public class, both required:

**A per-class stub**, `docs/api/<subsystem>/pyhealth.<module>.<ClassName>.rst`:

```rst
pyhealth.datasets.NewDataset
=============================

Overview
--------

Brief description of the dataset, its source, and key characteristics.

API Reference
-------------

.. autoclass:: pyhealth.datasets.NewDataset
    :members:
    :undoc-members:
    :show-inheritance:
```

**A toctree entry** in the subsystem hub page (`docs/api/datasets.rst`, `tasks.rst`,
`models.rst`, `processors.rst`, ...). A stub with no toctree entry renders nowhere and Sphinx
warns about an orphan document.

Directory sizes today, for a sense of the pattern: `models/` 44 stubs, `tasks/` 32,
`datasets/` 30, `processors/` 28, `interpret/` 9, `metrics/` 8. `calib/` has only module-level
pages; `medcode`, `tokenizer`, and `trainer` are single flat pages with no subdirectory.

## Examples

One runnable script or notebook under `examples/`, placed in the topical subdirectory that fits
— `mortality_prediction/`, `drug_recommendation/`, `length_of_stay/`, `readmission/`,
`interpretability/`, `conformal_eeg/`, `cxr/`, `eeg/`, `clinical_tasks/`, `tutorials/` — or at
the top level if none fits.

It must actually run against a dataset a reader can get. Do not commit an example that requires
credentialed data with no fallback; say in a comment what the reader needs.

## Tests

Location: `tests/core/test_<thing>.py`. Flat, ~118 files, no per-subsystem subdirectories.
`tests/base.py` provides `BaseTestCase(unittest.TestCase)` with seeding and logging helpers.

**Never use real datasets in tests.** From `docs/how_to_contribute.rst`:

- Generate data programmatically with pandas/numpy/dicts, in-memory
- Or use tiny fixture files in `test-resources/` — a few KB at most
- **2–5 patients, 5–20 events total** — just enough to exercise the logic
- Build data in `tempfile.mkdtemp()` and clean up afterwards

Run them with `python -m unittest discover -t tests -s tests/core`.

## Code style

- PEP 8, 88-character lines
- Google-style docstrings with `Args:` / `Returns:` / `Raises:` / `Examples:`
- Type hints on all public methods
- Python `>=3.12,<3.14`

## Branching

GitHub Flow, per `CONTRIBUTING.md`: `master` (stable), `develop` (integration), `*-release`,
plus feature and hotfix branches. Target `develop` unless told otherwise.

---

## Pre-flight checklist

Before opening a PR that touches `pyhealth/`:

- [ ] Implementation exports its public names from the subsystem `__init__.py`
- [ ] Every new public class/function has a `>>>` docstring example
- [ ] `docs/api/<subsystem>/pyhealth.<module>.<Name>.rst` exists
- [ ] Its toctree entry is added to the subsystem hub page
- [ ] An `examples/` script exists and runs
- [ ] `tests/core/test_<name>.py` exists, uses synthetic data, runs in seconds
- [ ] `ruff check` clean on the lines you touched
- [ ] `python -m unittest discover -t tests -s tests/core` passes
