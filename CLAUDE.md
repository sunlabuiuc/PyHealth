# PyHealth — notes for coding agents

**Before any PyHealth work** — loading a dataset, defining a task, choosing
processors or a model, training, evaluating, calibrating, interpreting, mapping
medical codes, or contributing a module upstream — read
[`skills/SKILL.md`](skills/SKILL.md).

It is a router. It decomposes the request into the PyHealth subsystems actually
involved, then points at the specific guides under `skills/guides/` that cover
them. Read only those; loading all thirteen defeats the design.

Two hard rules it enforces, both because the expensive failure in clinical ML is
decided before the first line of code: **do not write pipeline code before a
spec is confirmed**, and **do not write package code before the target file list
is agreed** — a PR touching `pyhealth/` that lacks `docs/` and `examples/` files
fails CI regardless of how good the module is.

Full manifest: [`skills/table-of-contents.md`](skills/table-of-contents.md).
Human overview: [`skills/README.md`](skills/README.md).

## Repo orientation

- `pyhealth/` — the installable package (`datasets`, `tasks`, `processors`,
  `models`, `metrics`, `trainer.py`, `medcode`, ...)
- `pyhealth/datasets/configs/*.yaml` — table configs; the model to copy when
  writing one for custom data
- `examples/`, `docs/`, `tests/` — usage examples, Sphinx docs, unit tests
- `skills/` — agent instructions (this file points at them); also shipped in the
  wheel at `pyhealth/skills/_bundle/`, installable with `python -m pyhealth.skills`
- `pyhealth/skills/` — the installer behind both entry points, plus the
  agent-only notice printed on first `import pyhealth`
- `tools/` — maintenance scripts: `install_skills.py`, `check_skills.py`
  (CI guard for the above), `check_pr_rules.py` (the PR gate)

Python `>=3.12,<3.14`. Tests:
`python -m unittest discover -t tests -s tests/core`.
