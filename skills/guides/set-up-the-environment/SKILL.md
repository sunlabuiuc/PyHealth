---
name: pyhealth-set-up-the-environment
description: Get PyHealth actually installed and running before any modeling work — detect what is already present, pick the right install path (user vs contributor), create the environment, verify the import, and register this skill in the user's project. Use at the start of any PyHealth request when the package is not yet importable, when the import errors, or when the user has just cloned the repo.
---

# Set up the environment

Every other guide assumes `import pyhealth` works. This one makes that true.

Do this work yourself. The user should not be looking up Python versions or
guessing which extras they need — they should answer at most two questions and
watch it happen.

---

## Step 1 — Find out where you actually are

Run this before asking the user anything. Most of what you need to know is
observable, and a question you could have answered by looking is a question you
should not ask.

```bash
python --version
python -P -c "import pyhealth; print(pyhealth.__version__, pyhealth.__file__)" 2>&1 | tail -2
ls pyproject.toml setup.py .venv pixi.toml 2>/dev/null
git remote -v 2>/dev/null | head -2
```

**Use `-P`, always.** Python puts the current directory on the import path, and the
PyHealth repo has a `pyhealth/` directory in its root. Run the check without `-P` from
inside a checkout and it imports the source tree, prints a version, and tells you
everything is fine — in a clone where nothing is installed and not one dependency is
present. `-P` drops cwd from the path so the answer is about the *environment*.

That trap is why a clone is the easiest place to get this wrong: the naive check passes,
you skip setup, and the failure surfaces later as a missing `litdata` from somewhere deep
in a dataset load.

That tells you which of four situations you are in:

| What you see | Situation | Go to |
|---|---|---|
| import works, version prints | already set up | Step 5 — register the skill, then leave |
| `ModuleNotFoundError: pyhealth` | nothing installed | Step 2 |
| `ModuleNotFoundError: litdata` (or `dask`, `torch`) | partial install — deps missing | Step 4 |
| a `pyproject.toml` with `name = "pyhealth"` | a clone of the repo itself | Step 3, contributor path |

**Check the Python version first and separately.** PyHealth requires
`>=3.12,<3.14`. On 3.11 or 3.14 nothing below will work, and the failure comes
out as an unrelated-looking resolver error several minutes into a download. If
the version is wrong, say so and stop — creating an environment on the right
interpreter is the user's call, not something to do silently:

> "This needs Python 3.12 or 3.13; you're on 3.11.9. Do you have a 3.12
> available, or should I set one up with `uv`/`conda`?"

---

## Step 2 — Which install, and where

Two paths, and the distinction is what the user intends to *do*, not what they
know:

| | User path | Contributor path |
|---|---|---|
| Intent | use PyHealth in their own project | change PyHealth itself |
| Install | `pip install pyhealth` | `git clone` + `pip install -e .` |
| Skill arrives via | `python -m pyhealth.skills` | already committed in the clone |
| Guide edits are | fixed at the installed version | live |

Ask once, with a default, and infer hard from context — someone who said "I have
patient CSVs" wants the user path; someone who said "I want to add my GRU
variant" wants the contributor path and should be told the CI gate exists.

**Get explicit permission before creating an environment or installing.** This
writes gigabytes to their disk:

> "I'll create `.venv` here and install PyHealth into it — that pulls PyTorch
> and Dask, so figure a few GB and a few minutes. OK?"

If they already have a venv, conda env, or pixi environment active, use it. Do
not create a second one beside a working one.

---

## Step 3 — Install

**User path:**

```bash
python -m venv .venv
.venv/bin/pip install pyhealth
```

Note the pre-release rule: `pip install pyhealth` resolves to the latest
*stable* version. Alpha releases need to be asked for by name:

```bash
.venv/bin/pip install 'pyhealth==2.1a1'    # or: pip install --pre pyhealth
```

If the user wants the agent-skill features specifically, they need an alpha —
say which version and why, rather than quietly installing something older that
lacks them.

**Contributor path:**

```bash
git clone https://github.com/sunlabuiuc/PyHealth.git
cd PyHealth
make init          # installs pixi, then solves every environment
```

`make init` is the supported route and handles the environments the repo
declares. If pixi is unwanted, `python -m venv .venv && .venv/bin/pip install -e .`
works and is what most contributors reach for. Run the tests once so a later
failure is attributable:

```bash
make test          # or: python -m unittest discover -t tests -s tests/core
```

On Windows the venv binary path is `.venv\Scripts\` rather than `.venv/bin/`.

---

## Step 4 — Verify, and mean it

An install that resolved is not an install that works. Check the layer the user
will actually hit:

```bash
.venv/bin/python -P -c "
import pyhealth
print('pyhealth', pyhealth.__version__)
from pyhealth.datasets import BaseDataset      # pulls litdata, dask, polars
from pyhealth.models import MLP                # pulls torch
print('imports OK')
"
```

`import pyhealth` alone proves almost nothing — the top-level package imports
only the standard library, so it succeeds even when every real dependency is
missing. The dataset and model imports are what exercise the heavy stack, and
they are also what catches the cwd trap from Step 1: run from a checkout, the
bare import resolves against the source tree while `pyhealth.datasets` still
dies on `litdata`.

Show the user the version and the OK line. If it fails:

- `ModuleNotFoundError: litdata` / `dask` / `polars` — installed with `--no-deps`,
  or a partially completed install. Re-run without `--no-deps`.
- A torch wheel that will not resolve — nearly always a Python version outside
  `>=3.12,<3.14`. Back to Step 1.
- `torch-geometric` missing — it is an optional extra, not part of the base
  install. Only graph models need it: `pip install pyhealth[graph]`.

**No GPU is required.** Everything in these guides runs on CPU; the demo-sized
data in `test-resources/` trains in seconds. Do not send someone shopping for
CUDA before they have a pipeline that runs.

---

## Step 5 — Register the skill in their project

So the next session finds it without anyone remembering to:

```bash
.venv/bin/python -m pyhealth.skills          # installed via pip
python tools/install_skills.py               # from a clone
```

This creates `.claude/skills/pyhealth` and appends a marked pointer block to
`AGENTS.md` and `.github/copilot-instructions.md`. It is idempotent, and
`--uninstall` reverses it. Skip it silently if those already exist — a clone of
the repo has them committed already.

Tell the user what it did in one line, and that it is reversible. Writing into
their repo is not something to do invisibly, even when it is harmless.

---

## Step 6 — Hand back to the router

Setup is not the goal, and stopping here leaves the user with a working
installation and no pipeline. Go straight on:

> "PyHealth 2.1a1 is installed and the skill is registered. Now — you mentioned
> predicting readmission from your CSVs. Let me get that data loaded first."

Then re-enter [SKILL.md](../../SKILL.md) and route the actual request: normally
`bring-your-own-data` or `use-a-dataset` next, and `scope-a-modeling-request`
before any modeling code.

---

## Common breakages

**A second environment beside a working one.** Check for an active `VIRTUAL_ENV`,
`CONDA_DEFAULT_ENV`, or `PIXI_ENVIRONMENT_NAME` before creating anything. Two
environments means the user runs one and you test the other, and the resulting
"but it works for me" costs an hour.

**Installing into the system Python.** If `python` is `/usr/bin/python3`, make a
virtual environment first. A `pip install` that needs `sudo` is a wrong turn, not
a permissions problem.

**Calling the wrong interpreter afterwards.** After creating `.venv`, every
command is `.venv/bin/python`, not `python`. Mixing them is the single most
common cause of "I installed it but it says not found".

**A stale install shadowing a checkout.** If someone pip-installed PyHealth and
also cloned it, `import pyhealth` may resolve to either. `pyhealth.__file__`
says which — print it whenever behavior does not match the source you are
reading.

**Assuming the alpha is the default.** `pip install pyhealth` will not give you
`2.1a1`. If a feature is missing that you know exists, check the version before
debugging anything else.
