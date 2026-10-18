"""Helpers to build the PyHealth tutorial notebooks as .ipynb files.

Each module in tutorials/ defines `FILENAME` and `cells()`; `build.py` writes
the notebooks one folder up. Cells tagged "install" are skipped when
`build.py --run` executes the notebooks against the installed PyHealth.
"""

import textwrap

import nbformat

PYHEALTH_SPEC = "pyhealth[xgboost] @ git+https://github.com/sunlabuiuc/PyHealth.git"

INSTALL = f'''# First run: installs PyHealth, then restarts the runtime once so Python
# loads the new package versions. Then choose Runtime > Run all again;
# this cell will see PyHealth is installed and skip the install.
import importlib.util
import os
import subprocess
import sys

PYHEALTH = "{PYHEALTH_SPEC}"

if importlib.util.find_spec("pyhealth") is None:
    print("Installing PyHealth (1-2 minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", PYHEALTH],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-5000:], result.stderr[-5000:])
        raise RuntimeError("Installing PyHealth failed; see the log above.")
    print("Installed. Restarting the runtime now; then choose Runtime > Run all again.")
    os.kill(os.getpid(), 9)  # Colab restarts the runtime automatically
else:
    import logging
    import pyhealth
    logging.getLogger("pyhealth").propagate = False  # print each log line once
    print("PyHealth is installed and ready.")'''
INSTALL_NOTE = (
    "These tutorials use features from the upcoming PyHealth 2.1 release, so "
    "for now we install the latest version straight from GitHub.\n\n"
    "**On Colab this takes two clicks of Runtime > Run all.** The first run "
    "installs PyHealth and restarts the runtime, because Colab has already "
    "loaded older versions of some packages (such as numpy). Colab then says "
    '"Your session crashed": that is expected. Choose Run all again and the '
    "tutorial runs through. On your own machine, install PyHealth once "
    '(`pip install "pyhealth[xgboost]"`) and this cell does nothing.'
)

SERIES = [
    ("00", "Quickstart: your first clinical prediction model"),
    ("01", "Datasets: patients, events and your own data"),
    ("02", "Tasks and processors: from patients to model-ready samples"),
    ("03", "Training and evaluating models properly"),
    ("04", "Choosing a model: from logistic regression to XGBoost"),
    ("05", "Interpreting predictions"),
    ("06", "Medical codes: lookups, mappings and groupers"),
    ("07", "Clinical text: classification and medical coding"),
    ("08", "Contributing a dataset or task to PyHealth"),
]


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str, tags: tuple[str, ...] = ()) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(textwrap.dedent(text).strip("\n"))
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def header(number: str, title: str, learn: list[str], minutes: int, prereq: str) -> list:
    lines = [
        f"# PyHealth Tutorial {number}: {title}",
        "",
        "**What you will learn**",
        *[f"- {x}" for x in learn],
        "",
        f"**Time:** about {minutes} minutes on a free Colab CPU runtime.  ",
        f"**Before this:** {prereq}",
        "",
        "All data used here is synthetic or public, so every cell runs without "
        "credentials.",
    ]
    return [
        nbformat.v4.new_markdown_cell("\n".join(lines)),
        md(f"## Setup\n\n{INSTALL_NOTE}"),
        code(INSTALL, tags=("install",)),
    ]


def footer(next_number: str | None) -> list:
    lines = ["## Where to go next", ""]
    if next_number is not None:
        lines += [f"**Next:** Tutorial {next_number}, *{dict(SERIES)[next_number]}*.", ""]
    lines += [
        "The whole series:",
        "",
        series_table(),
        "",
        "Questions or problems? Open an issue at "
        "https://github.com/sunlabuiuc/PyHealth/issues. If PyHealth is useful to "
        "you, a star on GitHub helps the project.",
    ]
    return [nbformat.v4.new_markdown_cell("\n".join(lines))]


def series_table() -> str:
    return "\n".join(f"{n}. {t}" for n, t in SERIES)


def notebook(cells: list) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    for i, cell in enumerate(cells):
        cell.id = f"cell-{i:02d}"  # stable ids keep rebuilds diff-free
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    return nb
