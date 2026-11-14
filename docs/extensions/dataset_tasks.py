"""Add a "Tasks for this dataset" list to every dataset API page.

Task docstrings name the datasets they run on with
``:class:`~pyhealth.datasets.X```. This extension reads those references
from ``pyhealth/tasks/*.py`` (by parsing, not importing) and, while Sphinx
reads ``api/datasets/pyhealth.datasets.X``, inserts a list of the tasks that
reference ``X`` just above its ``autoclass`` directive. A new task therefore
shows up on its dataset's page with no docs edits.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path

try:
    from sphinx.util import logging
except ImportError:  # imported outside a docs build, e.g. by the unit test
    import logging

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).resolve().parents[2] / "pyhealth" / "tasks"
DATASET_REF = re.compile(r":class:`~?pyhealth\.datasets\.(\w+)`")
PAGE = re.compile(r"^api/datasets/pyhealth\.datasets\.(\w+)$")
_TASKS = {}

# Pages that should also list the tasks of related dataset classes: the
# combined MIMIC-IV loader runs its modules' tasks, and MSD task pages
# inherit the tasks written against the shared base loader.
_ALSO = {
    "MIMIC4Dataset": [
        "MIMIC4EHRDataset", "MIMIC4NoteDataset", "MIMIC4CXRDataset",
        "MIMIC4EDDataset",
    ],
    "MSDHippocampusDataset": ["MSDSegmentationDataset"],
    "MSDProstateDataset": ["MSDSegmentationDataset"],
    "MSDSpleenDataset": ["MSDSegmentationDataset"],
}


def tasks_by_dataset():
    """Map dataset class name -> sorted public task class names citing it."""
    found = defaultdict(set)
    for path in sorted(TASKS_DIR.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as err:
            # Its tasks would silently vanish from every dataset page.
            logger.warning("dataset_tasks: skipped %s, could not parse it: %s", path.name, err)
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            for dataset in DATASET_REF.findall(ast.get_docstring(node) or ""):
                found[dataset].add(node.name)
    for page, related in _ALSO.items():
        for dataset in related:
            found[page] |= found.get(dataset, set())
    return {dataset: sorted(tasks) for dataset, tasks in found.items() if tasks}


def add_task_list(app, docname, source):
    match = PAGE.match(docname)
    if not match:
        return
    tasks = _TASKS.get(match.group(1))
    if not tasks:
        return
    # A leading dot resolves the class wherever it is documented
    # (pyhealth.tasks.X or pyhealth.tasks.<module>.X).
    section = "\n".join(
        ["Tasks for this dataset", "----------------------", ""]
        + [f"- :class:`.{task}`" for task in tasks]
        + ["", ""]
    )
    text = source[0]
    at = text.find(".. autoclass::")
    source[0] = text + "\n\n" + section if at < 0 else text[:at] + section + text[at:]


def init_mapping(app):
    _TASKS.clear()
    _TASKS.update(tasks_by_dataset())


def setup(app):
    app.connect("builder-inited", init_mapping)
    app.connect("source-read", add_task_list)
    return {"version": "1.0", "parallel_read_safe": True}
