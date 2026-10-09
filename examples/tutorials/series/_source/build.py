"""Builds the tutorial notebooks from the Python sources in tutorials/.

Edit the tutorials/tNN_*.py files, not the .ipynb files, then rebuild:

    python examples/tutorials/series/_source/build.py            # write all notebooks
    python examples/tutorials/series/_source/build.py t03         # only matching ones
    python examples/tutorials/series/_source/build.py t03 --run   # also execute them

``--run`` executes each notebook top to bottom with the current environment's
PyHealth (the install cell is skipped) and stops at the first error. Set
TUTORIAL_KERNEL to choose the Jupyter kernel (default "python3").
"""

import importlib
import os
import sys
import time
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
OUT = HERE.parent
sys.path.insert(0, str(HERE))
from nbkit import notebook  # noqa: E402


def build(name: str) -> Path:
    mod = importlib.import_module(f"tutorials.{name}")
    nb = notebook(mod.cells())
    path = OUT / mod.FILENAME
    nbformat.write(nb, path)
    return path


def run(path: Path, workdir: Path) -> bool:
    from nbclient import NotebookClient

    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if "install" in cell.metadata.get("tags", []):
            cell.source = "# (install skipped)"
    workdir.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        nb,
        kernel_name=os.environ.get("TUTORIAL_KERNEL", "python3"),
        timeout=3600,
        allow_errors=True,
        resources={"metadata": {"path": str(workdir)}},
    )
    start = time.time()
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            client.execute_cell(cell, i)
            errors = [o for o in cell.outputs if o.get("output_type") == "error"]
            if errors:
                print(f"  cell {i}: {errors[0]['ename']}: {errors[0]['evalue'][:300]}")
                return False
    print(f"  ok in {time.time() - start:.0f}s")
    return True


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = sorted(p.stem for p in (HERE / "tutorials").glob("t*.py"))
    if wanted:
        names = [n for n in names if any(n.startswith(w) for w in wanted)]
    ok = True
    for name in names:
        path = build(name)
        print(f"wrote {path.relative_to(OUT.parent.parent)}")
        if "--run" in sys.argv:
            ok &= run(path, Path("tutorial_runs") / path.stem)
    sys.exit(0 if ok else 1)
