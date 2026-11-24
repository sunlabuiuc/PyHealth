"""Which code a run used, and refusing to run on code that cannot be traced.

``code_version(*scripts)`` returns the git commit of the ``pyhealth`` that is
imported, or, when it runs from the Colab code zip, the time the zip was made
(``CODE_VERSION.json``). It also returns whether the tracked files have
uncommitted changes and the SHA-256 of the given scripts in this folder.
``report_code`` prints that version, with a warning when ``pyhealth`` has
uncommitted changes or its version is unknown. ``code_label`` gives the short
name used in default result file names. ``same_code`` tells whether a
checkpoint was made with the same code, and ``rng_state`` / ``set_rng_state``
save and restore the random generators for an exact resume.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

TOOLS = Path(__file__).resolve().parent


def code_version(*scripts: str) -> dict:
    import pyhealth

    out = {"tools_sha256": {
        name: hashlib.sha256((TOOLS / name).read_bytes()).hexdigest()
        for name in ("provenance.py", *scripts)
    }}
    root = Path(pyhealth.__file__).resolve().parents[1]

    def git(*cmd):
        return subprocess.run(["git", "-C", str(root), *cmd], capture_output=True,
                              text=True, check=True).stdout.strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            raise subprocess.CalledProcessError(1, "git")
        out["commit"] = git("rev-parse", "HEAD")
        out["pyhealth_modified"] = bool(
            git("status", "--porcelain", "--untracked-files=no", "--", "pyhealth"))
        out["source"] = "git"
    except (subprocess.CalledProcessError, FileNotFoundError):
        bundle = root / "CODE_VERSION.json"
        if bundle.is_file():
            out["commit"] = None
            out["bundle_created"] = json.loads(bundle.read_text())["created"]
            out["pyhealth_modified"] = False
            out["source"] = "bundle"
        else:
            out.update(commit=None, pyhealth_modified=None, source="unknown")
    return out


def code_label(code: dict) -> str:
    """The first 8 characters of the commit, or "bundle" for the Colab code zip."""
    return str(code["commit"])[:8] if code["commit"] else "bundle"


def same_code(old: dict, new: dict) -> bool:
    """True if two ``code_version`` records describe the same code."""
    return (old.get("commit") == new.get("commit")
            and old.get("bundle_created") == new.get("bundle_created")
            and old.get("tools_sha256") == new.get("tools_sha256"))


def rng_state() -> dict:
    """The torch (CPU and CUDA) and NumPy random states, loadable with
    ``torch.load(..., weights_only=True)``."""
    import numpy as np
    import torch

    kind, keys, pos, has_gauss, cached = np.random.get_state()
    state = {"torch": torch.get_rng_state(),
             "numpy": [kind, torch.from_numpy(keys.astype(np.int64)), int(pos),
                       int(has_gauss), float(cached)]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore what ``rng_state`` saved."""
    import numpy as np
    import torch

    torch.set_rng_state(state["torch"])
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, has_gauss,
                         cached))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(obj, path: Path) -> None:
    """``torch.save`` to a temporary file, then rename, so a disconnect never
    leaves a half-written checkpoint."""
    import os

    import torch

    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def report_code(code: dict) -> None:
    """Print the code version the run records (see ``code_version``)."""
    if code["source"] == "unknown":
        print("warning: the imported pyhealth is neither a git checkout nor the "
              "Colab code zip, so the report has no code version", flush=True)
    when = code.get("bundle_created", "")
    print(f"code: {code['source']} {code['commit'][:8] if code['commit'] else when}"
          f"{' (MODIFIED)' if code['pyhealth_modified'] else ''}", flush=True)


def load_embedding_table(path: str | Path) -> tuple[tuple, dict]:
    """``(index, embeddings)`` of a MedSigLIP ``.npz`` (``--embeddings``) and
    its record for the report: file name, SHA-256, rows and the settings
    stored in it (model, revision, processor). Aggregates only."""
    from pyhealth.datasets.medmod.medsiglip import load_embeddings

    path = Path(path)
    index, embeddings, meta = load_embeddings(path)
    record = {"file": path.name,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "rows": len(index), "meta": meta}
    return (index, embeddings), record
