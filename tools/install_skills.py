#!/usr/bin/env python3
"""Install the PyHealth agent skills into a project so coding agents find them.

Usage::

    python tools/install_skills.py                       # into the repo root
    python tools/install_skills.py --target ../app       # into another project
    python tools/install_skills.py --guide define-a-task # one guide, standalone
    python tools/install_skills.py --copy                # copy, don't symlink
    python tools/install_skills.py --uninstall           # remove what we added

The implementation lives in ``pyhealth/skills/_installer.py`` so pip users get
the same CLI as ``python -m pyhealth.skills``. This shim loads that module *by
path*, keeping the promise that the script runs on a bare checkout with nothing
pip-installed. Standard library only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_INSTALLER = REPO_ROOT / "pyhealth" / "skills" / "_installer.py"


def _load():
    """The installer module, loaded from the checkout or from site-packages."""
    if _INSTALLER.is_file():
        spec = importlib.util.spec_from_file_location(
            "_pyhealth_skills_installer", _INSTALLER
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from pyhealth.skills import _installer  # noqa: PLC0415

    return _installer


if __name__ == "__main__":
    # Unlike `python -m pyhealth.skills`, which defaults to the cwd, running
    # from the checkout has always meant "install into this repo".
    raise SystemExit(_load().main(sys.argv[1:], default_target=REPO_ROOT))
