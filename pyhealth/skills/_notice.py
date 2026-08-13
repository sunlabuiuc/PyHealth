"""Tell a coding agent, once, that the PyHealth skill exists.

``pip install pyhealth`` cannot print anything — PyPI serves a prebuilt wheel
and pip only unzips it, so no code of ours runs at install time. The first
``import pyhealth`` is therefore the earliest moment we can say anything, and in
an agent session it is usually the very next command after the install.

Deliberately narrow: it writes nothing, prints only when an agent harness is
detected, goes to stderr rather than stdout, and stops once the skill is
registered in the current project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ._installer import BEGIN, POINTER_TARGETS, SKILL_NAME, skill_root

#: Set by the agent harnesses that would act on the notice. A human running
#: ``import pyhealth`` in a REPL or a script sees nothing.
AGENT_ENV_VARS = ("CLAUDECODE", "CLAUDE_CODE", "CODEX_SANDBOX", "CURSOR_AGENT")

OPT_OUT_ENV_VAR = "PYHEALTH_NO_SKILL_NOTICE"

_notified = False


def _registered(project: Path) -> bool:
    """True once this project already points an agent at the skill."""
    if (project / ".claude" / "skills" / SKILL_NAME).exists():
        return True
    for rel in POINTER_TARGETS:
        pointer = project / rel
        try:
            if pointer.is_file() and BEGIN in pointer.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def _message(root: Path) -> str:
    return (
        f"[pyhealth] Agent skill available — a router over "
        f"{len(list((root / 'guides').glob('*/SKILL.md')))} PyHealth guides.\n"
        f"  Read:      {root / 'SKILL.md'}\n"
        f"  Register:  python -m pyhealth.skills\n"
        f"  Silence:   export {OPT_OUT_ENV_VAR}=1\n"
    )


def maybe_notify() -> bool:
    """Print the notice if this looks like an agent that has not seen it.

    Returns whether anything was printed. Never raises: a cosmetic notice must
    not be able to break ``import pyhealth``.
    """
    global _notified
    try:
        if _notified or os.environ.get(OPT_OUT_ENV_VAR):
            return False
        if not any(os.environ.get(var) for var in AGENT_ENV_VARS):
            return False
        root = skill_root()
        if root is None or _registered(Path.cwd()):
            return False
        _notified = True
        sys.stderr.write(_message(root))
        return True
    except Exception:  # noqa: BLE001 — a cosmetic notice must never break an import
        return False
