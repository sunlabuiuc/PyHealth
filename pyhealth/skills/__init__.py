"""The PyHealth agent skill — a router over 13 guides for coding agents.

The markdown lives at ``skills/`` in a git checkout and is force-included into
the wheel as ``pyhealth/skills/_bundle``; :func:`skill_root` resolves whichever
is present. Register it into a project with ``python -m pyhealth.skills``.

Nothing here imports the rest of ``pyhealth``: :mod:`pyhealth.skills._notice`
runs during ``import pyhealth``, and the installer is loaded by path from
``tools/install_skills.py`` before the package is installed at all.
"""

from ._installer import SKILL_NAME, main, skill_root

__all__ = ["SKILL_NAME", "main", "skill_root"]
