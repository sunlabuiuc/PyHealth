"""Install the PyHealth agent skills into a project so coding agents find them.

Three hosts, three conventions:

- Claude Code reads ``<target>/.claude/skills/<name>/SKILL.md``. We link (or
  copy) the ``skills/`` directory there as ``pyhealth``.
- Codex and most agent harnesses read ``<target>/AGENTS.md``.
- GitHub Copilot reads ``<target>/.github/copilot-instructions.md``.

For the latter two we append a marked block pointing at the skill. Re-running
replaces the block in place rather than duplicating it, so this is safe to run
repeatedly and safe to run against a file that already has other content.

Usage::

    python -m pyhealth.skills                       # install into the cwd
    python -m pyhealth.skills --target ../app       # into another project
    python -m pyhealth.skills --guide define-a-task # one guide, standalone
    python -m pyhealth.skills --copy                # copy instead of symlinking
    python -m pyhealth.skills --uninstall           # remove what we added

From a git checkout the same CLI is available as ``python
tools/install_skills.py``, which loads this module by path so it works before
anything is pip-installed.

Standard library only, and it must not import ``pyhealth`` — see above.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path

SKILL_NAME = "pyhealth"

# Two layouts hold the skill content. In a wheel it is force-included next to
# this file as ``_bundle``; in a git checkout (including ``pip install -e .``)
# it stays at the repo root as ``skills/``.
_BUNDLE = Path(__file__).resolve().parent / "_bundle"
_CHECKOUT = Path(__file__).resolve().parents[2] / "skills"

BEGIN = "<!-- BEGIN pyhealth-skills -->"
END = "<!-- END pyhealth-skills -->"

POINTER_TARGETS = (
    Path("AGENTS.md"),
    Path(".github") / "copilot-instructions.md",
)


def skill_root() -> Path | None:
    """Directory holding ``SKILL.md``, or ``None`` if the skill did not ship."""
    for candidate in (_BUNDLE, _CHECKOUT):
        if (candidate / "SKILL.md").is_file():
            return candidate
    return None


def _block(skill_path: str) -> str:
    return f"""{BEGIN}
## PyHealth skill

Before any PyHealth work — loading a dataset, defining a task, choosing
processors or a model, training, evaluating, calibrating, interpreting, mapping
medical codes, or contributing a module upstream — read `{skill_path}`.

It is a router: it decomposes the request into the PyHealth subsystems involved,
then points at the specific guides under `guides/` that cover them. Read only
those. Do not write pipeline code before a spec is confirmed, and do not write
package code before the target file list is agreed.
{END}
"""


def _rel(path: Path, start: Path) -> str:
    """Path relative to ``start`` when it stays inside it, absolute otherwise.

    A relative path that climbs out of the target project is worse than an
    absolute one — it is unreadable and breaks the moment the file moves.
    """
    try:
        rel = os.path.relpath(path, start)
    except ValueError:  # different drives on Windows
        return str(path)
    return str(path) if rel.startswith("..") else rel


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _write_block(file: Path, block: str) -> str:
    """Insert or replace the marked block in ``file``. Returns a status word."""
    existing = file.read_text(encoding="utf-8") if file.exists() else ""

    if BEGIN in existing and END in existing:
        head, _, rest = existing.partition(BEGIN)
        _, _, tail = rest.partition(END)
        updated = head + block.rstrip("\n") + tail
        status = "updated"
    else:
        prefix = existing.rstrip("\n")
        updated = (prefix + "\n\n" if prefix else "") + block
        status = "appended"

    if updated == existing:
        return "unchanged"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(updated, encoding="utf-8")
    return status


def _strip_block(file: Path) -> str:
    if not file.exists():
        return "absent"
    existing = file.read_text(encoding="utf-8")
    if BEGIN not in existing or END not in existing:
        return "unchanged"
    head, _, rest = existing.partition(BEGIN)
    _, _, tail = rest.partition(END)
    updated = (head.rstrip("\n") + "\n" + tail.lstrip("\n")).strip("\n")
    if updated:
        file.write_text(updated + "\n", encoding="utf-8")
        return "stripped"
    file.unlink()
    return "removed"


def _same_tree(a: Path, b: Path) -> bool:
    """Whether two directories hold the same files with the same contents."""
    names_a = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    names_b = sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    if names_a != names_b:
        return False
    _, mismatch, errors = filecmp.cmpfiles(a, b, names_a, shallow=False)
    return not mismatch and not errors


def _should_copy(src: Path, copy: bool) -> bool:
    """Whether to copy rather than symlink.

    A symlink keeps working as the skill is edited, and is right whenever the
    source is a checkout the user controls. Linking into site-packages is not:
    that path dies with the virtualenv, and would be committed as a
    machine-specific artifact. So the wheel bundle is always copied.
    """
    return copy or _inside(src, _BUNDLE.parent)


def _link_skill(target: Path, copy: bool, src: Path, name: str) -> str:
    dest = target / ".claude" / "skills" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy = _should_copy(src, copy)

    if dest.is_symlink() or dest.exists():
        if not copy and dest.is_symlink() and dest.resolve() == src:
            return f"unchanged  {_rel(dest, target)}"
        if copy and not dest.is_symlink() and dest.is_dir() and _same_tree(src, dest):
            return f"unchanged  {_rel(dest, target)}"
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    if copy:
        shutil.copytree(src, dest)
        return f"copied     {_rel(dest, target)}"
    dest.symlink_to(src, target_is_directory=True)
    return f"linked     {_rel(dest, target)} -> {src}"


def _unlink_skill(target: Path, name: str) -> str:
    dest = target / ".claude" / "skills" / name
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
        return f"removed    {_rel(dest, target)}"
    if dest.is_dir():
        shutil.rmtree(dest)
        return f"removed    {_rel(dest, target)}"
    return f"absent     {_rel(dest, target)}"


def main(argv: list[str] | None = None, default_target: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        type=Path,
        default=default_target or Path.cwd(),
        help="project directory to install into (default: the current directory)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copy the skill directory instead of symlinking it",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="remove the skill link and the marked pointer blocks",
    )
    parser.add_argument(
        "--guide",
        metavar="NAME",
        help=(
            "install a single guide standalone (e.g. define-a-task) instead of "
            "the whole router; skips the AGENTS.md / Copilot pointer blocks"
        ),
    )
    args = parser.parse_args(argv)

    root = skill_root()
    if root is None:
        print(
            f"error: skill content not found (looked in {_BUNDLE} and {_CHECKOUT})",
            file=sys.stderr,
        )
        return 1

    if args.guide:
        src = root / "guides" / args.guide
        name = f"{SKILL_NAME}-{args.guide}"
        if not (src / "SKILL.md").is_file():
            available = sorted(p.name for p in (root / "guides").iterdir() if p.is_dir())
            print(f"error: no guide named {args.guide!r}", file=sys.stderr)
            print("available: " + ", ".join(available), file=sys.stderr)
            return 1
    else:
        src, name = root, SKILL_NAME

    target = args.target.expanduser().resolve()
    if args.uninstall:
        # Creating the directory we are about to clean out would be absurd.
        if not target.is_dir():
            print(f"absent     {target}")
            return 0
        print(_unlink_skill(target, name))
        if not args.guide:
            for rel in POINTER_TARGETS:
                print(f"{_strip_block(target / rel):<10} {rel}")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    print(_link_skill(target, args.copy, src, name))

    # A symlinked install is readable at its source (`skills/SKILL.md`). A copy
    # is not — the source may be site-packages, which is no place to send an
    # agent — so point at the copy we just made.
    readable = (
        target / ".claude" / "skills" / name
        if _should_copy(src, args.copy)
        else src
    )
    skill_path = os.path.join(_rel(readable, target), "SKILL.md")
    if not args.guide:
        # A standalone guide is discovered by its own frontmatter; only the
        # router earns a pointer in the flat-prose agent files.
        block = _block(skill_path)
        for rel in POINTER_TARGETS:
            print(f"{_write_block(target / rel, block):<10} {rel}")

    print(f"\nSkill installed. Agents should read: {skill_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
