#!/usr/bin/env python3
"""CI guard for the agent skill under skills/.

The skill documents PyHealth's API in prose. Prose does not fail to import, so
without a check it rots silently as the package changes — and an agent that
follows a stale guide writes code that raises, or worse, code that quietly does
nothing.

Checks:

    1. Every relative markdown link resolves to a file that exists.
    2. Every backticked ``pyhealth/...`` path cited in prose exists.
    3. Every guide has frontmatter with ``name`` and ``description``, and its
       name is ``pyhealth-<directory>``.
    4. The routing table in SKILL.md, table-of-contents.md, and guides/ all
       list the same guides — no orphans in either direction.
    5. Every processor name quoted in a schema position is in
       PROCESSOR_REGISTRY (parsed from the @register_processor decorators).
    6. Every model table row in choose-a-model names a class that exists in the
       file the row cites.
    7. Example scripts are statically sane: lowercase ``event_type=`` literals
       and registered schema values.
    8. pyproject.toml still force-includes skills/ into the wheel.

Not checked: whether the examples run. That needs MIMIC data and litdata,
neither available in CI — hence the static pass, which catches the failure modes
that have actually occurred.

Usage:
    python tools/check_skills.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS = REPO_ROOT / "skills"
GUIDES = SKILLS / "guides"

problems: list[str] = []


def fail(where: Path | str, msg: str) -> None:
    rel = where.relative_to(REPO_ROOT) if isinstance(where, Path) else where
    problems.append(f"{rel}: {msg}")


def markdown_files() -> list[Path]:
    return sorted(p for p in SKILLS.rglob("*.md") if "__pycache__" not in p.parts)


def registered_processors() -> set[str]:
    names = set()
    for py in (REPO_ROOT / "pyhealth" / "processors").glob("*.py"):
        names |= set(re.findall(r'@register_processor\("([a-z_]+)"\)', py.read_text()))
    return names


# ---------------------------------------------------------------- 1. links
def check_links() -> None:
    for md in markdown_files():
        for _, link in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", md.read_text()):
            if link.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (md.parent / link.split("#")[0]).resolve()
            if not target.exists():
                fail(md, f"broken link -> {link}")


# ------------------------------------------------------- 2. cited pyhealth paths
def check_cited_paths() -> None:
    # `pyhealth/foo/bar.py` or `pyhealth/foo/bar.py:123`
    pattern = re.compile(r"`(pyhealth/[\w./-]+\.(?:py|yaml|rst|md))(?::\d+)?`")
    for md in markdown_files():
        for cited in set(pattern.findall(md.read_text())):
            if not (REPO_ROOT / cited).exists():
                fail(md, f"cites missing path {cited}")


# -------------------------------------------------------------- 3. frontmatter
def guide_frontmatter(skill_md: Path) -> dict[str, str]:
    text = skill_md.read_text()
    if not text.startswith("---\n"):
        return {}
    _, _, rest = text.partition("---\n")
    block, sep, _ = rest.partition("\n---")
    if not sep:
        return {}
    meta = {}
    for line in block.split("\n"):
        key, colon, value = line.partition(":")
        if colon and not key.startswith(" "):
            meta[key.strip()] = value.strip()
    return meta


def check_frontmatter() -> None:
    for skill_md in [SKILLS / "SKILL.md", *sorted(GUIDES.glob("*/SKILL.md"))]:
        meta = guide_frontmatter(skill_md)
        if not meta:
            fail(skill_md, "missing or malformed YAML frontmatter")
            continue
        for key in ("name", "description"):
            if not meta.get(key):
                fail(skill_md, f"frontmatter missing {key!r}")
        expected = (
            "pyhealth"
            if skill_md.parent == SKILLS
            else f"pyhealth-{skill_md.parent.name}"
        )
        if meta.get("name") not in (None, expected):
            fail(skill_md, f"frontmatter name is {meta['name']!r}, expected {expected!r}")


# ------------------------------------------------------------ 4. manifest sync
def check_manifest() -> None:
    on_disk = {p.name for p in GUIDES.iterdir() if p.is_dir()}
    router = set(re.findall(r"\(guides/([\w-]+)/SKILL\.md\)", (SKILLS / "SKILL.md").read_text()))
    toc = set(re.findall(r"^### `([\w-]+)`", (SKILLS / "table-of-contents.md").read_text(), re.M))

    for label, listed in (("SKILL.md routing table", router), ("table-of-contents.md", toc)):
        for missing in sorted(on_disk - listed):
            fail("skills/" + label, f"guide {missing!r} exists on disk but is not listed")
        for extra in sorted(listed - on_disk):
            fail("skills/" + label, f"lists {extra!r}, which has no directory")


# ------------------------------------------------------- 5. processor names
# Only look inside an actual `input_schema`/`output_schema` dict literal. A
# looser "any string-to-string mapping" regex matches YAML configs and metric
# dicts too, and reports them as bogus processor names.
SCHEMA_DICT = re.compile(r"(?:input|output)_schema[^=]*=\s*\{(.*?)\}", re.S)
SCHEMA_VALUE = re.compile(r'["\'][\w]+["\']\s*:\s*["\']([a-z_]+)["\']')


def schema_processor_names(text: str) -> set[str]:
    return {
        name
        for body in SCHEMA_DICT.findall(text)
        for name in SCHEMA_VALUE.findall(body)
    }


def check_processor_names() -> None:
    registry = registered_processors()
    if not registry:
        fail("tools/check_skills.py", "found no @register_processor decorators")
        return
    for md in markdown_files():
        for name in schema_processor_names(md.read_text()):
            if name not in registry:
                fail(md, f"schema uses unregistered processor {name!r}")


# ---------------------------------------------------------- 6. model table
MODEL_ROW = re.compile(r"^\| `(\w+)` \| ([\w]+\.py) \|", re.M)


def check_model_table() -> None:
    md = GUIDES / "choose-a-model" / "SKILL.md"
    if not md.exists():
        return
    for cls, filename in MODEL_ROW.findall(md.read_text()):
        source = REPO_ROOT / "pyhealth" / "models" / filename
        if not source.exists():
            fail(md, f"model row cites missing file pyhealth/models/{filename}")
        elif f"class {cls}" not in source.read_text():
            fail(md, f"pyhealth/models/{filename} has no `class {cls}`")


# ------------------------------------------------------------ 7. examples
def check_examples() -> None:
    registry = registered_processors()
    for py in sorted(SKILLS.rglob("examples/**/*.py")):
        text = py.read_text()
        for literal in set(re.findall(r'event_type=["\']([^"\']+)["\']', text)):
            if literal != literal.lower():
                fail(
                    py,
                    f"event_type={literal!r} is not lowercase — get_events returns [] "
                    "silently (see skills/guides/use-a-dataset/SKILL.md)",
                )
        for name in schema_processor_names(text):
            if name not in registry:
                fail(py, f"schema uses unregistered processor {name!r}")


# ------------------------------------------------------------ 8. wheel bundle
def check_wheel_bundle() -> None:
    """The one silent failure mode: reorganizing the build config un-ships the
    skill, and nothing else notices until a pip user has no guides."""
    pyproject = REPO_ROOT / "pyproject.toml"
    expected = '"skills" = "pyhealth/skills/_bundle"'
    if expected not in pyproject.read_text():
        fail(
            pyproject,
            f"missing {expected} under "
            "[tool.hatch.build.targets.wheel.force-include] — the skill would "
            "not ship in the wheel",
        )


def main() -> int:
    if not SKILLS.is_dir():
        print(f"error: no skills directory at {SKILLS}", file=sys.stderr)
        return 1

    check_links()
    check_cited_paths()
    check_frontmatter()
    check_manifest()
    check_processor_names()
    check_model_table()
    check_examples()
    check_wheel_bundle()

    if problems:
        print(f"{len(problems)} problem(s) in skills/:\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    guides = sum(1 for p in GUIDES.iterdir() if p.is_dir())
    print(f"skills/ OK — {guides} guides, {len(markdown_files())} markdown files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
