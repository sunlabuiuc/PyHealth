#!/usr/bin/env python3
"""
Smart version bumper for PyHealth.

Supports:
    --alpha-minor   (2.0a04 -> 2.0a05)
    --alpha-major   (2.0a04 -> 2.0a10)
    --minor         Normal release within current minor: 2.0a04 -> 2.0.0
                                    If 2.0.0 exists on PyPI, try 2.0.1, etc.
    --major         Next minor line release: 2.0.1 -> 2.1.0

Notes:
    - Assumes base major version '2' per request; does not enforce but warns.
    - Writes back to pyproject.toml by replacing the version line.
    - Uses two-digit alpha padding when writing (e.g., a05), which is
        accepted by PEP 440 tools (canonical form omits the padding).
    - Normal (non-alpha) releases always write three-part versions: X.Y.Z.
    - Attempts to detect existing releases on PyPI to choose the next patch
        version for --minor; if the network check fails, falls back to X.Y.0.
"""
import argparse
import os
import re
import sys
import json
import urllib.request

ROOT = os.path.dirname(os.path.dirname(__file__))
PYPROJECT = os.path.join(ROOT, "pyproject.toml")


VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"  # release
    r"(?:(?P<pre_l>a|b|rc)(?P<pre_n>\d+))?$",
)


def read_pyproject():
    with open(PYPROJECT, "r", encoding="utf-8") as f:
        return f.read()


def extract_version(text: str) -> str:
    m = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, flags=re.M)
    if not m:
        print("version field not found in pyproject.toml", file=sys.stderr)
        sys.exit(2)
    return m.group(1)


def replace_version(text: str, new_version: str) -> str:
    return re.sub(
        r"^(version\s*=\s*\")[^\"]+(\")",
        rf"\g<1>{new_version}\2",
        text,
        flags=re.M,
    )


def parse_version(v: str):
    m = VERSION_RE.match(v)
    if not m:
        print(f"Unsupported version format: {v}", file=sys.stderr)
        sys.exit(2)
    gd = m.groupdict()
    major = int(gd["major"]) if gd["major"] else 0
    minor = int(gd["minor"]) if gd["minor"] else 0
    patch = int(gd["patch"]) if gd["patch"] else 0
    pre_l = gd["pre_l"]  # 'a' | 'b' | 'rc' | None
    pre_n = int(gd["pre_n"]) if gd["pre_n"] else None
    return major, minor, patch, pre_l, pre_n


def fmt_version(
    major,
    minor,
    patch,
    pre_l,
    pre_n,
    *,
    force_patch: bool = False,
) -> str:
    base = f"{major}.{minor}"
    if force_patch or patch is not None and patch != 0:
        # Always include patch when forced (normal releases), or when non-zero
        base += f".{patch if patch is not None else 0}"
    if pre_l and pre_n is not None:
        # two-digit zero-padded pre number for readability (PEP 440 ok)
        return f"{base}{pre_l}{pre_n:02d}"
    return base


def bump_alpha_minor(v: str) -> str:
    major, minor, patch, pre_l, pre_n = parse_version(v)
    if pre_l not in ("a", None):
        # convert to alpha if not already
        pre_l, pre_n = "a", 0
    if pre_n is None:
        pre_n = 0
    pre_n += 1
    return fmt_version(major, minor, patch, pre_l, pre_n)


def bump_alpha_major(v: str) -> str:
    major, minor, patch, pre_l, pre_n = parse_version(v)
    if pre_l not in ("a", None):
        pre_l, pre_n = "a", 0
    if pre_n is None:
        pre_n = 0
    # next tens bucket (e.g., 4 -> 10, 14 -> 20)
    pre_n = ((pre_n // 10) + 1) * 10
    return fmt_version(major, minor, patch, pre_l, pre_n)


def _get_pypi_versions(project: str = "pyhealth") -> set[str]:
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
            data = json.load(resp)
        return set(data.get("releases", {}).keys())
    except Exception:
        return set()


def bump_minor(v: str, *, project: str = "pyhealth") -> str:
    """
    Normal release within current minor line:
      X.Y[anything] -> X.Y.0 (or .1, .2, ...).
    Uses PyPI to find the next available patch number when possible.
    Never lowers the version if current is already a normal release.
    """
    major, minor, cur_patch, pre_l, _pre_n = parse_version(v)
    existing = _get_pypi_versions(project)
    # if current is already a normal release, start from current patch
    patch = 0 if pre_l else (cur_patch if cur_patch is not None else 0)
    while True:
        candidate = fmt_version(major, minor, patch, None, None, force_patch=True)
        if candidate not in existing:
            return candidate
        patch += 1


def bump_major(v: str) -> str:
    """
    Next minor line release: X.Y.Z -> X.(Y+1).0 (drop any pre-release).
    """
    major, minor, _patch, _pre_l, _pre_n = parse_version(v)
    minor += 1
    patch = 0
    return fmt_version(major, minor, patch, None, None, force_patch=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--alpha-minor", action="store_true")
    g.add_argument("--alpha-major", action="store_true")
    g.add_argument("--minor", action="store_true")
    g.add_argument("--major", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print the new version without writing",
    )
    args = ap.parse_args()

    text = read_pyproject()
    cur = extract_version(text)
    if not str(cur).startswith("2."):
        print(f"Warning: current version '{cur}' not in 2.x as assumed.")

    if args.alpha_minor:
        new = bump_alpha_minor(cur)
    elif args.alpha_major:
        new = bump_alpha_major(cur)
    elif args.minor:
        new = bump_minor(cur)
    elif args.major:
        new = bump_major(cur)
    else:
        raise AssertionError

    if new == cur:
        print("No version change computed.")
        return 0

    if args.dry_run:
        print(f"Would bump version: {cur} -> {new}")
        return 0

    new_text = replace_version(text, new)
    with open(PYPROJECT, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"Bumped version: {cur} -> {new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
