#!/usr/bin/env python3
"""
Smart version bumper for PyHealth.

Supports:
    --alpha-minor   (2.0a7 -> 2.0a8)
    --alpha-major   (2.0a7 -> 2.0a10)
    --minor         Normal release within current minor: 2.0a7 -> 2.0.0
                                    If 2.0.0 exists on PyPI, try 2.0.1, etc.
    --major         Next minor line release: 2.0.1 -> 2.1.0

Notes:
    - Assumes base major version '2' per request; does not enforce but warns.
    - Writes back to pyproject.toml by replacing the version line.
    - Alpha versions use single-digit format (a8, a9, a10), no zero-padding.
    - Normal (non-alpha) releases always write three-part versions: X.Y.Z.
    - Attempts to detect existing releases on PyPI to choose the next patch
        version for --minor; if the network check fails, falls back to X.Y.0.
    - Normalizes versions using PEP 440 rules before checking PyPI.
    - If current version doesn't exist on PyPI, keeps it (no bump).
    - If current version is set too high, auto-corrects to next available.
"""
import argparse
import os
import re
import sys
import json
import urllib.request

try:
    from packaging.version import Version
except ImportError:
    print(
        "Error: packaging library required. " "Install with: pip install packaging",
        file=sys.stderr,
    )
    sys.exit(1)

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
        # No zero-padding for alpha versions (single digit: a8, a9, a10)
        return f"{base}{pre_l}{pre_n}"
    return base


def bump_alpha_minor(v: str, *, project: str = "pyhealth") -> str:
    """
    Bump to next alpha minor version, checking PyPI for existing versions.
    X.Y[.Z]aN -> X.Y[.Z]a(N+1), skipping versions that exist on PyPI.
    """
    major, minor, patch, pre_l, pre_n = parse_version(v)
    if pre_l not in ("a", None):
        # convert to alpha if not already
        pre_l, pre_n = "a", 0
    if pre_n is None:
        pre_n = 0

    existing = _get_pypi_versions(project)
    while True:
        pre_n += 1
        candidate = fmt_version(major, minor, patch, pre_l, pre_n)
        if not _version_exists_on_pypi(candidate, existing):
            return candidate


def bump_alpha_major(v: str, *, project: str = "pyhealth") -> str:
    """
    Bump to next alpha major version (next tens bucket), checking PyPI.
    X.Y[.Z]aN -> X.Y[.Z]a((N//10 + 1)*10), skipping existing versions.
    """
    major, minor, patch, pre_l, pre_n = parse_version(v)
    if pre_l not in ("a", None):
        pre_l, pre_n = "a", 0
    if pre_n is None:
        pre_n = 0

    existing = _get_pypi_versions(project)
    # Start at next tens bucket
    pre_n = ((pre_n // 10) + 1) * 10
    while True:
        candidate = fmt_version(major, minor, patch, pre_l, pre_n)
        if not _version_exists_on_pypi(candidate, existing):
            return candidate
        pre_n += 1


def _get_pypi_versions(project: str = "pyhealth") -> set[str]:
    """Get all versions from PyPI, normalized according to PEP 440."""
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:  # nosec B310
            data = json.load(resp)
        # Normalize all versions from PyPI
        return set(str(Version(v)) for v in data.get("releases", {}).keys())
    except Exception:
        return set()


def _version_exists_on_pypi(version_str: str, existing: set[str]) -> bool:
    """Check if a version exists on PyPI, using PEP 440 normalization."""
    try:
        normalized = str(Version(version_str))
        return normalized in existing
    except Exception:
        # If normalization fails, fall back to string comparison
        return version_str in existing


def _find_minimum_available_version(cur_version: str, existing: set[str]) -> str:
    """
    Find the minimum available version number starting from 1.

    This searches for the first version that doesn't exist on PyPI,
    starting from the lowest number (a1, a2, a3, ...).
    If the current version doesn't exist on PyPI, return it.
    Otherwise, find the next available version after the highest on PyPI.

    Args:
        cur_version: Current version from pyproject.toml
        existing: Set of normalized versions from PyPI

    Returns:
        The minimum available version string
    """
    major, minor, patch, pre_l, pre_n = parse_version(cur_version)

    # Find the highest version on PyPI with same major.minor[.patch] and type
    max_pre_n = -1
    for pypi_ver in existing:
        try:
            p_major, p_minor, p_patch, p_pre_l, p_pre_n = parse_version(pypi_ver)
            # Match same major.minor[.patch] and pre-release type
            if (
                p_major == major
                and p_minor == minor
                and p_patch == patch
                and p_pre_l == pre_l
                and p_pre_n is not None
            ):
                max_pre_n = max(max_pre_n, p_pre_n)
        except Exception:
            continue

    # Start searching from 1 or after the highest on PyPI
    if max_pre_n >= 0:
        # PyPI has versions, start from max + 1
        search_start = max_pre_n + 1
    else:
        # No versions on PyPI yet, start from 1
        search_start = 1

    # Find the first available version
    pre_n = search_start
    while True:
        candidate = fmt_version(major, minor, patch, pre_l, pre_n)
        if not _version_exists_on_pypi(candidate, existing):
            return candidate
        pre_n += 1
        # Safety check to prevent infinite loop
        if pre_n > 1000:
            raise RuntimeError("Could not find available version (checked up to a1000)")


def _find_next_available_version(
    cur_version: str, existing: set[str], bump_type: str
) -> str:
    """
    Find the next available version, starting from the highest on PyPI.

    This handles cases where pyproject.toml version is set too high.
    For example, if PyPI has 2.0a7 and pyproject.toml has 2.0a9,
    this will correctly suggest 2.0a8.

    Args:
        cur_version: Current version from pyproject.toml
        existing: Set of normalized versions from PyPI
        bump_type: One of 'alpha-minor', 'alpha-major', 'minor', 'major'

    Returns:
        Next available version string
    """
    major, minor, patch, pre_l, pre_n = parse_version(cur_version)

    # Find the highest matching version on PyPI
    max_pre_n = -1
    for pypi_ver in existing:
        try:
            p_major, p_minor, p_patch, p_pre_l, p_pre_n = parse_version(pypi_ver)
            # Match same major.minor[.patch] and pre-release type
            if (
                p_major == major
                and p_minor == minor
                and p_patch == patch
                and p_pre_l == pre_l
                and p_pre_n is not None
            ):
                max_pre_n = max(max_pre_n, p_pre_n)
        except Exception:
            continue

    # If we found versions on PyPI, start from the highest
    if max_pre_n >= 0:
        if bump_type == "alpha-minor":
            pre_n = max_pre_n + 1
        elif bump_type == "alpha-major":
            # Next tens bucket after the highest on PyPI
            pre_n = ((max_pre_n // 10) + 1) * 10
        # For minor/major, we use different logic (handled separately)

    # Find next available version
    while True:
        candidate = fmt_version(major, minor, patch, pre_l, pre_n)
        if not _version_exists_on_pypi(candidate, existing):
            return candidate
        pre_n += 1


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
        if not _version_exists_on_pypi(candidate, existing):
            return candidate
        patch += 1


def bump_major(v: str, *, project: str = "pyhealth") -> str:
    """
    Next minor line release: X.Y.Z -> X.(Y+1).0 (drop any pre-release).
    Checks PyPI to ensure the version doesn't already exist.
    """
    major, minor, _patch, _pre_l, _pre_n = parse_version(v)
    minor += 1
    patch = 0

    existing = _get_pypi_versions(project)
    while True:
        candidate = fmt_version(major, minor, patch, None, None, force_patch=True)
        if not _version_exists_on_pypi(candidate, existing):
            return candidate
        patch += 1


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

    # Get existing versions from PyPI
    existing = _get_pypi_versions()

    # Find the minimum available version (handles both cases:
    # 1. Current version doesn't exist -> use it
    # 2. Current version exists or is too high -> find next after highest)
    min_available = _find_minimum_available_version(cur, existing)

    # If current version is already the minimum available, keep it
    if not _version_exists_on_pypi(cur, existing):
        if cur == min_available:
            print(
                f"Current version '{cur}' not found on PyPI. "
                "Keeping current version (no bump needed)."
            )
            return 0
        else:
            print(
                f"Current version '{cur}' is set too high. "
                f"Auto-correcting to minimum available: {min_available}"
            )
            new = min_available
    else:
        # Current version exists on PyPI, so we need to bump
        # Determine bump type
        if args.alpha_minor:
            bump_type = "alpha-minor"
        elif args.alpha_major:
            bump_type = "alpha-major"
        elif args.minor:
            bump_type = "minor"
        elif args.major:
            bump_type = "major"
        else:
            raise AssertionError

        # For alpha versions, use the smart version finder
        # that handles cases where pyproject.toml is set too high
        if bump_type in ("alpha-minor", "alpha-major"):
            new = _find_next_available_version(cur, existing, bump_type)
        else:
            # For minor/major releases, use existing functions
            if args.minor:
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
