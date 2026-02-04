#!/usr/bin/env python3
"""
Upload validation script for PyHealth.

This script ensures safe uploads to PyPI by:
1. Checking that only one wheel exists in the dist directory
2. Verifying the version doesn't already exist on PyPI
3. Confirming the wheel version matches the pyproject.toml version
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from packaging import version

ROOT = os.path.dirname(os.path.dirname(__file__))
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
DIST_DIR = os.path.join(ROOT, "target", "dist")


def extract_version_from_pyproject() -> str:
    """Extract version from pyproject.toml"""
    with open(PYPROJECT, "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r"^version\s*=\s*\"([^\"]+)\"", content, flags=re.M)
    if not m:
        msg = "Error: version field not found in pyproject.toml"
        print(msg, file=sys.stderr)
        sys.exit(1)
    return m.group(1)


def get_pypi_versions(project: str = "pyhealth") -> list[str]:
    """Get all versions currently on PyPI, sorted by version number"""
    url = f"https://pypi.org/pypi/{project}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310
            data = json.load(resp)
        versions = list(data.get("releases", {}).keys())
        # Sort versions using packaging.version for proper semantic sorting
        return sorted(versions, key=lambda v: version.parse(v))
    except Exception as e:
        print(f"Warning: Could not fetch PyPI versions: {e}")
        return []


def get_latest_pypi_version(project: str = "pyhealth") -> str | None:
    """Get the latest version from PyPI"""
    versions = get_pypi_versions(project)
    return versions[-1] if versions else None


def parse_version_components(v: str) -> tuple[int, int, int, str | None, int | None]:
    """Parse version into components (major, minor, patch, pre_type, pre_num)"""
    # Handle alpha versions like 2.0a05
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?(?:(a|b|rc)(\d+))?$", v)
    if not match:
        raise ValueError(f"Invalid version format: {v}")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3)) if match.group(3) else 0
    pre_type = match.group(4)  # 'a', 'b', 'rc', or None
    pre_num = int(match.group(5)) if match.group(5) else None

    return major, minor, patch, pre_type, pre_num


def is_next_version(current: str, proposed: str) -> bool:
    """Check if proposed version is exactly +1 from current version"""
    try:
        cur_parts = parse_version_components(current)
        prop_parts = parse_version_components(proposed)
        cur_major, cur_minor, cur_patch, cur_pre_type, cur_pre_num = cur_parts
        (prop_major, prop_minor, prop_patch, prop_pre_type, prop_pre_num) = prop_parts

        # For alpha versions (most common case)
        if cur_pre_type == "a" and prop_pre_type == "a":
            # Same major.minor, alpha number should be +1
            if (
                cur_major == prop_major
                and cur_minor == prop_minor
                and cur_patch == prop_patch
                and prop_pre_num == cur_pre_num + 1
            ):
                return True

        # For alpha to release transition (e.g., 2.0a05 -> 2.0.0)
        elif cur_pre_type == "a" and prop_pre_type is None:
            if cur_major == prop_major and cur_minor == prop_minor and prop_patch == 0:
                return True

        # For release to release (patch increment)
        elif cur_pre_type is None and prop_pre_type is None:
            if (
                cur_major == prop_major
                and cur_minor == prop_minor
                and prop_patch == cur_patch + 1
            ):
                return True

        # For minor version bump
        elif cur_pre_type is None and prop_pre_type is None:
            if (
                cur_major == prop_major
                and prop_minor == cur_minor + 1
                and prop_patch == 0
            ):
                return True

        # For major version bump
        elif cur_pre_type is None and prop_pre_type is None:
            if prop_major == cur_major + 1 and prop_minor == 0 and prop_patch == 0:
                return True

        return False
    except Exception:
        return False


def get_wheel_files() -> list[Path]:
    """Get all wheel files in the dist directory"""
    dist_path = Path(DIST_DIR)
    if not dist_path.exists():
        return []
    return list(dist_path.glob("*.whl"))


def extract_version_from_wheel(wheel_path: Path) -> str:
    """Extract version from wheel filename"""
    # Wheel format: {name}-{version}-{python tag}-{abi tag}-{platform}.whl
    filename = wheel_path.name
    # Remove .whl extension and split by '-'
    parts = filename[:-4].split("-")
    if len(parts) < 2:
        raise ValueError(f"Invalid wheel filename format: {filename}")

    # The version is the second part
    return parts[1]


def validate_upload() -> bool:
    """
    Validate that the upload is safe to proceed.
    Returns True if safe, False otherwise.
    """
    errors = []
    warnings = []

    # 1. Check pyproject.toml version
    try:
        pyproject_version = extract_version_from_pyproject()
        print(f"pyproject.toml version: {pyproject_version}")
    except Exception as e:
        errors.append(f"Could not read version from pyproject.toml: {e}")
        return False

    # 2. Check wheel files
    wheel_files = get_wheel_files()
    if not wheel_files:
        errors.append(f"No wheel files found in {DIST_DIR}")
        return False

    if len(wheel_files) > 1:
        errors.append(f"Multiple wheel files found in {DIST_DIR}:")
        for wheel in wheel_files:
            errors.append(f"  - {wheel.name}")
        errors.append("This could result in uploading multiple versions!")
        return False

    wheel_file = wheel_files[0]
    print(f"Found wheel: {wheel_file.name}")

    # 3. Check wheel version matches pyproject.toml
    try:
        wheel_version = extract_version_from_wheel(wheel_file)
        print(f"Wheel version: {wheel_version}")

        if wheel_version != pyproject_version:
            errors.append(
                f"Version mismatch: pyproject.toml has {pyproject_version}, "
                f"wheel has {wheel_version}"
            )
            return False
    except Exception as e:
        errors.append(f"Could not extract version from wheel: {e}")
        return False

    # 4. Check PyPI for existing versions and enforce strict versioning
    pypi_versions = get_pypi_versions()
    if pypi_versions:
        # Check if version already exists
        if pyproject_version in pypi_versions:
            msg = f"Version {pyproject_version} already exists on PyPI!"
            errors.append(msg)
            return False

        # Get latest version and check if new version is exactly +1
        latest_version = pypi_versions[-1]  # Already sorted
        print(f"Latest PyPI version: {latest_version}")

        if not is_next_version(latest_version, pyproject_version):
            errors.append(
                f"Version {pyproject_version} is not exactly +1 from "
                f"latest PyPI version {latest_version}. "
                f"Only incremental versions are allowed to prevent conflicts."
            )
            return False

        msg = (
            f"Version {pyproject_version} is correctly +1 from "
            f"{latest_version} - safe to upload"
        )
        print(msg)
    else:
        msg = "Could not check PyPI versions - required for safe uploads"
        errors.append(msg)
        return False

    # Print any warnings
    for warning in warnings:
        print(f"Warning: {warning}")

    # If we get here, everything looks good
    if not errors:
        print("✓ Upload validation passed - safe to proceed")
        return True

    # Print errors
    for error in errors:
        print(f"Error: {error}", file=sys.stderr)

    return False


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate PyHealth upload safety")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check, don't exit with error code on failure",
    )

    args = parser.parse_args()

    is_valid = validate_upload()

    if not is_valid and not args.check_only:
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
