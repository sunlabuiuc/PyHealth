"""Skip helper for tests that need ogb (MoleRec)."""

import importlib.metadata as importlib_metadata

from packaging import version

OGB_MIN_VERSION = "1.3.5"


def require_ogb(testcase):
    """Skips ``testcase`` unless ogb>=1.3.5 is installed and importable.

    Examples:
        >>> import unittest
        >>> class T(unittest.TestCase):
        ...     def setUp(self):
        ...         require_ogb(self)
    """
    try:
        installed = importlib_metadata.version("ogb")
    except importlib_metadata.PackageNotFoundError:
        testcase.skipTest("ogb package not available")
    if version.parse(installed) < version.parse(OGB_MIN_VERSION):
        testcase.skipTest(f"ogb>={OGB_MIN_VERSION} required, but found {installed}")
    try:
        from ogb.utils import smiles2graph  # noqa: F401
    except Exception as exc:
        testcase.skipTest(f"ogb import failed: {exc}")
