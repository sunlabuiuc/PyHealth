import logging
import os
import sys
import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path


def _resolve_version() -> str:
    """The version of the code actually being imported.

    Read rather than hardcoded: ``tools/bump_version.py`` rewrites only
    pyproject, so a literal here drifts silently — it sat at 2.0.0 through the
    whole 2.0.x line, and docs/conf.py renders this value.

    A pyproject.toml beside the package means we are running from a source
    tree, and it wins: installed-distribution metadata may belong to some
    *other*, older pyhealth in site-packages that this import is shadowing.
    In a wheel there is no pyproject, and the metadata is authoritative.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    try:
        return tomllib.loads(pyproject.read_text())["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass
    try:
        return _pkg_version("pyhealth")
    except PackageNotFoundError:
        return "0.0.0.dev0"


__version__ = _resolve_version()

# package-level cache path
BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
# BASE_CACHE_PATH = "/srv/local/data/pyhealth-cache"
if not os.path.exists(BASE_CACHE_PATH):
    os.makedirs(BASE_CACHE_PATH)

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# One-line stderr pointer at the agent skill, printed only under a coding-agent
# harness and only until the project registers it. Silent for human users.
from pyhealth.skills._notice import maybe_notify

maybe_notify()

