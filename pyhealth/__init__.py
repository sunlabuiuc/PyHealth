import logging
import os
from pathlib import Path
import sys
import tempfile
import warnings

__version__ = "2.0.0"

def _ensure_cache_path(preferred_path: str, fallback_path: str) -> str:
    """Create the cache directory, falling back if the preferred path is not writable."""
    errors = []
    for path in dict.fromkeys((preferred_path, fallback_path)):
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError as exc:
            errors.append((path, exc))
    error_msg = "; ".join(f"{path}: {exc}" for path, exc in errors)
    raise OSError(f"Unable to initialize PyHealth cache path. {error_msg}")


_DEFAULT_CACHE_PATH = os.path.join(str(Path.home()), ".cache", "pyhealth")
_FALLBACK_CACHE_PATH = os.path.join(tempfile.gettempdir(), "pyhealth-cache")
_CONFIGURED_CACHE_PATH = os.environ.get("PYHEALTH_BASE_CACHE")
_PREFERRED_CACHE_PATH = _CONFIGURED_CACHE_PATH or _DEFAULT_CACHE_PATH

# package-level cache path
BASE_CACHE_PATH = _ensure_cache_path(_PREFERRED_CACHE_PATH, _FALLBACK_CACHE_PATH)
if BASE_CACHE_PATH != _PREFERRED_CACHE_PATH:
    warnings.warn(
        "Falling back to a temporary cache directory for PyHealth because the "
        f"configured cache path is not writable: {_PREFERRED_CACHE_PATH}",
        RuntimeWarning,
        stacklevel=2,
    )

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not any(getattr(handler, "_pyhealth_handler", False) for handler in logger.handlers):
    handler = logging.StreamHandler(sys.stdout)
    handler._pyhealth_handler = True
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
