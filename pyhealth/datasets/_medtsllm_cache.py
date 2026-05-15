"""Fingerprint-based ``.npz`` preprocessing cache for MedTsLLM datasets.

Internal helper for the LUDB / MIT-BIH / BIDMC loaders. Provides
``load_or_build`` so preprocessing runs the expensive wfdb-decode path
exactly once per raw file. Subsequent calls load from an ``.npz``
next to the raw file (or wherever the dataset chooses to cache).
Invalidation is keyed on raw-file stats + preprocessing params via
:func:`compute_fingerprint`.

Author: Anton Barchukov
"""

import hashlib
import json
import os
from typing import Callable

import numpy as np

_FINGERPRINT_KEY = "_cache_fingerprint"


def compute_fingerprint(raw_paths: list[str], params: dict) -> str:
    """Return a stable SHA-256 fingerprint for cache invalidation.

    Combines raw-file ``(path, mtime_ns, size)`` tuples with the
    preprocessing params (JSON-serializable) into a single hash. A
    change in any input — including editing a raw file or flipping a
    param — flips the fingerprint.

    Args:
        raw_paths: Absolute paths to the raw files whose decoded form
            is being cached. Order-insensitive.
        params: Preprocessing params that affect the cached arrays
            (e.g. ``{"trim": True, "downsample_factor": 3}``).

    Returns:
        64-char hex digest string.
    """
    h = hashlib.sha256()
    for path in sorted(raw_paths):
        st = os.stat(path)
        h.update(path.encode("utf-8"))
        h.update(b"|")
        h.update(str(st.st_mtime_ns).encode("ascii"))
        h.update(b"|")
        h.update(str(st.st_size).encode("ascii"))
        h.update(b"\n")
    h.update(json.dumps(params, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def load_or_build(
    cache_path: str,
    fingerprint: str,
    builder: Callable[[], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Load cached arrays if the fingerprint matches, else build + write.

    Args:
        cache_path: Target ``.npz`` path. Parent dirs are created.
        fingerprint: Expected fingerprint string. Mismatch triggers a
            rebuild.
        builder: Zero-arg callable returning ``{name: ndarray}``. Only
            invoked on cache miss.

    Returns:
        Dict of arrays — either freshly built or restored from disk.
    """
    cached = _try_load(cache_path, fingerprint)
    if cached is not None:
        return cached

    arrays = builder()
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = dict(arrays)
    payload[_FINGERPRINT_KEY] = np.array([fingerprint])
    np.savez(cache_path, allow_pickle=False, **payload)
    return arrays


def _try_load(
    cache_path: str, fingerprint: str
) -> dict[str, np.ndarray] | None:
    """Return cached arrays iff the file exists, parses, and matches."""
    if not os.path.exists(cache_path):
        return None
    try:
        npz = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None
    try:
        stored = str(npz[_FINGERPRINT_KEY][0])
    except (KeyError, IndexError):
        return None
    if stored != fingerprint:
        return None
    return {
        key: np.array(npz[key])
        for key in npz.files
        if key != _FINGERPRINT_KEY
    }
