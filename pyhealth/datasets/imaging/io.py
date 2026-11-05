"""Helpers shared by the medical imaging dataset loaders."""

from __future__ import annotations

from pathlib import Path


def _resolve_metadata_dir(cache_dir: str | Path | None, dataset_name: str) -> Path:
    """Return the directory where a dataset writes its metadata CSV."""
    if cache_dir is None:
        from pyhealth import BASE_CACHE_PATH

        base = Path(BASE_CACHE_PATH)
    else:
        base = Path(cache_dir)
    return base / "metadata" / dataset_name
