"""Helpers shared by the medical imaging dataset loaders."""

from __future__ import annotations

import logging
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

from ..utils import download_file

logger = logging.getLogger(__name__)


def _resolve_metadata_dir(cache_dir: str | Path | None, dataset_name: str) -> Path:
    """Return the directory where a dataset writes its metadata CSV."""
    if cache_dir is None:
        from pyhealth import BASE_CACHE_PATH

        base = Path(BASE_CACHE_PATH)
    else:
        base = Path(cache_dir)
    return base / "metadata" / dataset_name


def _absolute_path(root: str | Path, path: str | Path) -> str:
    """Resolve a possibly relative dataset path to an absolute path string."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(root) / candidate
    return str(candidate.resolve())


def _strip_nii_suffix(path: str | Path) -> str:
    """Return a case ID from a NIfTI path, removing ``.nii``/``.nii.gz``."""
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _safe_member_path(destination: Path, member: str) -> Path:
    """Resolve an archive member and reject paths outside destination."""
    target = (destination / member).resolve()
    root = destination.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Unsafe archive member: {member!r}")
    return target


def _extract_tar(archive: Path, destination: Path) -> None:
    """Extract a tar archive after checking every member path."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            _safe_member_path(destination, member.name)
        handle.extractall(destination, filter="data")


def _extract_zip(archive: Path, destination: Path) -> None:
    """Extract a zip archive after checking every member path."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.namelist():
            _safe_member_path(destination, member)
        handle.extractall(destination)



def _download_archives(
    urls: Sequence[str],
    destination: str | Path,
    archive_type: str,
    expected: Sequence[str],
    extract_to_parent: bool = False,
) -> None:
    """Download archives, extract them, and verify expected paths.

    This is the shared download path for imaging datasets. It deliberately
    uses :func:`pyhealth.utils.download_file` instead of implementing another
    downloader.

    Args:
        urls: Archive URLs to download.
        destination: Directory that should contain ``expected`` after
            extraction.
        archive_type: Either ``"tar"`` or ``"zip"``.
        expected: Relative paths that must exist after extraction.
        extract_to_parent: Extract into the parent of ``destination`` instead
            of ``destination`` itself. This is needed for archives that
            already contain the top-level dataset directory.

    Raises:
        ValueError: If ``archive_type`` is unsupported.
        RuntimeError: If an expected path is missing after extraction.
    """
    if archive_type not in {"tar", "zip"}:
        raise ValueError(f"Unsupported archive type: {archive_type!r}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extract_to = destination.parent if extract_to_parent else destination
    for index, url in enumerate(urls):
        suffix = f"_{index}" if len(urls) > 1 else ""
        archive = (
            destination.parent
            / f"{destination.name}{suffix}.{archive_type}"
        )
        logger.info("Downloading %s to %s", url, archive)
        download_file(url, str(archive))
        logger.info("Extracting %s to %s", archive, extract_to)
        if archive_type == "tar":
            _extract_tar(archive, extract_to)
        else:
            _extract_zip(archive, extract_to)
        archive.unlink(missing_ok=True)

    missing = [name for name in expected if not (destination / name).exists()]
    if missing:
        raise RuntimeError(
            f"Downloaded archives are missing expected paths under "
            f"{destination}: {missing}"
        )
