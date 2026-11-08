"""Shared helpers for datasets that index raw files in a metadata CSV.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: ``CHBMITDataset`` and ``EESM23Dataset`` both write a
    ``<name>-pyhealth.csv`` next to the raw files, or under
    ``~/.cache/pyhealth/<name>/<hash of root>`` when that directory is not
    writable. These helpers hold that logic in one place.
"""

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def metadata_cache_dir(root: str | Path, dataset_key: str) -> Path:
    """Return the home-cache directory used when ``root`` is not writable.

    The directory is keyed by a hash of the resolved root, so two dataset
    roots never share cached metadata or each other's ``signal_file`` paths.

    Args:
        root: Dataset root directory.
        dataset_key: Short dataset name, for example ``"chbmit"``.

    Returns:
        ``~/.cache/pyhealth/<dataset_key>/<hash of root>``.

    Examples:
        >>> from pyhealth.datasets.metadata_csv import metadata_cache_dir
        >>> metadata_cache_dir("/data/eesm23", "eesm23").parent.name
        'eesm23'
    """
    resolved = str(Path(root).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".cache" / "pyhealth" / dataset_key / digest


def metadata_csv_exists(root: str | Path, dataset_key: str, csv_name: str) -> bool:
    """Check whether the metadata CSV is already in ``root`` or in the cache.

    Args:
        root: Dataset root directory.
        dataset_key: Short dataset name, used for the cache directory.
        csv_name: CSV file name, for example ``"chbmit-pyhealth.csv"``.

    Returns:
        True when either location holds the CSV.

    Examples:
        >>> from pyhealth.datasets.metadata_csv import metadata_csv_exists
        >>> metadata_csv_exists("/nonexistent/root", "demo", "demo.csv")
        False
    """
    root = Path(root)
    return (root / csv_name).exists() or (
        metadata_cache_dir(root, dataset_key) / csv_name
    ).exists()


def resolve_metadata_root(root: str | Path, dataset_key: str, csv_name: str) -> str:
    """Pick the directory ``BaseDataset`` should read the CSV from.

    The raw root wins. The home cache is used only when the CSV exists
    there and not in the raw root.

    Args:
        root: Dataset root directory.
        dataset_key: Short dataset name, used for the cache directory.
        csv_name: CSV file name.

    Returns:
        ``str(root)`` or the cache directory path.

    Examples:
        >>> from pyhealth.datasets.metadata_csv import resolve_metadata_root
        >>> resolve_metadata_root("/nonexistent/root", "demo", "demo.csv")
        '/nonexistent/root'
    """
    root_path = Path(root)
    cache_dir = metadata_cache_dir(root_path, dataset_key)
    if not (root_path / csv_name).exists() and (cache_dir / csv_name).exists():
        logger.info("Using cached metadata from %s", cache_dir)
        return str(cache_dir)
    return str(root)


def write_metadata_csv(
    df: pd.DataFrame, root: str | Path, dataset_key: str, csv_name: str
) -> Path:
    """Write the metadata CSV to ``root``, or to the home cache on failure.

    Args:
        df: Metadata rows.
        root: Dataset root directory.
        dataset_key: Short dataset name, used for the cache directory.
        csv_name: CSV file name.

    Returns:
        Path of the CSV that was written.

    Examples:
        >>> import pandas as pd
        >>> from pyhealth.datasets.metadata_csv import write_metadata_csv
        >>> path = write_metadata_csv(  # doctest: +SKIP
        ...     pd.DataFrame({"patient_id": ["a"]}), "/tmp/root", "demo", "demo.csv"
        ... )
    """
    root = Path(root)
    shared_csv = root / csv_name
    try:
        shared_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(shared_csv, index=False)
        logger.info("Wrote metadata to %s", shared_csv)
        return shared_csv
    except (PermissionError, OSError):
        cache_dir = metadata_cache_dir(root, dataset_key)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_csv = cache_dir / csv_name
        df.to_csv(cache_csv, index=False)
        logger.info("Wrote metadata to cache: %s", cache_csv)
        return cache_csv
