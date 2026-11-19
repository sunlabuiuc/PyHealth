"""MedMod's fixed subject lists, downloaded from MedMod's GitHub on first use.

``testset_iv.csv`` and ``valset_iv.csv`` define MedMod's train / val / test
split (test list first, then the validation list among the rest). They are
published in MedMod's repository and, byte-identical, in MedFuse's, so PyHealth
does not ship a copy. The first use downloads each file from a pinned commit
(MedMod first, then MedFuse), checks its SHA-256, and caches it under
``~/.cache/pyhealth/medmod/``; later uses read the cache. To work offline, put
the two files in that folder, or pass their paths to ``MedModSplit`` /
``MedModResources.load``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from pyhealth import BASE_CACHE_PATH

logger = logging.getLogger(__name__)

#: Pinned commits: the download can't change under us.
MEDMOD_COMMIT = "7a47c0b909ffbd36370ff69dd023c34f71d9528a"
MEDFUSE_COMMIT = "3f8b54ec555ac9b8e84ebf98c4854aa6fbd0e4fa"
SOURCES: tuple[str, ...] = (
    f"https://raw.githubusercontent.com/nyuad-cai/MedMod/{MEDMOD_COMMIT}",
    f"https://raw.githubusercontent.com/nyuad-cai/MedFuse/{MEDFUSE_COMMIT}",
)
#: Path of each list inside both repositories.
REPO_PATHS: dict[str, str] = {
    "testset_iv.csv": "mimic4extract/mimic3benchmark/resources/testset_iv.csv",
    "valset_iv.csv": "mimic4extract/mimic3models/resources/valset_iv.csv",
}
SHA256: dict[str, str] = {
    "testset_iv.csv": (
        "9fe512f5feade04158a08eb2ff7aca90476e9d54ab5a45db658abc1dd3ce5b68"
    ),
    "valset_iv.csv": (
        "563917e13e65ab1974152ecda546a398e30b8f7cea378b51dc7fe4bff41b8594"
    ),
}
CACHE_DIR = Path(BASE_CACHE_PATH) / "medmod"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_list_path(name: str, cache_dir: str | Path | None = None) -> Path:
    """Local path of one of MedMod's subject lists, downloaded on first use.

    Args:
        name: ``"testset_iv.csv"`` or ``"valset_iv.csv"``.
        cache_dir: where to keep it (default ``~/.cache/pyhealth/medmod``).

    Returns:
        The path of the cached, checksum-verified file.

    Raises:
        OSError: if the file is not cached and no source can be downloaded.

    Examples:
        >>> from pyhealth.datasets.medmod.split_lists import split_list_path
        >>> split_list_path("testset_iv.csv").name  # doctest: +SKIP
        'testset_iv.csv'
    """
    if name not in REPO_PATHS:
        raise ValueError(f"unknown list {name!r}; expected one of {sorted(REPO_PATHS)}")
    folder = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    path = folder / name
    if path.is_file() and _sha256(path.read_bytes()) == SHA256[name]:
        return path
    errors = []
    for base in SOURCES:
        url = f"{base}/{REPO_PATHS[name]}"
        try:
            logger.info("Downloading %s", url)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except OSError as error:  # URLError and HTTPError are OSErrors
            logger.warning("Could not download %s: %s", url, error)
            errors.append(f"{url}: {error}")
            continue
        if _sha256(data) != SHA256[name]:
            errors.append(f"{url}: unexpected SHA-256")
            continue
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".partial")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)  # never leaves a half-written file
        return path
    raise OSError(
        f"Could not download MedMod's {name} from any source:\n  "
        + "\n  ".join(errors)
        + f"\nDownload it manually into {folder}, or pass its path explicitly."
    )
