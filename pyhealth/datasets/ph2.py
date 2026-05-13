"""
PyHealth dataset for the PH2 dermoscopic image database.

The PH2 dataset contains 200 dermoscopic images of melanocytic lesions with
three diagnostic categories: common nevus, atypical nevus, and melanoma.

Dataset source
--------------
Original dataset:
    https://www.fc.up.pt/addi/ph2%20database.html  (requires registration)

Mirror used by ``download=True``:
    https://github.com/vikaschouhan/PH2-dataset
    (200 JPEGs in ``images/``, labels in ``PH2_simple_dataset.csv``)


Directory structure expected under ``root``
-------------------------------------------
**Option A – downloaded via** ``download=True`` **(GitHub mirror)**::

    <root>/
        images/
            IMD002.jpg
            IMD003.jpg
            ...
        PH2_simple_dataset.csv   # image_name, diagnosis

**Option B – original PH2 release**::

    <root>/
        PH2_dataset.xlsx          # official Excel (12 header rows)
          — OR —
        PH2_dataset.csv           # user-converted CSV
        PH2_Dataset_images/
            IMD001/
                IMD001_Dermoscopic_Image/
                    IMD001.bmp
            ...

After the first call, a ``ph2_metadata_pyhealth.csv`` file is written next to
the source files and reused on subsequent loads instead of re-parsing.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from pyhealth.datasets import BaseDataset
from pyhealth.datasets.isic2018 import _download_file, _extract_zip

logger = logging.getLogger(__name__)

_MIRROR_ZIP_URL = (
    "https://github.com/vikaschouhan/PH2-dataset/archive/refs/heads/master.zip"
)
_IMAGES_DIR = "images"
_SIMPLE_CSV = "PH2_simple_dataset.csv"

# Canonical label strings stored in ph2_metadata_pyhealth.csv
_LABEL_MAP = {
    "Common Nevus": "common_nevus",
    "Atypical Nevus": "atypical_nevus",
    "Melanoma": "melanoma",
}


class PH2Dataset(BaseDataset):
    """Base image dataset for the PH2 dermoscopic image database.

    The PH2 dataset contains 200 dermoscopic images of melanocytic lesions
    in three diagnostic categories: common nevus, atypical nevus, and melanoma.

    Args:
        root: Path to the directory containing the PH2 source files.
        download: If ``True``, automatically download data from the GitHub
            mirror when the image directory is absent.  Requires ~30 MB.
            Defaults to ``False``.
        dataset_name: Optional override for the internal dataset identifier.
            Defaults to ``"ph2"``.
        config_path: Optional path to a custom YAML config.  Defaults to the
            bundled ``configs/ph2.yaml``.
        cache_dir: Optional directory for litdata cache.
        dev: If ``True``, load only the first 1 000 records (for quick testing).

    Raises:
        FileNotFoundError: If required source files are missing and
            ``download=False``.
        requests.HTTPError: If ``download=True`` and the download fails.

    Examples:
        >>> dataset = PH2Dataset(root="/path/to/ph2", download=True)
        >>> from pyhealth.tasks import PH2MelanomaClassification
        >>> samples = dataset.set_task(PH2MelanomaClassification())
    """

    def __init__(
        self,
        root: str,
        download: bool = False,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir=None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "ph2.yaml")

        if download:
            self._download(root)

        self._verify_data(root)

        metadata_path = Path(root) / "ph2_metadata_pyhealth.csv"
        if not metadata_path.exists():
            self._prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["ph2"],
            dataset_name=dataset_name or "ph2",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self, root: str) -> None:
        """Download PH2 data from the GitHub mirror.

        Downloads and extracts ~30 MB.  Skips if ``images/`` already exists.

        Args:
            root: Target directory.

        Raises:
            requests.HTTPError: If the download request fails.
        """
        os.makedirs(root, exist_ok=True)
        images_dest = os.path.join(root, _IMAGES_DIR)

        if os.path.isdir(images_dest):
            logger.info("PH2 images already present at %s, skipping download.", images_dest)
            return

        zip_path = os.path.join(root, "ph2_mirror.zip")
        if not os.path.isfile(zip_path):
            logger.info("Downloading PH2 mirror from GitHub: %s", _MIRROR_ZIP_URL)
            _download_file(_MIRROR_ZIP_URL, zip_path)

        logger.info("Extracting PH2 archive to %s ...", root)
        self._extract_mirror(zip_path, root)
        os.remove(zip_path)
        logger.info("PH2 data ready at %s", root)

    @staticmethod
    def _extract_mirror(zip_path: str, dest: str) -> None:
        """Extract GitHub mirror zip, flattening the top-level directory.

        The GitHub archive has a single top-level folder
        (``PH2-dataset-master/``).  This method moves ``images/`` and
        ``PH2_simple_dataset.csv`` directly into *dest*.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            # Determine top-level prefix (e.g. "PH2-dataset-master/")
            prefix = members[0].split("/")[0] + "/" if members else ""

            for member in members:
                # Strip the top-level prefix
                rel = member[len(prefix):]
                if not rel:
                    continue
                target = os.path.join(dest, rel)
                if member.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as out:
                        out.write(src.read())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _verify_data(self, root: str) -> None:
        """Raise informative errors if required source files are missing."""
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        has_simple_csv = os.path.isfile(os.path.join(root, _SIMPLE_CSV))
        has_orig_xlsx = os.path.isfile(os.path.join(root, "PH2_dataset.xlsx"))
        has_orig_csv = os.path.isfile(os.path.join(root, "PH2_dataset.csv"))
        has_images = os.path.isdir(os.path.join(root, _IMAGES_DIR))
        has_bmp_images = os.path.isdir(os.path.join(root, "PH2_Dataset_images"))

        if not (has_simple_csv or has_orig_xlsx or has_orig_csv):
            raise FileNotFoundError(
                f"No PH2 metadata file found in {root}.\n"
                "Expected one of: PH2_simple_dataset.csv, PH2_dataset.xlsx, PH2_dataset.csv\n"
                "Pass download=True to fetch data automatically."
            )

        if not (has_images or has_bmp_images):
            raise FileNotFoundError(
                f"No PH2 image directory found in {root}.\n"
                "Expected 'images/' (GitHub mirror) or 'PH2_Dataset_images/' (original).\n"
                "Pass download=True to fetch data automatically."
            )

    # ------------------------------------------------------------------
    # Metadata preparation
    # ------------------------------------------------------------------

    def _prepare_metadata(self, root: str) -> None:
        """Parse source files and write ``ph2_metadata_pyhealth.csv``.

        Supports two source layouts:

        * **GitHub mirror**: ``PH2_simple_dataset.csv`` + ``images/IMDXXX.jpg``
        * **Original PH2**: ``PH2_dataset.xlsx`` / ``PH2_dataset.csv`` +
          ``PH2_Dataset_images/IMDXXX/IMDXXX_Dermoscopic_Image/IMDXXX.bmp``

        Args:
            root: Directory containing the PH2 source files.
        """
        logger.info("Processing PH2 metadata…")
        root_path = Path(root)

        has_simple_csv = (root_path / _SIMPLE_CSV).exists()
        has_orig_images = (root_path / "PH2_Dataset_images").exists()

        # Prefer original BMP format when PH2_Dataset_images/ is present,
        # even if PH2_simple_dataset.csv is also in the directory.
        if has_orig_images or not has_simple_csv:
            df = self._load_original(root_path)
        else:
            df = self._load_simple_csv(root_path)

        output_path = root_path / "ph2_metadata_pyhealth.csv"
        df[["image_id", "path", "diagnosis"]].to_csv(str(output_path), index=False)
        logger.info("Saved PH2 metadata to %s (%d images)", output_path, len(df))

    def _load_simple_csv(self, root: Path) -> pd.DataFrame:
        """Load GitHub mirror format (flat JPEGs + PH2_simple_dataset.csv)."""
        df = pd.read_csv(str(root / _SIMPLE_CSV))
        df = df.rename(columns={"image_name": "image_id"})

        # Normalise diagnosis strings
        df["diagnosis"] = df["diagnosis"].map(
            lambda v: _LABEL_MAP.get(str(v).strip(), "Unknown")
        )

        image_dir = root / _IMAGES_DIR

        def _path(img_id: str) -> Optional[str]:
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = image_dir / f"{img_id}{ext}"
                if p.exists():
                    return str(p)
            return None

        df["path"] = df["image_id"].apply(_path)
        df = df.dropna(subset=["path"])
        df = df[df["diagnosis"] != "Unknown"]
        return df

    def _load_original(self, root: Path) -> pd.DataFrame:
        """Load original PH2 format (nested BMPs + Excel/CSV)."""
        xlsx = root / "PH2_dataset.xlsx"
        csv = root / "PH2_dataset.csv"

        if xlsx.exists():
            raw = pd.read_excel(str(xlsx), header=12)
        elif csv.exists():
            raw = pd.read_csv(str(csv))
        else:
            raise FileNotFoundError(
                f"Could not find PH2_dataset.xlsx or PH2_dataset.csv in {root}"
            )

        raw = raw.rename(
            columns={
                "Image Name": "image_id",
                "Common Nevus": "common_nevus",
                "Atypical Nevus": "atypical_nevus",
                "Melanoma": "melanoma",
            }
        )

        def _diagnosis(row: pd.Series) -> str:
            if row.get("melanoma") == "X":
                return "melanoma"
            if row.get("atypical_nevus") == "X":
                return "atypical_nevus"
            if row.get("common_nevus") == "X":
                return "common_nevus"
            return "Unknown"

        raw["diagnosis"] = raw.apply(_diagnosis, axis=1)

        image_root = root / "PH2_Dataset_images"

        def _path(img_id: str) -> Optional[str]:
            p = image_root / img_id / f"{img_id}_Dermoscopic_Image" / f"{img_id}.bmp"
            return str(p) if p.exists() else None

        raw["path"] = raw["image_id"].apply(_path)
        raw = raw.dropna(subset=["path"])
        raw = raw[raw["diagnosis"] != "Unknown"]
        return raw

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    @property
    def default_task(self):
        """Return the default task (:class:`~pyhealth.tasks.PH2MelanomaClassification`)."""
        from pyhealth.tasks import PH2MelanomaClassification

        return PH2MelanomaClassification()

