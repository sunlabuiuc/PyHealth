"""SIIM-ISIC 2020 melanoma dermoscopy dataset."""

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock

from ..utils import download_file
from .base_dataset import BaseDataset
from .imaging.io import _download_archives, _resolve_metadata_dir

logger = logging.getLogger(__name__)

ISIC2020_BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2020/"
"""Current host of the ISIC 2020 files. The older
``isic-challenge-data.s3.amazonaws.com`` host now answers HTTP 200 with a
small HTML redirect page instead of the file."""

_GROUND_TRUTH_FILES = (
    "ISIC_2020_Training_GroundTruth_v2.csv",
    "ISIC_2020_Training_GroundTruth.csv",
)
_DUPLICATES_FILE = "ISIC_2020_Training_Duplicates.csv"
_IMAGE_ZIP = "ISIC_2020_Training_JPEG.zip"
_IMAGE_DIR = "train"
_METADATA_FILE = "isic2020-metadata-pyhealth.csv"
_REQUIRED_COLUMNS = {
    "image_name",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "diagnosis",
    "benign_malignant",
    "target",
}


def _download_csv(url: str, path: Path, header_token: str) -> None:
    """Download a CSV and reject a response that is not that CSV.

    The old ISIC host returns an HTML page with status 200, so a successful
    request alone does not prove the file arrived.
    """
    download_file(url, str(path))
    with open(path, errors="replace") as handle:
        first_line = handle.readline()
    if header_token not in first_line:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {url} but it is not the expected CSV "
            f"(first line: {first_line[:80]!r})."
        )


class ISIC2020Dataset(BaseDataset):
    """SIIM-ISIC 2020 dermoscopy dataset for benign vs. malignant lesions.

    33,126 dermoscopic training images of skin lesions from 2,056 patients,
    collected at six sites by the International Skin Imaging Collaboration
    (ISIC). 584 images (1.8%) are malignant, all confirmed by
    histopathology. Each image is one event of the ``isic2020`` table, and
    the event's patient is the ISIC ``patient_id``, so a
    :class:`~pyhealth.datasets.PatientSplit` keeps all images of a patient
    in one part. Labels of the 10,982 challenge test images were never
    released, so only the training set is indexed.

    ``ISIC_2020_Training_Duplicates.csv`` lists 425 pairs of duplicate
    images (each pair within one patient). By default the second image of
    each pair is dropped.

    Getting the data:
        - Source: https://challenge.isic-archive.com/data/#2020 (ISIC 2020
          challenge data page; https://challenge2020.isic-archive.com/
          redirects there).
        - Access: open. No account, registration or data-use agreement;
          the files are plain HTTPS links on the ISIC S3 bucket.
        - Download: from ``https://isic-archive.s3.amazonaws.com/challenges/2020/``,
          ``ISIC_2020_Training_JPEG.zip`` (24.7 GB; the page shows 23 GiB),
          ``ISIC_2020_Training_GroundTruth_v2.csv`` (2.4 MB; the older
          ``ISIC_2020_Training_GroundTruth.csv``, without lesion ids, is
          also accepted) and ``ISIC_2020_Training_Duplicates.csv`` (12 KB).
          The DICOM archives and the unlabelled test set are not used::

              BASE=https://isic-archive.s3.amazonaws.com/challenges/2020
              mkdir isic2020 && cd isic2020
              wget "$BASE/ISIC_2020_Training_GroundTruth_v2.csv"
              wget "$BASE/ISIC_2020_Training_Duplicates.csv"
              wget "$BASE/ISIC_2020_Training_JPEG.zip"
              unzip -q ISIC_2020_Training_JPEG.zip   # -> train/

          Or pass ``download=True``, which downloads and unzips the missing
          files into ``root``. A ``train/`` folder with fewer JPEGs than the
          ground-truth CSV has rows (for example after an interrupted unzip)
          counts as missing: the zip is downloaded and extracted again, with
          a warning.
        - Unpack: the zip extracts to a ``train/`` folder of 33,126
          ``ISIC_xxxxxxx.jpg`` files; the zip can then be deleted.
        - ``root`` layout::

              isic2020/                                <- root
                  ISIC_2020_Training_GroundTruth_v2.csv
                  ISIC_2020_Training_Duplicates.csv
                  train/ISIC_0015719.jpg ...

          ``root`` is only read, unless ``download=True``, which writes the
          CSVs and the ``train/`` folder into it. The derived metadata CSV
          always goes to ``metadata_dir``. It is rebuilt when ``root``,
          ``drop_duplicates`` or the number of JPEGs in ``train/`` changes,
          so images restored after a partial unzip are picked up.

    Licence:
        CC BY-NC 4.0 (attribution, non-commercial use only; derivatives
        allowed). See https://creativecommons.org/licenses/by-nc/4.0/.

    Citation:
        International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge
        Dataset (2020). https://doi.org/10.34970/2020-ds01

        Rotemberg, V., Kurtansky, N., Betz-Stablein, B., et al. A
        patient-centric dataset of images and metadata for identifying
        melanomas using clinical context. Scientific Data 8, 34 (2021).
        https://doi.org/10.1038/s41597-021-00815-z

    Args:
        root: Folder holding the ground-truth CSV and ``train/``. Only
            read, unless ``download=True``.
        drop_duplicates: Drop the second image of each pair in
            ``ISIC_2020_Training_Duplicates.csv``. Defaults to True.
        download: Download any missing CSV and the image zip into ``root``.
            The zip is 24.7 GB, and is fetched again if ``train/`` holds
            fewer JPEGs than the ground-truth CSV has rows. Defaults to False.
        dataset_name: Name of the dataset. Defaults to ``"isic2020"``.
        config_path: Table configuration YAML. Defaults to the bundled one.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the metadata CSV. Defaults to
            ``<cache_dir or the PyHealth cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import ISIC2020Dataset
        >>> dataset = ISIC2020Dataset(root="/path/to/isic2020")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task()  # doctest: +SKIP
    """

    default_dataset_name = "isic2020"
    default_table = "isic2020"

    def __init__(
        self,
        root: str,
        drop_duplicates: bool = True,
        download: bool = False,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.source_root = Path(root).expanduser().resolve()
        self.image_dir = self.source_root / _IMAGE_DIR
        self.drop_duplicates = bool(drop_duplicates)
        if download:
            self._download()

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_FILE
        # The JPEG count makes a restored or re-extracted image folder
        # trigger a rebuild, so rows dropped for missing files come back.
        self._source_key = (
            f"root={self.source_root}|drop_duplicates={int(self.drop_duplicates)}"
            f"|images={self._image_count()}"
        )
        marker = self.metadata_dir / "isic2020-source.txt"
        with FileLock(str(self.metadata_path) + ".lock"):
            current = marker.read_text().strip() if marker.exists() else None
            if (
                refresh_metadata
                or not self.metadata_path.exists()
                or current != self._source_key
            ):
                self.prepare_metadata()
                marker.write_text(self._source_key)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "isic2020.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _image_count(self) -> int:
        """The number of JPEGs in ``train/`` (0 if it does not exist)."""
        if not self.image_dir.is_dir():
            return 0
        with os.scandir(self.image_dir) as entries:
            return sum(e.name.lower().endswith(".jpg") for e in entries)

    def _download(self) -> None:
        """Download the missing ground-truth CSVs and image zip into root.

        A ``train/`` folder with fewer JPEGs than the ground-truth CSV has
        rows is treated as incomplete, and the zip is fetched again.
        """
        self.source_root.mkdir(parents=True, exist_ok=True)
        if not any((self.source_root / f).exists() for f in _GROUND_TRUTH_FILES):
            name = _GROUND_TRUTH_FILES[0]
            _download_csv(
                ISIC2020_BASE_URL + name, self.source_root / name, "image_name"
            )
        if not (self.source_root / _DUPLICATES_FILE).exists():
            _download_csv(
                ISIC2020_BASE_URL + _DUPLICATES_FILE,
                self.source_root / _DUPLICATES_FILE,
                "image_name_1",
            )
        complete = self.image_dir.is_dir()
        if complete:
            gt_path = next(
                self.source_root / f
                for f in _GROUND_TRUTH_FILES
                if (self.source_root / f).exists()
            )
            n_rows = len(pd.read_csv(gt_path, usecols=["image_name"]))
            n_found = self._image_count()
            if n_found < n_rows:
                logger.warning(
                    "%s has %d of %d images; downloading %s again",
                    self.image_dir,
                    n_found,
                    n_rows,
                    _IMAGE_ZIP,
                )
                complete = False
        if not complete:
            _download_archives(
                [ISIC2020_BASE_URL + _IMAGE_ZIP],
                self.image_dir,
                archive_type="zip",
                expected=["."],
                extract_to_parent=True,
            )

    def prepare_metadata(self) -> None:
        """Write the metadata CSV, one row per image, to ``metadata_dir``.

        Rows whose JPEG is missing are dropped with a warning, so a partial
        download still loads; once the missing files are restored, the next
        load rebuilds the CSV because the JPEG count is part of the source key.

        Raises:
            FileNotFoundError: If no ground-truth CSV or no image is found.
            ValueError: If the ground-truth CSV lacks a required column.

        Examples:
            >>> dataset = ISIC2020Dataset(root="/path/to/isic2020")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        gt_path = next(
            (
                self.source_root / f
                for f in _GROUND_TRUTH_FILES
                if (self.source_root / f).exists()
            ),
            None,
        )
        if gt_path is None:
            raise FileNotFoundError(
                f"No ISIC 2020 ground-truth CSV ({' or '.join(_GROUND_TRUTH_FILES)})"
                f" in {self.source_root}. Download it from {ISIC2020_BASE_URL}"
                " or pass download=True."
            )
        gt = pd.read_csv(gt_path, dtype={"patient_id": str, "lesion_id": str})
        missing = _REQUIRED_COLUMNS - set(gt.columns)
        if missing:
            raise ValueError(f"{gt_path.name} lacks columns {sorted(missing)}.")
        if "lesion_id" not in gt.columns:
            gt["lesion_id"] = ""

        dup_path = self.source_root / _DUPLICATES_FILE
        if self.drop_duplicates:
            if dup_path.exists():
                dups = pd.read_csv(dup_path)
                gt = gt[~gt["image_name"].isin(dups["image_name_2"])]
            else:
                logger.warning(
                    "%s not found in %s; duplicates are kept.",
                    _DUPLICATES_FILE,
                    self.source_root,
                )

        paths = [str(self.image_dir / f"{name}.jpg") for name in gt["image_name"]]
        exists = pd.Series([Path(p).is_file() for p in paths], index=gt.index)
        if not exists.any():
            raise FileNotFoundError(
                f"No ISIC 2020 images found in {self.image_dir}. Unzip "
                f"{_IMAGE_ZIP} into {self.source_root}."
            )
        if not exists.all():
            logger.warning(
                "Dropping %d of %d images that are missing from %s",
                int((~exists).sum()),
                len(gt),
                self.image_dir,
            )
        gt = gt.assign(path=paths)[exists]

        out = pd.DataFrame(
            {
                "patient_id": gt["patient_id"],
                "image_id": gt["image_name"],
                "path": gt["path"],
                "lesion_id": gt["lesion_id"],
                "sex": gt["sex"],
                "age_approx": gt["age_approx"],
                "anatom_site": gt["anatom_site_general_challenge"],
                "diagnosis": gt["diagnosis"],
                "benign_malignant": gt["benign_malignant"],
                "target": gt["target"].astype(int),
            }
        )
        out.to_csv(self.metadata_path, index=False)
        logger.info("Wrote %d ISIC 2020 images to %s", len(out), self.metadata_path)

    @property
    def default_task(self) -> Any:
        """The default task, :class:`~pyhealth.tasks.ISIC2020MelanomaClassification`.

        Examples:
            >>> dataset = ISIC2020Dataset(root="/path/to/isic2020")  # doctest: +SKIP
            >>> dataset.default_task.task_name  # doctest: +SKIP
            'ISIC2020MelanomaClassification'
        """
        from ..tasks.isic2020_melanoma_classification import (
            ISIC2020MelanomaClassification,
        )

        return ISIC2020MelanomaClassification()
