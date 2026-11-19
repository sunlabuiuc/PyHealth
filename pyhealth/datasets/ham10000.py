"""HAM10000 / ISIC 2018 Task 3 dermoscopy dataset."""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock

from ..utils import download_file
from .base_dataset import BaseDataset
from .imaging.io import _download_archives, _resolve_metadata_dir

logger = logging.getLogger(__name__)

HAM10000_BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
"""Current host of the ISIC 2018 files. The older
``isic-challenge-data.s3.amazonaws.com`` host now answers HTTP 200 with a
small HTML redirect page instead of the file."""

HAM10000_CLASSES = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
"""The seven diagnosis codes, in the column order of the ground-truth CSVs."""

_SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Test"}
_GROUPINGS_FILE = "ISIC2018_Task3_Training_LesionGroupings.csv"
_DATAVERSE_METADATA = "HAM10000_metadata.csv"
_METADATA_FILE = "ham10000-metadata-pyhealth.csv"


def _input_dir(split: str) -> str:
    return f"ISIC2018_Task3_{_SPLIT_NAMES[split]}_Input"


def _ground_truth_dir(split: str) -> str:
    return f"ISIC2018_Task3_{_SPLIT_NAMES[split]}_GroundTruth"


class HAM10000Dataset(BaseDataset):
    """HAM10000 dermoscopy dataset, as released for ISIC 2018 Task 3.

    10,015 dermoscopic training images of 7,470 pigmented skin lesions from
    the Medical University of Vienna and a practice in Queensland, each with
    one of seven diagnoses: melanoma (MEL), melanocytic nevus (NV), basal
    cell carcinoma (BCC), actinic keratosis / intraepithelial carcinoma
    (AKIEC), benign keratosis (BKL), dermatofibroma (DF) and vascular lesion
    (VASC). Several images often show the same lesion, so the PyHealth
    patient is the ``lesion_id`` and a
    :class:`~pyhealth.datasets.PatientSplit` keeps all images of a lesion
    in one part. The images do not identify patients, so a patient with
    several lesions can still appear in more than one part.

    The official validation (193) and test (1,512) images are indexed too
    when ``splits`` asks for them. Their lesion ids were not released, so
    each of those images is its own patient. Every event has a ``split``
    attribute (``"train"``, ``"val"`` or ``"test"``).

    All 10,015 training images are also in the ISIC 2019 training set, and
    the ISIC 2018 validation and test images are in the ISIC 2019 test set,
    so do not train on one and test on the other.

    Getting the data:
        - Source: https://challenge.isic-archive.com/data/#2018 (ISIC 2018
          challenge data page, Task 3). The HAM10000 images are also on
          Harvard Dataverse (https://doi.org/10.7910/DVN/DBW86T), but this
          class reads the ISIC 2018 Task 3 layout below.
        - Access: open. No account, registration or data-use agreement;
          the files are plain HTTPS links on the ISIC S3 bucket.
        - Download: from ``https://isic-archive.s3.amazonaws.com/challenges/2018/``,
          for the training set ``ISIC2018_Task3_Training_Input.zip``
          (2.8 GB), ``ISIC2018_Task3_Training_GroundTruth.zip`` (37 KB) and
          ``ISIC2018_Task3_Training_LesionGroupings.csv`` (493 KB, required:
          it holds the lesion ids). For the official holdout also
          ``ISIC2018_Task3_Validation_Input.zip`` (53 MB),
          ``ISIC2018_Task3_Validation_GroundTruth.zip`` (8 KB),
          ``ISIC2018_Task3_Test_Input.zip`` (421 MB) and
          ``ISIC2018_Task3_Test_GroundTruth.zip`` (12 KB). Sizes are in
          decimal units; the ISIC page shows binary ones. For the training
          set::

              BASE=https://isic-archive.s3.amazonaws.com/challenges/2018
              mkdir isic2018_task3 && cd isic2018_task3
              wget "$BASE/ISIC2018_Task3_Training_LesionGroupings.csv"
              for f in Input GroundTruth; do
                  wget "$BASE/ISIC2018_Task3_Training_$f.zip"
                  unzip -q "ISIC2018_Task3_Training_$f.zip"
              done

          Or pass ``download=True``, which downloads and unzips the missing
          files of the requested ``splits`` into ``root``.
        - Unpack: each zip extracts to a folder of the same name; the zips
          can then be deleted. Instead of the lesion groupings CSV, the
          Dataverse ``HAM10000_metadata`` file is accepted, saved as
          ``HAM10000_metadata.csv`` (``wget -O HAM10000_metadata.csv
          "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original"``).
        - ``root`` layout (validation and test folders only when
          ``splits`` asks for them)::

              isic2018_task3/                                <- root
                  ISIC2018_Task3_Training_LesionGroupings.csv
                  ISIC2018_Task3_Training_Input/ISIC_0024306.jpg ...
                  ISIC2018_Task3_Training_GroundTruth/
                      ISIC2018_Task3_Training_GroundTruth.csv
                  ISIC2018_Task3_Validation_Input/  ISIC2018_Task3_Validation_GroundTruth/
                  ISIC2018_Task3_Test_Input/        ISIC2018_Task3_Test_GroundTruth/

          ``root`` is only read, unless ``download=True``, which writes the
          downloaded files into it. The derived metadata CSV always goes to
          ``metadata_dir``.

    Licence:
        CC BY-NC 4.0 (attribution, non-commercial use only; derivatives
        allowed), as stated on the ISIC data page, in the ``LICENSE.txt``
        inside each image zip and on the current Dataverse version. See
        https://creativecommons.org/licenses/by-nc/4.0/.

    Citation:
        Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large
        collection of multi-source dermatoscopic images of common pigmented
        skin lesions. Scientific Data 5, 180161 (2018).
        https://doi.org/10.1038/sdata.2018.161

        Codella, N., et al. Skin lesion analysis toward melanoma detection
        2018: a challenge hosted by the International Skin Imaging
        Collaboration (ISIC). arXiv:1902.03368 (2019).

    Args:
        root: Folder holding the ISIC 2018 Task 3 folders and CSVs.
        splits: Which official splits to index, from ``"train"``,
            ``"val"`` and ``"test"``. Defaults to ``("train",)``.
        download: Download and unzip any missing file of the requested
            splits into ``root`` (2.8 GB for train). Defaults to False.
        dataset_name: Name of the dataset. Defaults to ``"ham10000"``.
        config_path: Table configuration YAML. Defaults to the bundled one.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the metadata CSV. Defaults to
            ``<cache_dir or the PyHealth cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import HAM10000Dataset
        >>> dataset = HAM10000Dataset(root="/path/to/isic2018_task3")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task()  # doctest: +SKIP
    """

    default_dataset_name = "ham10000"
    default_table = "ham10000"

    def __init__(
        self,
        root: str,
        splits: Sequence[str] = ("train",),
        download: bool = False,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        if isinstance(splits, str):
            splits = (splits,)
        unknown = set(splits) - set(_SPLIT_NAMES)
        if unknown or not splits:
            raise ValueError(
                f"splits must be a non-empty subset of {sorted(_SPLIT_NAMES)}, "
                f"got {list(splits)}."
            )
        self.splits = tuple(s for s in _SPLIT_NAMES if s in splits)
        self.source_root = Path(root).expanduser().resolve()
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
        self._source_key = f"root={self.source_root}|splits={','.join(self.splits)}"
        marker = self.metadata_dir / "ham10000-source.txt"
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
            config_path = str(Path(__file__).parent / "configs" / "ham10000.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _download(self) -> None:
        """Download the missing files of the requested splits into root."""
        self.source_root.mkdir(parents=True, exist_ok=True)
        for split in self.splits:
            for folder in (_input_dir(split), _ground_truth_dir(split)):
                if not (self.source_root / folder).is_dir():
                    _download_archives(
                        [f"{HAM10000_BASE_URL}{folder}.zip"],
                        self.source_root / folder,
                        archive_type="zip",
                        expected=["."],
                        extract_to_parent=True,
                    )
        groupings = self.source_root / _GROUPINGS_FILE
        if "train" in self.splits and not groupings.exists():
            download_file(HAM10000_BASE_URL + _GROUPINGS_FILE, str(groupings))
            with open(groupings, errors="replace") as handle:
                first_line = handle.readline()
            if "lesion_id" not in first_line:
                groupings.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded {_GROUPINGS_FILE} is not the expected CSV "
                    f"(first line: {first_line[:80]!r})."
                )

    def _lesion_ids(self) -> pd.DataFrame:
        """Return ``image``, ``lesion_id``, ``diagnosis_confirm_type``."""
        groupings = self.source_root / _GROUPINGS_FILE
        if groupings.exists():
            return pd.read_csv(groupings)[
                ["image", "lesion_id", "diagnosis_confirm_type"]
            ]
        dataverse = self.source_root / _DATAVERSE_METADATA
        if dataverse.exists():
            meta = pd.read_csv(dataverse)
            return meta.rename(
                columns={"image_id": "image", "dx_type": "diagnosis_confirm_type"}
            )[["image", "lesion_id", "diagnosis_confirm_type"]]
        raise FileNotFoundError(
            f"Neither {_GROUPINGS_FILE} nor {_DATAVERSE_METADATA} is in "
            f"{self.source_root}. The lesion ids are needed to split the "
            f"training set by lesion; download {_GROUPINGS_FILE} from "
            f"{HAM10000_BASE_URL}."
        )

    def prepare_metadata(self) -> None:
        """Write the metadata CSV, one row per image, to ``metadata_dir``.

        Rows whose JPEG is missing are dropped with a warning, so a partial
        download still loads.

        Raises:
            FileNotFoundError: If a ground-truth CSV, the training lesion
                ids, or every image of a split is missing.
            ValueError: If a ground-truth row does not have exactly one
                diagnosis.

        Examples:
            >>> dataset = HAM10000Dataset(root="/path/to/isic2018")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        frames = []
        for split in self.splits:
            gt_dir = _ground_truth_dir(split)
            gt_path = self.source_root / gt_dir / f"{gt_dir}.csv"
            if not gt_path.exists():
                raise FileNotFoundError(
                    f"{gt_path} not found. Download {gt_dir}.zip from "
                    f"{HAM10000_BASE_URL} and unzip it into {self.source_root}."
                )
            gt = pd.read_csv(gt_path)
            onehot = gt[list(HAM10000_CLASSES)].to_numpy()
            if not (onehot.sum(axis=1) == 1).all():
                raise ValueError(f"{gt_path.name}: every row needs one diagnosis.")
            gt["diagnosis"] = [HAM10000_CLASSES[i] for i in onehot.argmax(axis=1)]
            if split == "train":
                gt = gt.merge(self._lesion_ids(), on="image", how="left")
                n_missing = int(gt["lesion_id"].isna().sum())
                if n_missing:
                    logger.warning(
                        "%d training images have no lesion id; each is its own "
                        "patient.",
                        n_missing,
                    )
            else:
                gt["lesion_id"] = None
                gt["diagnosis_confirm_type"] = None
            image_dir = self.source_root / _input_dir(split)
            gt["path"] = [str(image_dir / f"{i}.jpg") for i in gt["image"]]
            exists = gt["path"].map(lambda p: Path(p).is_file())
            if not exists.any():
                raise FileNotFoundError(
                    f"No images found in {image_dir}. Unzip "
                    f"{_input_dir(split)}.zip into {self.source_root}."
                )
            if not exists.all():
                logger.warning(
                    "Dropping %d of %d %s images missing from %s",
                    int((~exists).sum()),
                    len(gt),
                    split,
                    image_dir,
                )
            gt = gt[exists]
            frames.append(
                pd.DataFrame(
                    {
                        "patient_id": gt["lesion_id"].fillna(gt["image"]),
                        "image_id": gt["image"],
                        "lesion_id": gt["lesion_id"],
                        "path": gt["path"],
                        "split": split,
                        "diagnosis": gt["diagnosis"],
                        "diagnosis_confirm_type": gt["diagnosis_confirm_type"],
                    }
                )
            )
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(self.metadata_path, index=False)
        logger.info("Wrote %d HAM10000 images to %s", len(out), self.metadata_path)

    @property
    def default_task(self) -> Any:
        """The default task, :class:`~pyhealth.tasks.HAM10000Classification`.

        Examples:
            >>> dataset = HAM10000Dataset(root="/path/to/isic2018")  # doctest: +SKIP
            >>> dataset.default_task.label_scheme  # doctest: +SKIP
            'ham7'
        """
        from ..tasks.ham10000_classification import HAM10000Classification

        return HAM10000Classification()
