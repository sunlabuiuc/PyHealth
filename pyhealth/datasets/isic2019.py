"""ISIC 2019 dermoscopy dataset."""

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

ISIC2019_BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2019/"
"""Current host of the ISIC 2019 files. The older
``isic-challenge-data.s3.amazonaws.com`` host now answers HTTP 200 with a
small HTML redirect page instead of the file."""

ISIC2019_CLASSES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK")
"""The diagnosis codes, in the column order of the ground-truth CSVs. UNK
("none of the others") occurs only in the test set."""

_SPLIT_NAMES = {"train": "Training", "test": "Test"}
_METADATA_FILE = "isic2019-metadata-pyhealth.csv"
# The ISIC 2018 Task 3 validation and test images (HAM10000 holdout) are the
# ISIC 2019 test images with ids in this range: all 1,705 of them, and no
# other image.
_HAM10000_TEST_ID_RANGE = (34321, 36064)


def _ham10000_test_image(image_id: str) -> bool:
    """True for ISIC 2019 test images that are ISIC 2018 holdout images."""
    try:
        number = int(image_id[5:12])
    except ValueError:
        return False
    low, high = _HAM10000_TEST_ID_RANGE
    return low <= number <= high


def _download_csv(url: str, path: Path) -> None:
    """Download a CSV and reject a response that is not that CSV."""
    download_file(url, str(path))
    with open(path, errors="replace") as handle:
        first_line = handle.readline()
    if not first_line.startswith("image,"):
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded {url} but it is not the expected CSV "
            f"(first line: {first_line[:80]!r})."
        )


class ISIC2019Dataset(BaseDataset):
    """ISIC 2019 dermoscopy dataset for skin lesion diagnosis.

    25,331 dermoscopic training images, each with one of eight diagnoses:
    melanoma (MEL), melanocytic nevus (NV), basal cell carcinoma (BCC),
    actinic keratosis (AK), benign keratosis (BKL), dermatofibroma (DF),
    vascular lesion (VASC) and squamous cell carcinoma (SCC). They come from
    three sources: HAM10000 (10,015 images, Vienna), BCN_20000 (12,413,
    Barcelona) and MSK (Memorial Sloan Kettering). The 8,238 test images,
    whose labels are now public, add a ninth class, UNK (2,047 images,
    "none of the above").

    The PyHealth patient is the ``lesion_id`` (11,847 lesions over 23,247
    training images); the other 2,084 training images and every test image
    have no lesion id, so each of those is its own patient. A
    :class:`~pyhealth.datasets.PatientSplit` thus keeps the images of a
    known lesion together. There are no patient ids, so a patient with
    several lesions can still appear in more than one part.

    Overlap with HAM10000: all 10,015 HAM10000 training images are in this
    training set (lesion ids starting ``HAM_``), and all 1,705 ISIC 2018
    validation and test images are in this test set (image ids
    ISIC_0034321 to ISIC_0036064). Each event has ``is_ham10000``; the task's
    ``exclude_ham10000=True`` drops them, e.g. to evaluate a model trained
    on :class:`~pyhealth.datasets.HAM10000Dataset`.

    Getting the data:
        - Source: https://challenge.isic-archive.com/data/#2019
        - Access: open, no account needed.
        - Download: from ``https://isic-archive.s3.amazonaws.com/challenges/2019/``,
          ``ISIC_2019_Training_Input.zip`` (9.8 GB),
          ``ISIC_2019_Training_GroundTruth.csv`` and
          ``ISIC_2019_Training_Metadata.csv``; for the test set also
          ``ISIC_2019_Test_Input.zip`` (3.8 GB),
          ``ISIC_2019_Test_GroundTruth.csv`` and
          ``ISIC_2019_Test_Metadata.csv``. Or pass ``download=True``.
        - ``root``: the folder holding the CSVs and the unzipped
          ``ISIC_2019_Training_Input/`` (and ``ISIC_2019_Test_Input/``)
          folders. ``root`` is only read: the metadata CSV is written to
          ``metadata_dir``.

    Licence:
        CC BY-NC 4.0 (attribution, non-commercial use only). See
        https://creativecommons.org/licenses/by-nc/4.0/.

    Citation:
        Tschandl, P., Rosendahl, C., Kittler, H. The HAM10000 dataset, a large
        collection of multi-source dermatoscopic images of common pigmented
        skin lesions. Scientific Data 5, 180161 (2018).

        Codella, N., et al. Skin lesion analysis toward melanoma detection: a
        challenge at the 2017 ISBI, hosted by the ISIC. arXiv:1710.05006
        (2017).

        Hernández-Pérez, C., et al. BCN20000: Dermoscopic lesions in the
        wild. Scientific Data 11, 641 (2024).
        https://doi.org/10.1038/s41597-024-03387-w

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Differences from CLIMB:
        - Splits are by lesion (``PatientSplit`` on ``lesion_id``), not a
          random 80/20 split of images.
        - The default labels are the eight official classes; CLIMB's five
          (AK and SCC merged into AKIEC; BKL, DF and VASC into OTHER) are
          the task's ``label_scheme="climb5"``.
        - The HAM10000 overlap is flagged and can be excluded.
        - The public test set can be indexed. CLIMB's loader needs a manual
          download and calls ``DataFrame.append``, removed in pandas 2.

    Args:
        root: Folder holding the ISIC 2019 CSVs and image folders.
        splits: Which splits to index, from ``"train"`` and ``"test"``.
            Defaults to ``("train",)``.
        download: Download any missing file of the requested splits into
            ``root``. Defaults to False.
        dataset_name: Name of the dataset. Defaults to ``"isic2019"``.
        config_path: Table configuration YAML. Defaults to the bundled one.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the metadata CSV. Defaults to
            ``<cache_dir or the PyHealth cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import ISIC2019Dataset
        >>> dataset = ISIC2019Dataset(root="/path/to/isic2019")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task()  # doctest: +SKIP
    """

    default_dataset_name = "isic2019"
    default_table = "isic2019"

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
        marker = self.metadata_dir / "isic2019-source.txt"
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
            config_path = str(Path(__file__).parent / "configs" / "isic2019.yaml")
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
            prefix = f"ISIC_2019_{_SPLIT_NAMES[split]}"
            for kind in ("GroundTruth", "Metadata"):
                path = self.source_root / f"{prefix}_{kind}.csv"
                if not path.exists():
                    _download_csv(f"{ISIC2019_BASE_URL}{path.name}", path)
            image_dir = self.source_root / f"{prefix}_Input"
            if not image_dir.is_dir():
                _download_archives(
                    [f"{ISIC2019_BASE_URL}{prefix}_Input.zip"],
                    image_dir,
                    archive_type="zip",
                    expected=["."],
                    extract_to_parent=True,
                )

    def prepare_metadata(self) -> None:
        """Write the metadata CSV, one row per image, to ``metadata_dir``.

        Rows whose JPEG is missing are dropped with a warning, so a partial
        download still loads.

        Raises:
            FileNotFoundError: If a ground-truth CSV or every image of a
                split is missing.
            ValueError: If a ground-truth row does not have exactly one
                diagnosis.

        Examples:
            >>> dataset = ISIC2019Dataset(root="/path/to/isic2019")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        frames = []
        for split in self.splits:
            prefix = f"ISIC_2019_{_SPLIT_NAMES[split]}"
            gt_path = self.source_root / f"{prefix}_GroundTruth.csv"
            if not gt_path.exists():
                raise FileNotFoundError(
                    f"{gt_path} not found. Download it from {ISIC2019_BASE_URL}."
                )
            gt = pd.read_csv(gt_path)
            onehot = gt[list(ISIC2019_CLASSES)].to_numpy()
            if not (onehot.sum(axis=1) == 1).all():
                raise ValueError(f"{gt_path.name}: every row needs one diagnosis.")
            gt["diagnosis"] = [ISIC2019_CLASSES[i] for i in onehot.argmax(axis=1)]
            meta_path = self.source_root / f"{prefix}_Metadata.csv"
            if meta_path.exists():
                gt = gt.merge(pd.read_csv(meta_path), on="image", how="left")
            else:
                logger.warning("%s not found; no lesion ids or age/sex/site.", meta_path)
            for column in ("lesion_id", "age_approx", "anatom_site_general", "sex"):
                if column not in gt.columns:
                    gt[column] = None
            if "score_weight" not in gt.columns:
                gt["score_weight"] = None
            if split == "train":
                is_ham = gt["lesion_id"].fillna("").str.startswith("HAM_")
            else:
                is_ham = gt["image"].map(_ham10000_test_image)

            image_dir = self.source_root / f"{prefix}_Input"
            gt["path"] = [str(image_dir / f"{i}.jpg") for i in gt["image"]]
            exists = gt["path"].map(lambda p: Path(p).is_file())
            if not exists.any():
                raise FileNotFoundError(
                    f"No images found in {image_dir}. Unzip {prefix}_Input.zip "
                    f"into {self.source_root}."
                )
            if not exists.all():
                logger.warning(
                    "Dropping %d of %d %s images missing from %s",
                    int((~exists).sum()),
                    len(gt),
                    split,
                    image_dir,
                )
            gt, is_ham = gt[exists], is_ham[exists]
            frames.append(
                pd.DataFrame(
                    {
                        "patient_id": gt["lesion_id"].fillna(gt["image"]),
                        "image_id": gt["image"],
                        "lesion_id": gt["lesion_id"],
                        "path": gt["path"],
                        "split": split,
                        "diagnosis": gt["diagnosis"],
                        "is_ham10000": is_ham.astype(int),
                        "age_approx": gt["age_approx"],
                        "anatom_site": gt["anatom_site_general"],
                        "sex": gt["sex"],
                        "score_weight": gt["score_weight"],
                    }
                )
            )
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(self.metadata_path, index=False)
        logger.info("Wrote %d ISIC 2019 images to %s", len(out), self.metadata_path)

    @property
    def default_task(self) -> Any:
        """The default task, :class:`~pyhealth.tasks.ISIC2019Classification`.

        Examples:
            >>> dataset = ISIC2019Dataset(root="/path/to/isic2019")  # doctest: +SKIP
            >>> dataset.default_task.label_scheme  # doctest: +SKIP
            'isic8'
        """
        from ..tasks.isic2019_classification import ISIC2019Classification

        return ISIC2019Classification()
