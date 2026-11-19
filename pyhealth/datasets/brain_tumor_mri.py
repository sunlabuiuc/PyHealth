"""Brain Tumor Classification (MRI) dataset of Bhuvaji et al."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

#: Class folder names in label-index order (CLIMB's order).
BRAIN_TUMOR_MRI_CLASSES = (
    "no_tumor",
    "glioma_tumor",
    "meningioma_tumor",
    "pituitary_tumor",
)
_SPLITS = {"Training": "train", "Testing": "test"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_METADATA_CSV = "brain-tumor-mri-metadata-pyhealth.csv"
_MARKER = "brain-tumor-mri-metadata-source.txt"
#: ``fold`` value of the official Testing images. Training images get folds
#: 1-9 from a hash of their duplicate group, so copies share a fold.
TEST_FOLD = 10


def _md5(path: Path) -> str:
    """Return the hex MD5 of a file's bytes."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_fold(group_id: str) -> int:
    """Deterministic fold 1-9 for a Training duplicate group."""
    return int(hashlib.sha1(group_id.encode("utf-8")).hexdigest(), 16) % 9 + 1


class BrainTumorMRIDataset(BaseDataset):
    """Brain Tumor Classification (MRI): 3,264 brain MRI slices in 4 classes.

    The dataset of Bhuvaji et al. holds 2D brain MRI slices (JPEG, mixed
    planes and sequences) sorted into ``Training/`` (2,870) and ``Testing/``
    (394) folders, each with one subfolder per class: ``no_tumor``,
    ``glioma_tumor``, ``meningioma_tumor`` and ``pituitary_tumor``
    (:data:`BRAIN_TUMOR_MRI_CLASSES`, in label-index order). This class
    indexes every image as one ``images`` event.

    Duplicates and leakage: the release has no patient ids, and it contains
    byte-identical copies of the same image: 166 groups of identical files
    (403 files), 64 of them with copies in both Training and Testing. No group
    mixes classes. The loader hashes every file (``content_md5``) and uses
    the hash as ``patient_id``, so identical images always share a
    "patient" and a patient-level split keeps them together. With
    ``drop_duplicates=True`` (default) only one copy of each group is kept:
    a Testing copy when the group reaches Testing (so no test image is also
    a training image), otherwise the first Training copy by path. Copies
    within one split are removed too, and Testing has many: the defaults
    give 3,027 images, 2,713 Training and 314 Testing (from 394; pituitary
    Testing drops from 74 to 36). With ``drop_duplicates=False`` every
    file is kept and ``duplicate_count`` / ``cross_split_duplicate`` flag the
    copies. Near-duplicates (resized or re-encoded copies, adjacent slices of
    one patient) are not detected, so scores on the official split can still
    be optimistic.

    Columns: ``patient_id`` (the first 16 hex digits of ``content_md5``,
    shared by byte-identical copies), ``record_id``
    (``Training|Testing/<class>/<file>``, the path under ``root``),
    ``image_path``, ``split`` (``train`` / ``test``), ``label`` (class
    folder name),
    ``content_md5``, ``duplicate_count``, ``cross_split_duplicate`` and
    ``fold``. ``fold`` is 10 (:data:`TEST_FOLD`) for Testing images and 1-9
    for Training images, set by the duplicate group, so
    :func:`~pyhealth.datasets.split_by_strat_fold` gives the official test
    set with a validation fold carved from Training.

    Getting the data:
        - Source: https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet
          (archived, read-only). The same 3,264 files (93 MB) are on Kaggle
          at https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
          (doi 10.34740/KAGGLE/DSV/1183165). The citation is below.
        - Access: open. GitHub needs no account. Kaggle needs a free account
          and, for the CLI, an API token (Kaggle "Settings" > "API" >
          "Create New Token", saved as ``~/.kaggle/kaggle.json``) and
          ``pip install kaggle``.
        - Download, either:
          ``git clone --depth 1
          https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet``
          (175 MB on disk with ``.git``), or
          ``kaggle datasets download -d
          sartajbhuvaji/brain-tumor-classification-mri -p <root> --unzip``.
        - Unpacking: the clone needs none; ``--unzip`` unpacks the Kaggle
          zip (without it, run ``unzip brain-tumor-classification-mri.zip
          -d <root>``).
        - ``root`` layout (any other files are ignored)::

              root/
                Training/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/*.jpg
                Testing/{glioma_tumor,meningioma_tumor,no_tumor,pituitary_tumor}/*.jpg

        - Licence: MIT, as given on the Kaggle listing. The archived GitHub
          repository carries no LICENSE file.

    The metadata CSV is written to ``metadata_dir`` (by default
    ``<cache_dir>/metadata/brain_tumor_mri``), never into ``root``. It is
    rebuilt when ``root``, the set of image files or ``drop_duplicates``
    changes.

    Citation:
        Bhuvaji, S., Kadam, A., Bhumkar, P., Dedge, S., Kanchan, S. Brain
        Tumor Classification (MRI). Kaggle, 2020.
        https://doi.org/10.34740/KAGGLE/DSV/1183165

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).
    Deviations from CLIMB: CLIMB uses the official Training / Testing split
    as is, with the 64 cross-split duplicate groups, so some test images are
    also training images, and with repeated copies inside each split. Here
    every group is reduced to one image by default, a Testing one where
    possible, so test scores are not comparable with CLIMB's; pass
    ``drop_duplicates=False`` for the original 2,870 / 394 split. The class
    order and the four classes match CLIMB.

    Args:
        root: The cloned dataset folder holding ``Training/`` and
            ``Testing/``.
        drop_duplicates: Keep one copy of each group of byte-identical
            images. Defaults to True.
        dataset_name: Optional dataset name. Defaults to
            ``"brain_tumor_mri"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for PyHealth caches.
        metadata_dir: Explicit directory for the metadata CSV.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import BrainTumorMRIDataset
        >>> from pyhealth.tasks import BrainTumorMRIClassification
        >>> dataset = BrainTumorMRIDataset(
        ...     root="/data/Brain-Tumor-Classification-DataSet"
        ... )  # doctest: +SKIP
        >>> samples = dataset.set_task(BrainTumorMRIClassification())  # doctest: +SKIP
    """

    default_dataset_name = "brain_tumor_mri"
    default_table = "images"

    def __init__(
        self,
        root: str,
        drop_duplicates: bool = True,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.source_root = Path(root).expanduser().resolve()
        missing = [name for name in _SPLITS if not (self.source_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"{', '.join(missing)} not found under {self.source_root}. Clone "
                "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet "
                "and pass the cloned folder as root."
            )
        self.drop_duplicates = bool(drop_duplicates)
        self.image_paths = self._list_images()

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_CSV
        marker = self.metadata_dir / _MARKER
        source_key = self._source_key()
        with FileLock(str(self.metadata_path) + ".lock"):
            current = marker.read_text().strip() if marker.exists() else None
            if (
                refresh_metadata
                or not self.metadata_path.exists()
                or current != source_key
            ):
                self.prepare_metadata()
                marker.write_text(source_key)

        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "brain_tumor_mri.yaml"
            )
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _list_images(self) -> list[tuple[str, str, Path]]:
        """Return ``(split, class, path)`` for every image, sorted by path."""
        images = []
        for folder, split in _SPLITS.items():
            for label in BRAIN_TUMOR_MRI_CLASSES:
                class_dir = self.source_root / folder / label
                if not class_dir.is_dir():
                    logger.warning("Missing class folder %s", class_dir)
                    continue
                for path in sorted(class_dir.iterdir()):
                    if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                        images.append((split, label, path))
        if not images:
            raise FileNotFoundError(f"No images found under {self.source_root}.")
        return images

    def _source_key(self) -> str:
        """Hash of the inputs the metadata CSV depends on."""
        parts = [str(self.source_root), f"drop_duplicates={self.drop_duplicates}"]
        for _, _, path in self.image_paths:
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def prepare_metadata(self) -> None:
        """Hash every image and write the metadata CSV.

        Examples:
            >>> dataset = BrainTumorMRIDataset(root="/data/btm")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        rows = []
        for split, label, path in self.image_paths:
            folder = "Training" if split == "train" else "Testing"
            rows.append(
                {
                    "record_id": f"{folder}/{label}/{path.name}",
                    "image_path": str(path),
                    "split": split,
                    "label": label,
                    "content_md5": _md5(path),
                }
            )
        frame = pd.DataFrame(rows)
        groups = frame.groupby("content_md5")
        frame["duplicate_count"] = groups["record_id"].transform("size")
        frame["cross_split_duplicate"] = (
            groups["split"].transform("nunique").gt(1).astype(int)
        )
        conflicting = int((groups["label"].nunique() > 1).sum())
        if conflicting:
            logger.warning(
                "%d groups of identical images have different labels.", conflicting
            )
        n_groups = int((groups.size() > 1).sum())
        n_cross = int(frame.loc[frame["cross_split_duplicate"] == 1, "content_md5"].nunique())
        if self.drop_duplicates:
            # Testing copies first, then path order: keep the test set whole.
            frame["_order"] = (frame["split"] != "test").astype(int)
            frame = (
                frame.sort_values(["_order", "record_id"])
                .drop_duplicates("content_md5", keep="first")
                .drop(columns="_order")
                .sort_values("record_id")
            )
        frame.insert(0, "patient_id", frame["content_md5"].str[:16])
        frame["fold"] = [
            TEST_FOLD if split == "test" else _train_fold(group)
            for split, group in zip(frame["split"], frame["content_md5"])
        ]
        frame.to_csv(self.metadata_path, index=False)
        logger.info(
            "Wrote %d brain MRI images to %s (%d duplicate groups, %d spanning "
            "Training and Testing, drop_duplicates=%s)",
            len(frame),
            self.metadata_path,
            n_groups,
            n_cross,
            self.drop_duplicates,
        )

    @property
    def default_task(self):
        """Return the default four-class classification task.

        Examples:
            >>> BrainTumorMRIDataset(root="/data/btm").default_task  # doctest: +SKIP
        """
        from ..tasks.brain_tumor_mri_classification import (
            BrainTumorMRIClassification,
        )

        return BrainTumorMRIClassification()
