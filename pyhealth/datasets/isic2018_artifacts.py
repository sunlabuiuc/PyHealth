"""
PyHealth dataset for dermoscopy images with per-image artifact annotations.

Overview
--------
Dermoscopy images frequently contain non-clinical artifacts — visual elements
introduced during image acquisition that are unrelated to the underlying
pathology.  When these artifacts correlate with diagnostic labels in training
data they create *spurious shortcuts* that inflate reported model accuracy
without capturing genuine disease features.

This dataset pairs any collection of dermoscopy images with a per-image
artifact annotation CSV.  The default annotation file is ``isic_bias.csv``
from Bissoto et al. (2020), which was created for the ISIC 2018 Task 1/2
image set, but the dataset class accepts any CSV that follows the same column
format.

Default annotation source
--------------------------
Using ``ISIC2018ArtifactsDataset`` requires **two separate downloads**:

1. **Artifact annotations** (``isic_bias.csv``) — Bissoto et al. (2020):
   https://github.com/alceubissoto/debiasing-skin

   See ``artefacts-annotation/`` in that repository for the annotation files.

   Reference:
       Bissoto et al. "Debiasing Skin Lesion Datasets and Models? Not So Fast"
       ISIC Skin Image Analysis Workshop @ CVPR 2020

2. **ISIC 2018 Task 1/2 images and segmentation masks** (~8 GB):
   https://challenge.isic-archive.com/data/#2018

   * Training images: ``ISIC2018_Task1-2_Training_Input.zip``
   * Segmentation masks: ``ISIC2018_Task1_Training_GroundTruth.zip``

Both can be fetched automatically by passing ``download=True`` to the
constructor (see class docs for details).

Artifact types
--------------
The default CSV provides seven binary artifact labels per image
(1 = present, 0 = absent).  Any additional binary columns in a custom CSV
are also preserved and accessible on events.

=================  =============================================================
Label              Description
=================  =============================================================
``dark_corner``    Dark vignetting at the image periphery, typically from the
                   dermoscope lens edge.
``hair``           Hair strands crossing the field of view and obscuring skin
                   surface details.
``gel_border``     Visible boundary of the contact gel or immersion fluid used
                   during dermoscopic examination.
``gel_bubble``     Air bubbles trapped in the contact gel, appearing as
                   circular bright reflections.
``ruler``          Measurement scale or ruler placed in the frame for size
                   reference.
``ink``            Ink or marker pen markings drawn on the skin before
                   acquisition (e.g., for surgical planning).
``patches``        Adhesive patches or stickers visible in the image.
=================  =============================================================

Image preprocessing
--------------------
Image preprocessing is not handled by this dataset.  Pass an
``input_processors={"image": <processor>}`` argument to ``set_task`` to
apply any image transformation.  The dataset exposes ``dataset.mask_dir``
so downstream processors can locate segmentation masks without hard-coding
the path.

CSV format
----------
The annotation CSV must be **semicolon-delimited** (``sep=";"``), with an
unnamed integer index as the first column, and must contain:

* ``image``        — image filename (must match files present in ``image_dir``).
* ``label``        — binary classification label (1 = malignant, 0 = benign).

Any additional columns are preserved as event attributes.  Columns beginning
with ``split_`` are treated as fold / trap-set assignment columns.

Cross-validation and trap-set protocol
---------------------------------------
When using the default Bissoto et al. CSV, five-fold splits
(``split_1`` … ``split_5``) support standard K-fold evaluation.

The *trap-set* protocol (Bissoto et al. 2020) studies whether models trained
on artifact-biased data learn spurious correlations.
"""

import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import requests
import yaml

from pyhealth.datasets import BaseDataset
from pyhealth.datasets.isic2018 import (
    TASK12_IMAGES_DIR as _IMAGES_DIR,
    TASK12_IMAGES_URL as _IMAGES_URL,
    TASK12_MASKS_DIR as _MASKS_DIR,
    TASK12_MASKS_URL as _MASKS_URL,
    _T12_IMAGES_ZIP as _IMAGES_ZIP,
    _T12_MASKS_ZIP as _MASKS_ZIP,
    _CHECKSUMS as _ISIC_CHECKSUMS,
    _download_file,
    _extract_zip,
)
logger = logging.getLogger(__name__)

_BIAS_CSV = "isic_bias.csv"  # default Bissoto et al. annotation filename
_BIAS_CSV_URL = (
    "https://raw.githubusercontent.com/alceubissoto/debiasing-skin/"
    "master/artefacts-annotation/isic_bias.csv"
)

#: The seven dermoscopic artifact categories annotated in ``isic_bias.csv``.
#: Each label is a binary column (1 = artifact present, 0 = absent).
#: See the module docstring for a detailed description of each type.
ARTIFACT_LABELS: List[str] = [
    "dark_corner",   # dark lens-edge vignetting
    "hair",          # hair strands overlapping the field of view
    "gel_border",    # visible contact-gel or immersion-fluid boundary
    "gel_bubble",    # air bubbles in the contact gel
    "ruler",         # measurement scale / ruler in the frame
    "ink",           # ink or marker-pen markings on the skin
    "patches",       # adhesive patches or stickers
]


class ISIC2018ArtifactsDataset(BaseDataset):
    """PyHealth dataset for dermoscopy images with per-image artifact annotations.

    Pairs a directory of dermoscopy images with an artifact annotation CSV.
    Any CSV that contains an ``image`` column (filenames), a ``label`` column
    (binary classification target), and one or more binary artifact columns is
    supported.  The default CSV is ``isic_bias.csv`` from Bissoto et al. (2020),
    annotated on the ISIC 2018 Task 1/2 image set.

    Image preprocessing is decoupled from this dataset.  Supply an image
    processor via ``input_processors={"image": <processor>}`` when calling
    ``set_task``.  The resolved mask directory is available as
    ``dataset.mask_dir`` for convenience.

    Attributes:
        artifact_labels (List[str]): The seven well-known artifact types from
            Bissoto et al. (2020).  Any subset present in the CSV is exposed.
        mask_dir (str): Resolved absolute path to the segmentation-mask
            directory.

    The expected directory structure under ``root`` is::

        <root>/
            <annotations_csv>              ← downloadable via download=True
            <image_dir>/                   ← downloadable via download=True
                ISIC_0024306.jpg
                ISIC_0024307.jpg
                ...
            <mask_dir>/                    ← downloadable via download=True; only
                ISIC_0024306_segmentation.png   required for non-"whole" modes
                ...

    When ``download=True`` is used with the default annotation CSV, the
    following files are fetched automatically:

    * ``isic_bias.csv`` — from the Bissoto et al. GitHub repository.
    * ``ISIC2018_Task1-2_Training_Input/`` — ISIC 2018 Task 1/2 training
      images (~8 GB), extracted from the ISIC S3 archive.
    * ``ISIC2018_Task1_Training_GroundTruth/`` — ISIC 2018 Task 1
      segmentation masks, extracted from the ISIC S3 archive.

    Pass ``image_dir="ISIC2018_Task1-2_Training_Input"`` and
    ``mask_dir="ISIC2018_Task1_Training_GroundTruth"`` to match the
    extracted layout.

    Example — Bissoto et al. default CSV with on-demand download::

        >>> dataset = ISIC2018ArtifactsDataset(
        ...     root="/data/isic",
        ...     image_dir="ISIC2018_Task1-2_Training_Input",
        ...     mask_dir="ISIC2018_Task1_Training_GroundTruth",
        ...     download=True,   # fetches CSV + ~8 GB images on first run
        ... )

    Example — images already on disk::

        >>> dataset = ISIC2018ArtifactsDataset(
        ...     root="/data/isic",
        ...     image_dir="2018_train_task1-2",
        ...     mask_dir="2018_train_task1-2_segmentations",
        ... )
        >>> sample_ds = dataset.set_task(dataset.default_task, input_processors={"image": my_processor})

    Example — custom annotation CSV::

        >>> dataset = ISIC2018ArtifactsDataset(
        ...     root="/data/my_dataset",
        ...     annotations_csv="my_annotations.csv",
        ...     image_dir="images",
        ...     mask_dir="masks",
        ... )
    """

    artifact_labels: List[str] = ARTIFACT_LABELS

    def __init__(
        self,
        root: str = ".",
        annotations_csv: str = _BIAS_CSV,
        image_dir: str = "images",
        mask_dir: str = "masks",
        download: bool = False,
        **kwargs,
    ) -> None:
        """Initialise the artifact dataset.

        Args:
            root (str): Root directory containing the annotation CSV, the
                image directory, and the segmentation-mask directory.
            annotations_csv (str): Filename of the annotation CSV inside
                ``root``.  Defaults to ``"isic_bias.csv"`` (Bissoto et al.
                2020).  The file must contain at minimum an ``image`` column
                (filenames) and a ``label`` column (binary target).
            image_dir (str): Sub-directory name (or absolute path) for the
                dermoscopy images.  Defaults to ``"images"``.
            mask_dir (str): Sub-directory name (or absolute path) for the
                segmentation masks.  Defaults to ``"masks"``.  The resolved
                path is exposed as ``dataset.mask_dir``.
            download (bool): If ``True`` and ``annotations_csv`` is the
                default ``"isic_bias.csv"``, download all missing data
                automatically:

                * ``isic_bias.csv`` — from the Bissoto et al. GitHub repo.
                * ISIC 2018 Task 1/2 training images (~8 GB) — from the
                  ISIC S3 archive; extracted to
                  ``<root>/ISIC2018_Task1-2_Training_Input/``.
                * ISIC 2018 Task 1 segmentation masks — from the ISIC S3
                  archive; extracted to
                  ``<root>/ISIC2018_Task1_Training_GroundTruth/``.

                Pass ``image_dir="ISIC2018_Task1-2_Training_Input"`` and
                ``mask_dir="ISIC2018_Task1_Training_GroundTruth"`` to use the
                extracted directories.  Raises :exc:`ValueError` if used
                with a custom ``annotations_csv``.  Defaults to ``False``.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~pyhealth.datasets.BaseDataset`.

        Raises:
            ValueError: If ``download=True`` is used with a custom
                ``annotations_csv``, or no images match the CSV.
            FileNotFoundError: If ``root``, the annotation CSV, the image
                directory, or the mask directory is missing.
            requests.HTTPError: If ``download=True`` and the CSV download
                fails.
        """
        if download and annotations_csv != _BIAS_CSV:
            raise ValueError(
                "download=True is only supported for the default "
                f"annotations_csv='{_BIAS_CSV}'. "
                "Provide your own CSV or omit the download flag."
            )

        self.annotations_csv = annotations_csv

        self._image_dir = (image_dir if os.path.isabs(
            image_dir) else os.path.join(root, image_dir))
        self.mask_dir = (mask_dir if os.path.isabs(
            mask_dir) else os.path.join(root, mask_dir))
        self._bias_csv_path = os.path.join(root, annotations_csv)

        if download:
            self._download_bias_csv(root)
            self._download_images(root)

        self._verify_data(root)
        config_path = self._index_data(root)

        super().__init__(
            root=root,
            tables=["isic_artifacts"],
            dataset_name="ISICArtifact",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self):
        """Return the default task for this dataset.

        Returns:
            None: No default task is registered until the ISIC task classes
            are available. Use ``dataset.set_task(task)`` directly.
        """
        return None

    def _download_bias_csv(self, root: str) -> None:
        """Download the default Bissoto et al. ``isic_bias.csv`` from GitHub.

        Skips the download if the file already exists.

        Args:
            root: Dataset root directory where the CSV will be saved.

        Raises:
            requests.HTTPError: If the HTTP request returns an error status.
        """
        if os.path.isfile(self._bias_csv_path):
            logger.info(
                "%s already present, skipping download.",
                self.annotations_csv)
            return

        os.makedirs(root, exist_ok=True)
        logger.info("Downloading %s from GitHub...", self.annotations_csv)
        response = requests.get(_BIAS_CSV_URL, timeout=60)
        response.raise_for_status()
        with open(self._bias_csv_path, "wb") as fh:
            fh.write(response.content)
        logger.info("Saved %s → %s", self.annotations_csv, self._bias_csv_path)

    def _download_images(self, root: str) -> None:
        """Download and extract ISIC 2018 Task 1/2 images and masks.

        Skips download if ZIP files already exist (they may be partial/incomplete).
        Always attempts extraction if ZIP is present.  The images archive is
        ~8 GB — this may take several minutes.

        Args:
            root: Dataset root directory.

        Raises:
            requests.HTTPError: If any HTTP request returns an error status.
            ValueError: If MD5 checksum verification fails.
        """
        os.makedirs(root, exist_ok=True)

        images_dest = os.path.join(root, _IMAGES_DIR)
        if not os.path.isdir(images_dest):
            zip_path = os.path.join(root, _IMAGES_ZIP)
            # Skip download if ZIP already exists (may be partial/incomplete)
            if not os.path.isfile(zip_path):
                logger.info(
                    "Downloading ISIC 2018 Task 1/2 images (~8 GB): %s",
                    _IMAGES_URL)
                _download_file(
                    _IMAGES_URL,
                    zip_path,
                    _ISIC_CHECKSUMS.get(_IMAGES_ZIP))
            if os.path.isfile(zip_path):
                logger.info("Extracting images to %s ...", root)
                _extract_zip(zip_path, root)
                os.remove(zip_path)
                logger.info("Images ready at %s", images_dest)
        else:
            logger.info(
                "Image directory already present, skipping: %s",
                images_dest)

        masks_dest = os.path.join(root, _MASKS_DIR)
        if not os.path.isdir(masks_dest):
            zip_path = os.path.join(root, _MASKS_ZIP)
            # Skip download if ZIP already exists (may be partial/incomplete)
            if not os.path.isfile(zip_path):
                logger.info(
                    "Downloading ISIC 2018 Task 1 segmentation masks: %s",
                    _MASKS_URL)
                _download_file(
                    _MASKS_URL,
                    zip_path,
                    _ISIC_CHECKSUMS.get(_MASKS_ZIP))
            if os.path.isfile(zip_path):
                logger.info("Extracting masks to %s ...", root)
                _extract_zip(zip_path, root)
                os.remove(zip_path)
                logger.info("Masks ready at %s", masks_dest)
        else:
            logger.info(
                "Mask directory already present, skipping: %s",
                masks_dest)

    def _verify_data(self, root: str) -> None:
        """Check required paths exist and raise informative errors if not."""
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

        if not os.path.isfile(self._bias_csv_path):
            msg = f"Annotation CSV not found: {self._bias_csv_path}"
            if self.annotations_csv == _BIAS_CSV:
                msg += (
                    "\nDownload it from: "
                    "https://github.com/alceubissoto/debiasing-skin"
                    "/tree/main/artefacts-annotation"
                    "\nOr pass download=True to fetch it automatically."
                )
            raise FileNotFoundError(msg)

        if not os.path.isdir(self._image_dir):
            raise FileNotFoundError(
                f"Image directory not found: {self._image_dir}\n"
                "Download images with download=True (requires ~8 GB), or "
                "obtain them manually from: "
                "https://challenge.isic-archive.com/data/#2018"
            )

        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}\n"
                "Download masks with download=True, or "
                "obtain them manually from: "
                "https://challenge.isic-archive.com/data/#2018"
            )

    def _index_data(self, root: str) -> str:
        """Parse ``isic_bias.csv`` and write a metadata CSV + YAML config.

        All columns present in ``isic_bias.csv`` (artifact labels, split
        columns, trap-set columns) are preserved so tasks can filter by any
        of them.

        Args:
            root: Dataset root directory.

        Returns:
            str: Path to the generated YAML configuration file.

        Raises:
            ValueError: If no images from the CSV are found on disk.
        """
        df = pd.read_csv(self._bias_csv_path, sep=";", index_col=0)

        # Keep only rows whose image file exists on disk.
        # Resolve extension mismatches (e.g. CSV lists .png but files are
        # .jpg).
        available = {f.name: f.name for f in Path(
            self._image_dir).iterdir() if f.is_file()}
        stem_to_actual = {
            Path(name).stem: name for name in available
        }

        def _resolve_filename(csv_name: str) -> str:
            """Return the actual on-disk filename, resolving extension mismatches."""
            if csv_name in available:
                return csv_name
            stem = Path(csv_name).stem
            return stem_to_actual.get(stem, "")

        df["image"] = df["image"].apply(_resolve_filename)
        df = df[df["image"] != ""].copy()

        if df.empty:
            raise ValueError(
                f"No matching images found in '{self._image_dir}'. "
                "Ensure image filenames in the annotation CSV correspond to "
                "files present in the image directory."
            )

        # Derive image_id (stem without extension) and patient_id
        df["image_id"] = df["image"].str.replace(
            r"\.[A-Za-z]+$", "", regex=True)
        df["patient_id"] = df["image_id"]

        # Absolute path to the image file
        df["path"] = df["image"].apply(
            lambda name: os.path.join(self._image_dir, name)
        )

        metadata_path = os.path.join(
            root, "isic-artifact-metadata-pyhealth.csv")
        df.to_csv(metadata_path, index=False)

        # Build YAML config dynamically so all CSV columns are accessible
        fixed_attrs = ["path", "image_id", "label"]
        artifact_attrs = [c for c in ARTIFACT_LABELS if c in df.columns]
        split_attrs = sorted(c for c in df.columns if c.startswith("split_"))
        attributes = fixed_attrs + artifact_attrs + split_attrs

        config: dict = {
            "version": "1.0",
            "tables": {
                "isic_artifacts": {
                    "file_path": "isic-artifact-metadata-pyhealth.csv",
                    "patient_id": "patient_id",
                    "timestamp": None,
                    "attributes": attributes,
                }
            },
        }

        config_path = os.path.join(root, "isic-artifact-config-pyhealth.yaml")
        with open(config_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

        logger.info(
            "ISIC2018ArtifactsDataset: indexed %d images → %s",
            len(df),
            metadata_path,
        )
        return config_path
