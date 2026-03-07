"""PyHealth dataset for composite dermoscopic image classification.

Combines three dermoscopy datasets — ISIC 2018, HAM10000, and PH2 — into
a single unified dataset for binary melanoma classification. Each dataset
provides dermoscopic images with segmentation masks, enabling mode-based
processing (whole image, lesion only, or background only).

The composite dataset normalizes metadata from all three sources into a
single CSV with a unified schema, following PyHealth conventions.

Data Sources:
    ISIC 2018:
        - Part of the ISIC Skin Lesion Analysis challenge
        - JPG images with PNG segmentation masks
    HAM10000:
        - "Human Against Machine with 10000 training images"
        - JPG images with PNG segmentation masks
    PH2:
        - Dermoscopic Image Database from Pedro Hispano hospital
        - BMP images with BMP segmentation masks in nested directories

Author:
    Generated for PyHealth integration of dermoscopic_artifacts datasets.
"""

import logging
import os
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.processors.dermoscopy_image_processor import DermoscopyImageProcessor
from pyhealth.tasks.dermoscopy_melanoma_classification import (
    DermoscopyMelanomaClassification,
)

logger = logging.getLogger(__name__)


class DermoscopyDataset(BaseDataset):
    """Composite dermoscopy dataset combining ISIC 2018, HAM10000, and PH2.

    This dataset unifies three dermoscopic image datasets into a single
    PyHealth-compatible dataset for binary melanoma classification. Each
    sample includes an image path, a segmentation mask path, a binary label,
    and a source dataset identifier.

    The dataset expects a root directory containing sub-dataset folders and
    external metadata CSV files that describe the images and labels.

    Args:
        root: Root directory containing sub-dataset image/mask folders.
            Expected structure:
                root/
                    isic2018/
                        images/     (JPG files)
                        masks/      (PNG segmentation files)
                    ham10000/
                        images/     (JPG files)
                        masks/      (PNG segmentation files)
                    ph2/
                        IMD{NNN}/   (nested BMP structure)
        datasets: List of sub-datasets to include. Any subset of
            ["isic2018", "ham10000", "ph2"]. Defaults to all three.
        isic2018_metadata_path: Path to the ISIC 2018 metadata CSV.
            Expected columns: "image" (filename), "label" (0/1 int).
        ham10000_metadata_path: Path to the HAM10000 metadata CSV.
            Expected columns: "image_id" (base name), "label" (0/1 int).
        ph2_metadata_path: Path to the PH2 metadata CSV.
            Expected columns: "Name" (folder name), "label" (0/1 int).
        dataset_name: Optional name for the dataset. Defaults to "dermoscopy".
        config_path: Optional path to YAML config. Defaults to built-in config.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of parallel workers. Defaults to 1.
        dev: If True, loads a small subset for development. Defaults to False.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        available_datasets: List of sub-datasets included.

    Examples:
        >>> from pyhealth.datasets import DermoscopyDataset
        >>> dataset = DermoscopyDataset(
        ...     root="/path/to/dermoscopy_data",
        ...     isic2018_metadata_path="/path/to/isic_bias.csv",
        ...     ham10000_metadata_path="/path/to/ham10000_metadata.csv",
        ...     ph2_metadata_path="/path/to/ph2_metadata.csv",
        ... )
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    SUPPORTED_DATASETS = ("isic2018", "ham10000", "ph2")

    def __init__(
        self,
        root: str,
        datasets: Optional[List[str]] = None,
        isic2018_metadata_path: Optional[str] = None,
        ham10000_metadata_path: Optional[str] = None,
        ph2_metadata_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        self.available_datasets = datasets or list(self.SUPPORTED_DATASETS)
        for ds in self.available_datasets:
            if ds not in self.SUPPORTED_DATASETS:
                raise ValueError(
                    f"Unknown dataset '{ds}'. Must be one of {self.SUPPORTED_DATASETS}"
                )

        self._metadata_paths = {
            "isic2018": isic2018_metadata_path,
            "ham10000": ham10000_metadata_path,
            "ph2": ph2_metadata_path,
        }

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "dermoscopy.yaml"

        # Prepare unified metadata CSV if it doesn't exist
        metadata_csv = os.path.join(root, "dermoscopy-metadata-pyhealth.csv")
        if not os.path.exists(metadata_csv):
            self.prepare_metadata(root)

        default_tables = ["dermoscopy"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "dermoscopy",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self, root: str) -> None:
        """Prepare unified metadata CSV from all enabled sub-datasets.

        Reads each sub-dataset's metadata, normalizes column names and paths,
        then concatenates everything into a single CSV with the schema:
            patient_id, image_path, mask_path, label, source_dataset

        Args:
            root: Root directory containing sub-dataset folders.
        """
        all_frames = []

        if "isic2018" in self.available_datasets:
            df = self._prepare_isic2018(root)
            if df is not None:
                all_frames.append(df)

        if "ham10000" in self.available_datasets:
            df = self._prepare_ham10000(root)
            if df is not None:
                all_frames.append(df)

        if "ph2" in self.available_datasets:
            df = self._prepare_ph2(root)
            if df is not None:
                all_frames.append(df)

        if not all_frames:
            raise ValueError(
                "No metadata could be loaded. Provide at least one metadata "
                "CSV path for the enabled datasets."
            )

        combined = pd.concat(all_frames, axis=0, ignore_index=True)
        output_path = os.path.join(root, "dermoscopy-metadata-pyhealth.csv")
        combined.to_csv(output_path, index=False)
        logger.info(
            f"Saved combined metadata ({len(combined)} samples) to {output_path}"
        )

    def _prepare_isic2018(self, root: str) -> Optional[pd.DataFrame]:
        """Prepare ISIC 2018 sub-dataset metadata.

        Expects the metadata CSV to have columns: "image" (filename), "label" (0/1).
        Images are JPG files; masks are PNG segmentation files.
        """
        csv_path = self._metadata_paths.get("isic2018")
        if csv_path is None:
            logger.warning("ISIC 2018 metadata path not provided, skipping.")
            return None

        df = pd.read_csv(csv_path)
        image_dir = os.path.join(root, "isic2018", "images")
        mask_dir = os.path.join(root, "isic2018", "masks")

        records = []
        for _, row in df.iterrows():
            img_name = row["image"]
            label = int(row["label"])
            image_path = os.path.join(image_dir, img_name)
            # Derive mask path: ISIC_XXXX.JPG -> ISIC_XXXX_segmentation.png
            mask_name = img_name.replace(".JPG", "_segmentation.png").replace(
                ".jpg", "_segmentation.png"
            )
            mask_path = os.path.join(mask_dir, mask_name)
            patient_id = f"isic_{Path(img_name).stem}"
            records.append(
                {
                    "patient_id": patient_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": label,
                    "source_dataset": "isic2018",
                }
            )

        result = pd.DataFrame(records)
        logger.info(f"Prepared ISIC 2018: {len(result)} samples")
        return result

    def _prepare_ham10000(self, root: str) -> Optional[pd.DataFrame]:
        """Prepare HAM10000 sub-dataset metadata.

        Expects the metadata CSV to have columns: "image_id" (base name), "label" (0/1).
        Images are {image_id}.jpg; masks are {image_id}_segmentation.png.
        """
        csv_path = self._metadata_paths.get("ham10000")
        if csv_path is None:
            logger.warning("HAM10000 metadata path not provided, skipping.")
            return None

        df = pd.read_csv(csv_path)
        image_dir = os.path.join(root, "ham10000", "images")
        mask_dir = os.path.join(root, "ham10000", "masks")

        records = []
        for _, row in df.iterrows():
            image_id = row["image_id"]
            label = int(row["label"])
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            mask_path = os.path.join(mask_dir, f"{image_id}_segmentation.png")
            patient_id = f"ham_{image_id}"
            records.append(
                {
                    "patient_id": patient_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": label,
                    "source_dataset": "ham10000",
                }
            )

        result = pd.DataFrame(records)
        logger.info(f"Prepared HAM10000: {len(result)} samples")
        return result

    def _prepare_ph2(self, root: str) -> Optional[pd.DataFrame]:
        """Prepare PH2 sub-dataset metadata.

        Expects the metadata CSV to have columns: "Name" (folder name), "label" (0/1).
        Images follow nested BMP structure:
            {root}/ph2/{Name}/{Name}_Dermoscopic_Image/{Name}.bmp
        Masks:
            {root}/ph2/{Name}/{Name}_lesion/{Name}_lesion.bmp
        """
        csv_path = self._metadata_paths.get("ph2")
        if csv_path is None:
            logger.warning("PH2 metadata path not provided, skipping.")
            return None

        df = pd.read_csv(csv_path)
        ph2_dir = os.path.join(root, "ph2")

        records = []
        for _, row in df.iterrows():
            name = row["Name"]
            label = int(row["label"])
            image_path = os.path.join(
                ph2_dir, name, f"{name}_Dermoscopic_Image", f"{name}.bmp"
            )
            mask_path = os.path.join(
                ph2_dir, name, f"{name}_lesion", f"{name}_lesion.bmp"
            )
            patient_id = f"ph2_{name}"
            records.append(
                {
                    "patient_id": patient_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": label,
                    "source_dataset": "ph2",
                }
            )

        result = pd.DataFrame(records)
        logger.info(f"Prepared PH2: {len(result)} samples")
        return result

    @property
    def default_task(self) -> DermoscopyMelanomaClassification:
        """Returns the default task for this dataset.

        Returns:
            DermoscopyMelanomaClassification: Binary melanoma classification task.
        """
        return DermoscopyMelanomaClassification()

    @wraps(BaseDataset.set_task)
    def set_task(self, *args, **kwargs):
        """Override to inject default DermoscopyImageProcessor if none provided.

        If no input processor is specified for the "image" key, a default
        DermoscopyImageProcessor with mode="whole" is injected.
        """
        input_processors = kwargs.get("input_processors", None)

        if input_processors is None:
            input_processors = {}

        if "image" not in input_processors:
            input_processors["image"] = DermoscopyImageProcessor(mode="whole")

        kwargs["input_processors"] = input_processors

        return super().set_task(*args, **kwargs)
