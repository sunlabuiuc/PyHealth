"""TCGA-CRCk dataset loader for PyHealth."""

import csv
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TCGACRCkDataset(BaseDataset):
    """Dataset class for the TCGA-CRCk dataset.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        cache_dir (str): Path to the cache directory.
        num_workers (int): Number of worker processes.
        dev (bool): Whether the dataset is in development mode.
    """

    classes: List[str] = ["MSIMUT", "MSS"]

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = "tcga_crck",
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: Optional[int] = None,
        dev: bool = False,
    ) -> None:
        """Initializes the TCGA-CRCk dataset.

        Args:
            root: Root directory containing the raw TCGA-CRCk image folders.
            dataset_name: Dataset name used by PyHealth cache management.
            config_path: Optional YAML config path. If omitted, the bundled
                tcga_crck.yaml config is used.
            cache_dir: Optional cache directory for processed artifacts.
            num_workers: Number of workers used by the PyHealth processing
                pipeline.
            dev: Whether to enable development-mode shortcuts.
        """
        self.root = root

        if num_workers is None:
            num_workers = 1

        self._verify_root()

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "tcga_crck.yaml"

        self.metadata_path = os.path.join(
            self.root,
            "tcga_crck_metadata-pyhealth.csv",
        )

        super().__init__(
            root=root,
            tables=["tcga_crck"],
            dataset_name=dataset_name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

        if not os.path.exists(self.metadata_path):
            logger.info("Preparing TCGA_CRCk metadata...")
            self.prepare_metadata(self.root)

        self._verify_metadata()

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Creates the PyHealth metadata CSV from the raw image folders.

        Args:
            root: Root directory containing CRC_DX_TRAIN and
                CRC_DX_TEST folders with class-specific PNG tiles.
        """
        wsi_regex = re.compile(r"^blk-.+-(TCGA-..-....-...-..-...)\.png$")
        csv_path = Path(os.path.join(root, "tcga_crck_metadata-pyhealth.csv"))
        csv_data = []
        for split in ["TRAIN", "TEST"]:
            for label_name in TCGACRCkDataset.classes:
                raw_dir = Path(os.path.join(root, f"CRC_DX_{split}", label_name))
                if not raw_dir.is_dir():
                    logger.warning(
                        "Unexpected format for raw TCGA-CRCk dataset."
                        f"Expected directory at {raw_dir}"
                    )
                    continue
                wsi_dict = {}
                for tile_path in raw_dir.glob("*.png"):
                    match = wsi_regex.search(tile_path.name)
                    if match:
                        slide_id = match.group(1)

                        tile_index = wsi_dict.get(slide_id, 0)
                        wsi_dict[slide_id] = tile_index + 1

                        csv_data.append(
                            {
                                "patient_id": slide_id[:12],
                                "slide_id": slide_id,
                                "tile_path": str(tile_path.resolve()),
                                "tile_index": tile_index,
                                "data_split": split.lower(),
                                "label": (
                                    1
                                    if label_name == "MSIMUT"
                                    else 0
                                ),
                            }
                        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fields = [
                "patient_id",
                "slide_id",
                "tile_path",
                "tile_index",
                "data_split",
                "label",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_data)

    def _verify_root(self) -> None:
        """Verifies that the dataset root exists."""
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

    def _verify_metadata(self) -> None:
        """Verifies that the normalized metadata file is present and valid."""
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(
                f"Dataset metadata file does not exist: {self.metadata_path}"
            )

        df = pd.read_csv(self.metadata_path)
        required_cols = {
            "patient_id",
            "slide_id",
            "tile_path",
            "tile_index",
            "data_split",
            "label",
        }
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(
                "Metadata file is missing required columns: "
                f"{sorted(missing)}"
            )
        if df.empty:
            logger.warning("Metadata file is empty.")

        nonexistent_paths = [
            p for p in df["tile_path"].tolist() if not os.path.isfile(str(p))
        ]
        if nonexistent_paths:
            raise FileNotFoundError(
                "Some metadata paths do not exist. Example: "
                f"{nonexistent_paths[0]}"
            )

        invalid_splits = set(df["data_split"].unique()).difference(
            {"train", "test"}
        )
        if invalid_splits:
            raise ValueError(
                f"data_split must be train/test. Found: {invalid_splits}"
            )
            
        invalid_labels = set(df["label"].unique()).difference({0, 1})
        if invalid_labels:
            raise ValueError(
                f"label must be binary 0/1. Found: {invalid_labels}"
            )

    @property
    def default_task(self):
        """Returns the default task for this dataset."""
        from pyhealth.tasks import TCGACRCkMSIClassification

        return TCGACRCkMSIClassification()
