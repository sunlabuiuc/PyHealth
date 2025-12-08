"""
PyHealth dataset class for the HAM10000 dermoscopic lesion dataset.

Dataset link:
    https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Dataset paper:
    Philipp Tschandl et al. "HAM10000 Dataset: A large collection
    of multi-source dermatoscopic images of pigmented lesions."
    Scientific Data, 2018.

This dataset contains 10,015 dermoscopic lesion images across 7 classes:
    akiec: Actinic keratoses and intraepithelial carcinoma
    bcc: Basal cell carcinoma
    bkl: Benign keratosis-like lesions
    df: Dermatofibroma
    mel: Melanoma
    nv: Melanocytic nevi
    vasc: Vascular lesions

Author:
    Kacper Dural
"""

import os
import logging
from pathlib import Path
from typing import Optional, List

import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.processors import ImageProcessor
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class HAM10000Dataset(BaseDataset):
    """Dataset class for the HAM10000 dermoscopic image dataset."""

    classes: List[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = None,
    ):
        """
        Args:
            root: dataset root directory containing:
                - images/ folder with .jpg images
                - metadata.csv describing samples

            config_path: path to dataset config (optional)
        """
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.meta_path = os.path.join(root, "HAM10000_metadata.csv")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError("metadata.csv not found in dataset root.")

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError("images/ folder not found in dataset root.")

        # build internal metadata table
        self._index_data()

        super().__init__(
            root=root,
            tables=["ham10000"],
            dataset_name="HAM10000",
            config_path=config_path,
        )

    @property
    def default_task(self) -> BaseTask:
        """Default task: multiclass dermatology label classification."""
        from pyhealth.tasks import image_multiclass_classification_fn
        return image_multiclass_classification_fn

    def set_task(self, *args, **kwargs):
        """Attach an ImageProcessor if user does not supply one."""
        input_processors = kwargs.get("input_processors", None) or {}
        if "image" not in input_processors:
            input_processors["image"] = ImageProcessor(
                image_size=224,
                mode="RGB",
            )
        kwargs["input_processors"] = input_processors
        return super().set_task(*args, **kwargs)

    def _index_data(self) -> None:
        """Reads metadata.csv and builds ham10000-metadata-pyhealth.csv."""

        df = pd.read_csv(self.meta_path)

        if "image_id" not in df.columns or "dx" not in df.columns:
            raise ValueError("metadata.csv must contain image_id and dx.")

        # Full paths
        df["path"] = df["image_id"].apply(
            lambda x: os.path.join(self.image_dir, f"{x}.jpg")
        )

        # Verify images exist
        missing = df[~df["path"].apply(os.path.exists)]
        if len(missing) > 0:
            logger.warning(f"{len(missing)} images listed in metadata were not found.")

        df.rename(columns={
            "dx": "label",
            "image_id": "image",
        }, inplace=True)

        # Save cleaned metadata file for reproducibility
        out_path = os.path.join(self.root, "ham10000-metadata-pyhealth.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved processed metadata to {out_path}")
