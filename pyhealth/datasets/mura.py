import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# from ..tasks import MURAClassification  # You need to implement this
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MURADataset(BaseDataset):
    """MURA Dataset for musculoskeletal radiographs.

    Dataset is available at:
    https://stanfordmlgroup.github.io/competitions/mura/

    The dataset contains 7 anatomical regions:
        - ELBOW, FINGER, FOREARM, HAND, HUMERUS, SHOULDER, WRIST

    Each study is labeled as either positive (abnormal, label=1) or negative (normal, label=0).

    Args:
        root: Root directory of the MURA dataset (should contain MURA-v1.1/...).
        split: One of ["train", "valid", "all"].
        dataset_name: Optional name of the dataset. Defaults to "mura".
        config_path: Optional path to the configuration file. Defaults to "configs/mura.yaml".

    Example:
        >>> dataset = MURADataset(root="/path/to/MURA-v1.1", split="train")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev : Optional[bool] = False
    ) -> None:
        # self.split = split
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "mura.yaml"

        if not os.path.exists(os.path.join(root, "mura-metadata-pyhealth.csv")):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["mura"],
            dataset_name=dataset_name or "mura",
            config_path=config_path,
            dev = dev
        )

        # Apply split filtering
        # if self.split in ["train", "valid"]:
        #     split_prefix = os.path.join(root, self.split)
        #     self.tables["mura"].df = self.tables["mura"].df[
        #         self.tables["mura"].df["path"].str.startswith(split_prefix)
        #     ]
        # elif self.split != "all":
        #     raise ValueError("split must be one of ['train', 'valid', 'all']")

    def prepare_metadata(self, root: str) -> None:
        """Generate mura-metadata-pyhealth.csv from raw CSVs."""

        def read_study_labels(file_path):
            df = pd.read_csv(os.path.join(root, file_path), header=None)
            df.columns = ["study_path", "label"]
            return dict(zip(df.study_path.str.strip(), df.label))

        def build_df(image_path_file, label_dict):
            df = pd.read_csv(os.path.join(root, image_path_file), header=None)
            df.columns = ["path"]
            df["study_path"] = df["path"].apply(lambda x: os.path.dirname(x).strip() + '/')
            df["label"] = df["study_path"].map(label_dict)
            df = df.dropna()

            df["patient_id"] = df["path"].apply(lambda x: x.split("/")[2])
            df["visit_id"] = df["path"].apply(lambda x: x.split("/")[3])
            df["path"] = df["path"].apply(lambda x: x.replace("MURA-v1.1/", "", 1))
            df["path"] = df["path"].apply(lambda x: str(Path(root) / Path(x)))

            return df[["path", "label", "patient_id", "visit_id"]]

        train_labels = read_study_labels("train_labeled_studies.csv")
        valid_labels = read_study_labels("valid_labeled_studies.csv")
        train_df = build_df("train_image_paths.csv", train_labels)
        valid_df = build_df("valid_image_paths.csv", valid_labels)

        full_df = pd.concat([train_df, valid_df], ignore_index=True)

        for path in full_df["path"]:
            assert os.path.isfile(path), f"Missing file: {path}"

        metadata_path = os.path.join(root, "mura-metadata-pyhealth.csv")
        full_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved MURA metadata to {metadata_path}")

    # @property
    # def default_task(self) -> MURAClassification:
    #     """Returns the default classification task."""
    #     return MURAClassification()
