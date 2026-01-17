import logging
from pathlib import Path
from typing import Optional
import os

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class ISICBiasDataset(BaseDataset):
    """Dataset for ISIC bias with artifact annotations for ISIC 2018 - Task1/2

    Dataset is available at: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation

    Artifact Annotations Metadata is available at:
    https://github.com/alceubissoto/debiasing-skin/blob/master/artefacts-annotation/isic_bias.csv
    The tool used for this annotation is available at: https://github.com/phillipecardenuto/VisualizationLib

    Citations:
    -------------
    If you use this data, please refer to:
    [1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba,
        Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern:
        “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging
        Collaboration (ISIC)”, 2018; https://arxiv.org/abs/1902.03368

    Artifact annotations reference:
    [2] Alceu Bissoto, Eduardo Valle, Sandra Avila "Debiasing Skin Lesion Datasets and Models? Not So Fast", 2020;
        https://doi.org/10.48550/arXiv.2004.11457

    Args:
    root (str): The root directory where the dataset CSV file is stored.
    tables (List[str]): A list of tables to be included (typically ["isic_artifacts_raw"]).
    dataset_name (Optional[str]): The name of the dataset. Defaults to "isic_bias".
    config_path (Optional[str]): The path to the configuration file. If not provided,
        uses the default config.
    **kwargs: Additional arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import ISICBiasDataset
        >>> dataset = ISICBiasDataset(
        ...     root="/path/to/isic_artifacts_raw/data",
        ...     tables=["isic_artifacts_raw"]
        ... )
        >>> dataset.stats()
    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (str): The name of the dataset.
        config_path (str): The path to the configuration file.
    """

    def __init__(
            self,
            root: str,
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
    ) -> None:
        root_path = Path(root)
        images_path = root_path / "images"
        normalize_csv_delimiters(root_path)
        normalize_image_extensions(root_path, images_path)

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "isic_bias.yaml"
        default_tables = ["isic_artifacts_raw"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "isic_bias",
            config_path=config_path,
        )
        return


def normalize_csv_delimiters(root_path: Path):
    """
    If any CSVs under the root folder use semicolon (;) delimiters,
    convert them into standard comma-separated CSVs in-place.
    This is done because the original dataset uploaded by the authors is ; delimited
    """
    for csv_file in root_path.rglob("*.csv"):
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                first_line = f.readline()

            # Detect semicolon formatting
            if ";" in first_line and "," not in first_line:
                logger.info(f"Normalizing semicolon-delimited CSV: {csv_file}")

                # Read with semicolon delimiter
                df = pd.read_csv(csv_file, sep=";")

                df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
                # Rewrite as comma CSV
                df.to_csv(csv_file, index=False)
        except Exception as e:
            logger.warning(f"Failed to check/normalize CSV {csv_file}: {e}")


def normalize_image_extensions(root_path: Path, images_path: Path) -> None:
    """
    Normalize image metadata when CSV references .png but actual images on disk
    are .jpg/.jpeg (or vice versa).

    Assumes CSV column `image` contains only the filename (e.g., "ISIC_10.png").
    Looks for files under `images_path`.
    Updates CSVs in-place only when a matching file exists.
    """
    IMAGE_COL = "image"
    IMAGE_EXTS = (".png", ".jpg", ".jpeg")

    for csv_file in root_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if IMAGE_COL not in df.columns:
                continue

            updated = False

            for idx, image_val in df[IMAGE_COL].items():
                if pd.isna(image_val):
                    continue

                raw_name = str(image_val).strip()
                if not raw_name:
                    continue

                rel_name = Path(raw_name).name  # enforce filename-only
                abs_img_path = images_path / rel_name

                # If referenced file exists, nothing to do
                if abs_img_path.exists():
                    continue

                p = Path(rel_name)
                stem = p.stem if p.suffix else p.name

                # Try alternate extensions under images_path
                found_name = None
                for ext in IMAGE_EXTS:
                    candidate = images_path / f"{stem}{ext}"
                    if candidate.exists():
                        found_name = f"{stem}{ext}"
                        break

                if found_name is not None:
                    df.at[idx, IMAGE_COL] = found_name
                    updated = True
                    logger.info(
                        f"Normalized image extension in {csv_file.name}: "
                        f"{raw_name} → {found_name}"
                    )

            if updated:
                df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
                df.to_csv(csv_file, index=False)

        except Exception as e:
            logger.warning(f"Failed to normalize image extensions in {csv_file}: {e}")
