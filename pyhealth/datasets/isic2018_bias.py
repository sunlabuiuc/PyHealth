import logging
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class ISIC2018BiasDataset(BaseDataset):
    """Dataset for ISIC bias with artifcat annotations for ISIC 2018 - Task1/2 (2594 images)

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
    [2] Alceu Bissoto, Eduardo Valle, "Sandra Avila Debiasing Skin Lesion Datasets and Models? Not So Fast", 2020;
      https://doi.org/10.48550/arXiv.2004.11457

    Args:
    root (str): The root directory where the dataset CSV file is stored.
    tables (List[str]): A list of tables to be included (typically ["isic2018"]).
    dataset_name (Optional[str]): The name of the dataset. Defaults to "isic2018".
    config_path (Optional[str]): The path to the configuration file. If not provided,
        uses the default config.
    **kwargs: Additional arguments passed to BaseDataset.

    Examples:
        >>> from pyhealth.datasets import ISIC2018BiasDataset
        >>> dataset = ISIC2018BiasDataset(
        ...     root="/path/to/isic2018/data",
        ...     tables=["isic2018_artifacts"]
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
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "isic_2018_bias.yaml"
        default_tables = ["isic2018_artifacts"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "isic_bias_metadata",
            config_path=config_path,
        )
        return