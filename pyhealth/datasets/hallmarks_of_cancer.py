import logging
from pathlib import Path
from typing import Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class HallmarksOfCancerDataset(BaseDataset):
    """Hallmarks of Cancer (HOC) sentence corpus for multi-label text classification.

    This dataset expects a **single** CSV file under ``root`` (default filename
    ``hallmarks_of_cancer.csv``) with one row per sentence. Prepare the file by
    exporting from the BigBio Hugging Face dataset (see
    ``examples/data_prep/export_hallmarks_of_cancer_bigbio.py``) or any tool that
    produces the same columns.

    Original corpus (PubMed abstracts, sentence-level hallmark labels):
    Baker et al., *Bioinformatics* 2016. Hugging Face:
    https://huggingface.co/datasets/bigbio/hallmarks_of_cancer

    The upstream data is distributed under **GPL-3.0**; cite the original paper
    and comply with the license when redistributing derived files.

    **CSV columns**

    - ``sentence_id``: unique ID per sentence (used as ``patient_id`` in PyHealth).
    - ``document_id``: document-level identifier (e.g. PMID-based id from BigBio).
    - ``text``: sentence text.
    - ``labels``: multi-label column. Use ``##`` between labels, e.g. ``none`` or
      ``activating invasion and metastasis##sustaining proliferative signaling``.
    - ``split``: one of ``train``, ``validation``, ``test`` (Hugging Face naming).

    Args:
        root: Directory containing ``hallmarks_of_cancer.csv`` (or path given in YAML).
        dataset_name: Logical name for caching. Defaults to ``hallmarks_of_cancer``.
        config_path: Optional YAML config; defaults to the package config file.

    Examples:
        >>> from pyhealth.datasets import HallmarksOfCancerDataset
        >>> from pyhealth.tasks import HallmarksOfCancerSentenceClassification
        >>> ds = HallmarksOfCancerDataset(root="/path/to/hoc_root")
        >>> task = HallmarksOfCancerSentenceClassification(split="train")
        >>> samples = ds.set_task(task)
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir=None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default Hallmarks of Cancer config")
            config_path = (
                Path(__file__).parent / "configs" / "hallmarks_of_cancer.yaml"
            )
        super().__init__(
            root=root,
            tables=["hoc"],
            dataset_name=dataset_name or "hallmarks_of_cancer",
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @property
    def default_task(self):
        """Default task: training split, multi-label sentence classification."""
        from pyhealth.tasks import HallmarksOfCancerSentenceClassification

        return HallmarksOfCancerSentenceClassification(split="train")
