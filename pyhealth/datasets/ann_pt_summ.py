import logging
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ANNPTSummDataset(BaseDataset):
    """Annotated patient summary dataset from PhysioNet's ann-pt-summ release.

    Each table corresponds to a newline-delimited JSON file containing at least
    the discharge ``text`` (context) and ``summary`` fields, plus optional
    ``labels`` for hallucination spans or other metadata.

    Args:
        root: Root directory containing the ann-pt-summ release.
        tables: List of table names defined in ``ann_pt_summ.yaml``.
            Defaults to ``["mimic_di_bhc_all"]`` (all Brief Hospital Course
            summaries).
        dataset_name: Optional dataset name override.
        config_path: Optional path to the dataset config yaml.
        **kwargs: Passed to :class:`BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import ANNPTSummDataset
        >>> dataset = ANNPTSummDataset(
        ...     root="/tmp/ann-pt-summ/1.0.1",
        ...     tables=["hallucinations_mimic_di"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "ann_pt_summ.yaml"
            logger.info("Using default ann-pt-summ config: %s", config_path)

        default_tables = tables or ["mimic_di_bhc_all"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "ann_pt_summ",
            config_path=str(config_path),
            **kwargs,
        )

    @property
    def default_task(self):
        """Dataset currently ships without a default downstream task."""
        return None
