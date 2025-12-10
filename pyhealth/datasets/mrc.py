import logging
from pathlib import Path
from typing import Optional, Sequence

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class NoisyMRCICUMortalityDataset(BaseDataset):
    """ICU mortality dataset from the Noisy Minimax Risk Classifier paper.

    This dataset comes from the experiments in
    "Minimax Risk Classifiers for Mislabeled Data: a Study on Patient Outcome
    Prediction Tasks" (NoisyMRC). It contains tabular ICU data with
    demographics, vitals, lab measurements, APACHE scores, and a binary
    in-hospital mortality label.

    The original repository provides two variants of the same task:

    - ``mortality_alsocat``: includes all features with categorical variables
      expanded into one-hot columns (e.g., ``ethnicity_*``, ``icu_type_*``).
    - ``mortality_nocat``: numeric-only version with categorical variables
      removed or encoded as numeric IDs.

    In PyHealth we treat each variant as a single table and configure it
    through a YAML file (``mrc.yaml``) so that the dataset can be used with
    standard PyHealth tasks (e.g., mortality prediction).

    Args:
        root: Root directory containing the mortality CSV files. This directory
            should include at least one of:
            ``mortality_alsocat.csv``, ``mortality_nocat.csv``.
        table: Name of the table to load. Must be one of
            ``\"mortality_alsocat\"`` or ``\"mortality_nocat\"``.
        dataset_name: Optional dataset name. If not provided, a name is
            derived from the selected table (e.g., ``\"mrc_mortality_alsocat\"``).
        config_path: Optional path to the YAML config. If ``None``, uses the
            default ``mrc.yaml`` next to this file.
        dev: Whether to enable dev mode (limit the number of patients for
            quicker experiments).

    Examples:
        >>> from pyhealth.datasets import NoisyMRCICUMortalityDataset
        >>> dataset = NoisyMRCICUMortalityDataset(
        ...     root="/srv/local/data/noisy_mrc/data_mortality/",
        ...     table="mortality_alsocat",
        ...     dev=True,
        ... )
        >>> dataset.stats()
    """

    # Supported table names in the YAML config
    VALID_TABLES: Sequence[str] = ("mortality_alsocat", "mortality_nocat")

    def __init__(
        self,
        root: str,
        table: str = "mortality_alsocat",
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
    ):
        # Validate table selection early to surface clear errors
        if table not in self.VALID_TABLES:
            raise ValueError(
                f"Invalid table '{table}'. "
                f"Expected one of {list(self.VALID_TABLES)}."
            )

        # Default to the local YAML config if not provided
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "mrc.yaml"
            logger.info(f"No config_path provided, using default: {config_path}")

        # Derive a reasonable default dataset name if not given
        if dataset_name is None:
            dataset_name = f"mrc_{table}"

        logger.info(
            "Initializing NoisyMRCICUMortalityDataset with root=%s, table=%s, dev=%s",
            root,
            table,
            dev,
        )

        # Delegate actual table loading to BaseDataset
        super().__init__(
            root=root,
            tables=[table],
            dataset_name=dataset_name,
            config_path=str(config_path),
            dev=dev,
        )

    @property
    def table_name(self) -> str:
        """Return the single table loaded for this dataset."""
        # BaseDataset exposes self.tables; here we know we only passed one.
        return self.tables[0]


if __name__ == "__main__":
    # Minimal example for local testing. Users should update `root`
    # to point to their local copy of the NoisyMRC data.
    dataset = NoisyMRCICUMortalityDataset(
        root="/srv/local/data/noisy_mrc/data_mortality/",
        table="mortality_alsocat",
        dev=True,
    )
    dataset.stats()

