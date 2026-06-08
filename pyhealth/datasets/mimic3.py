import logging
import warnings
from pathlib import Path
from typing import List, Optional

import narwhals as pl

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC3Dataset(BaseDataset):
    """
    A dataset class for handling MIMIC-III data.

    This class is responsible for loading and managing the MIMIC-III dataset,
    which includes tables such as patients, admissions, and icustays.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> # Load MIMIC-III dataset with clinical tables
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "labevents"],
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the MIMIC4Dataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "mimic3".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3.yaml"
        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + tables
        if "prescriptions" in tables:
            warnings.warn(
                "Events from prescriptions table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3",
            config_path=config_path,
            **kwargs,
        )
        return

    def preprocess_noteevents(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Table-specific preprocess function which will be called by BaseDataset.load_table().

        Preprocesses the noteevents table by ensuring that the charttime column
        is populated. If charttime is null, it uses chartdate with a default
        time of 00:00:00.

        See: https://mimic.mit.edu/docs/iii/tables/noteevents/#chartdate-charttime-storetime.

        Args:
            df (pl.LazyFrame): The input dataframe containing noteevents data.

        Returns:
            pl.LazyFrame: The processed dataframe with updated charttime
            values.
        """
        df = df.with_columns(
            pl.when(pl.col("charttime").is_null())
            .then(pl.col("chartdate") + pl.lit(" 00:00:00"))
            .otherwise(pl.col("charttime"))
            .alias("charttime")
        )
        return df


class MIMIC3NoteDataset(BaseDataset):
    """MIMIC-III clinical notes dataset for evidence retrieval tasks.

    This dataset specialises the MIMIC-III data loading for NLP and evidence
    retrieval use-cases. It always loads the ``noteevents`` and
    ``diagnoses_icd`` tables alongside the core demographic tables
    (``patients``, ``admissions``), providing everything needed for the
    zero-shot EHR evidence retrieval pipeline introduced by
    `Ahsan et al. (2024) <https://arxiv.org/abs/2309.04550>`_.

    Compared with the general :class:`MIMIC3Dataset`, this class:

    - Uses a dedicated YAML config (``mimic3_note.yaml``) that exposes the
      ``iserror`` flag on note events so erroneous notes can be filtered
      downstream.
    - Always includes ``noteevents`` and ``diagnoses_icd`` so that tasks can
      pair note text with ICD-9 condition labels without extra configuration.
    - Applies ``preprocess_noteevents`` to fill missing ``charttime`` values
      from ``chartdate``.

    Args:
        root (str): Root directory of the MIMIC-III 1.4 release (the folder
            that contains ``NOTEEVENTS.csv.gz``, ``PATIENTS.csv.gz``, etc.).
        tables (List[str]): Additional tables to load beyond the defaults
            (``patients``, ``admissions``, ``diagnoses_icd``, ``noteevents``).
        dataset_name (str): Name used for cache-directory keying.
            Defaults to ``"mimic3_note"``.
        config_path (Optional[str]): Path to an alternative YAML config.
            When ``None`` (default) the bundled ``mimic3_note.yaml`` is used.
        **kwargs: Forwarded verbatim to :class:`~pyhealth.datasets.BaseDataset`
            (e.g. ``dev``, ``cache_dir``, ``num_workers``).

    Examples:
        >>> from pyhealth.datasets import MIMIC3NoteDataset
        >>> dataset = MIMIC3NoteDataset(
        ...     root="/path/to/mimic-iii/1.4",
        ... )
        >>> dataset.stats()

        Load with extra tables and developer mode:

        >>> dataset = MIMIC3NoteDataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["procedures_icd"],
        ...     dev=True,
        ... )
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: str = "mimic3_note",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialise the MIMIC3NoteDataset.

        Args:
            root (str): Root directory of the MIMIC-III 1.4 release.
            tables (Optional[List[str]]): Extra tables to load on top of the
                defaults. Defaults to ``None`` (only defaults are loaded).
            dataset_name (str): Cache-key name. Defaults to
                ``"mimic3_note"``.
            config_path (Optional[str]): Override the bundled YAML config.
            **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "mimic3_note.yaml"
            logger.info("Using default MIMIC-III note config: %s", config_path)

        default_tables = ["patients", "admissions", "diagnoses_icd", "noteevents"]
        extra = list(tables) if tables else []
        all_tables = default_tables + [t for t in extra if t not in default_tables]

        super().__init__(
            root=root,
            tables=all_tables,
            dataset_name=dataset_name,
            config_path=str(config_path),
            **kwargs,
        )

    def preprocess_noteevents(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Fill missing ``charttime`` from ``chartdate`` and cast ``iserror``.

        MIMIC-III note events sometimes have a null ``charttime``; the
        original ``chartdate`` is used with a midnight default in that case.
        The ``iserror`` column is coerced to a string so that downstream code
        can safely compare it against ``"1"`` without worrying about dtype.

        Args:
            df (pl.LazyFrame): Raw note-events lazy frame as loaded by the
                base class.

        Returns:
            pl.LazyFrame: Processed frame with ``charttime`` and ``iserror``
            normalised.
        """
        df = df.with_columns(
            pl.when(pl.col("charttime").is_null())
            .then(pl.col("chartdate") + pl.lit(" 00:00:00"))
            .otherwise(pl.col("charttime"))
            .alias("charttime")
        )
        if "iserror" in df.columns:
            df = df.with_columns(
                pl.col("iserror").cast(pl.String).alias("iserror")
            )
        return df
