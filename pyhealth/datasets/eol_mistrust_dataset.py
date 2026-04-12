"""Native BaseDataset entrypoint for the EOL mistrust cohort tables."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import narwhals as pl

from .base_dataset import BaseDataset
from .configs import load_yaml_config

logger = logging.getLogger(__name__)

DATASET_PREPARE_MODE_DEFAULT = "default"
DATASET_PREPARE_MODE_PAPER_LIKE = "paper_like"

_ROUTE_SETTINGS = {
    DATASET_PREPARE_MODE_DEFAULT: {
        "paper_like_dataset_prepare": False,
        "code_status_mode": "corrected",
        "autopsy_label_mode": "corrected",
    },
    DATASET_PREPARE_MODE_PAPER_LIKE: {
        "paper_like_dataset_prepare": True,
        "code_status_mode": "paper_like",
        "autopsy_label_mode": "paper_like",
    },
}


class EOLMistrustDataset(BaseDataset):
    """PyHealth dataset wrapper for the combined EOL mistrust CSV export tree.

    This dataset provides a proper :class:`~pyhealth.datasets.BaseDataset`
    entrypoint for the custom EOL mistrust replication tables stored under a
    combined root such as::

        root/
            mimiciii_clinical/
            mimiciii_notes/
            mimiciii_derived/

    The default table set favors the admission-level task pipeline and only
    requires the core tables that are available in the managed workspace export.
    Optional EHR context tables such as ``diagnoses_icd`` or ``prescriptions``
    can be added via ``tables=[...]`` when they are present in the root.

    Args:
        root: Root directory containing the combined EOL mistrust export.
        tables: Additional table names to load. The dataset always includes the
            core ``patients``, ``admissions``, and ``icustays`` tables.
        dataset_name: Optional dataset name override.
        config_path: Optional YAML config path. Defaults to the bundled
            ``eol_mistrust.yaml`` config.
        **kwargs: Additional :class:`BaseDataset` keyword arguments.
    """

    CORE_TABLES = ["patients", "admissions", "icustays"]
    DEFAULT_OPTIONAL_TABLES = [
        "diagnoses_icd",
        "procedures_icd",
        "prescriptions",
        "noteevents",
        "d_items",
        "chartevents",
    ]

    @staticmethod
    def _normalize_dataset_prepare_mode(mode: str | None) -> str:
        normalized = (
            DATASET_PREPARE_MODE_DEFAULT
            if mode is None
            else str(mode).strip().lower()
        )
        if normalized not in _ROUTE_SETTINGS:
            raise ValueError(
                "dataset_prepare_mode must be one of "
                f"{DATASET_PREPARE_MODE_DEFAULT!r} or "
                f"{DATASET_PREPARE_MODE_PAPER_LIKE!r}"
            )
        return normalized

    @staticmethod
    def _path_variants(root: str, relative_path: str) -> list[Path]:
        csv_path = Path(root) / relative_path
        if csv_path.suffix == ".gz":
            return [csv_path, csv_path.with_suffix("")]
        return [csv_path, Path(f"{csv_path}.gz")]

    @classmethod
    def _table_assets_exist(cls, root: str, config, table_name: str) -> bool:
        if table_name not in config.tables:
            return False

        table_cfg = config.tables[table_name]
        required_paths = [table_cfg.file_path]
        join_cfg = getattr(table_cfg, "join", None) or []
        required_paths.extend(join.file_path for join in join_cfg)

        for relative_path in required_paths:
            if not any(
                path.exists() for path in cls._path_variants(root, relative_path)
            ):
                return False
        return True

    @classmethod
    def _discover_optional_tables(
        cls,
        root: str,
        config_path: str,
    ) -> list[str]:
        config = load_yaml_config(config_path)
        available_tables: list[str] = []
        for table_name in cls.DEFAULT_OPTIONAL_TABLES:
            if cls._table_assets_exist(root, config, table_name):
                available_tables.append(table_name)
        return available_tables

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dataset_prepare_mode: str = DATASET_PREPARE_MODE_DEFAULT,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default EOL mistrust config")
            config_path = str(
                Path(__file__).parent / "configs" / "eol_mistrust.yaml"
            )

        self.dataset_prepare_mode = self._normalize_dataset_prepare_mode(
            dataset_prepare_mode
        )
        route_settings = _ROUTE_SETTINGS[self.dataset_prepare_mode]
        self.paper_like_dataset_prepare = bool(
            route_settings["paper_like_dataset_prepare"]
        )
        self.code_status_mode = str(route_settings["code_status_mode"])
        self.autopsy_label_mode = str(route_settings["autopsy_label_mode"])

        if tables is None:
            requested_tables = self._discover_optional_tables(root, config_path)
        else:
            requested_tables = list(tables)
        resolved_tables: list[str] = []
        for table_name in [*self.CORE_TABLES, *requested_tables]:
            if table_name not in resolved_tables:
                resolved_tables.append(table_name)

        super().__init__(
            root=root,
            tables=resolved_tables,
            dataset_name=dataset_name or "eol_mistrust",
            config_path=config_path,
            **kwargs,
        )

    def preprocess_noteevents(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Fill missing note ``charttime`` values from ``chartdate``.

        MIMIC-III note rows may omit ``charttime`` while still providing a
        ``chartdate``. PyHealth requires a single timestamp column for event
        ordering, so we backfill missing times with midnight of the chart date.

        Args:
            df: Lazy noteevents frame before BaseDataset event normalization.

        Returns:
            LazyFrame with a populated ``charttime`` column.
        """
        columns = set(df.collect_schema().names())
        if "charttime" not in columns:
            raise ValueError("noteevents must include charttime for EOLMistrustDataset")
        if "chartdate" not in columns:
            return df

        return df.with_columns(
            pl.when(pl.col("charttime").is_null())
            .then(pl.col("chartdate") + pl.lit(" 00:00:00"))
            .otherwise(pl.col("charttime"))
            .alias("charttime")
        )

__all__ = [
    "DATASET_PREPARE_MODE_DEFAULT",
    "DATASET_PREPARE_MODE_PAPER_LIKE",
    "EOLMistrustDataset",
]
