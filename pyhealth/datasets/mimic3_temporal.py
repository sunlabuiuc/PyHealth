"""
Author: Simona Bernfeld
Paper: Feature Robustness in Non-stationary Health Records:
Caveats to Deployable Model Performance in Common Clinical Machine Learning Tasks
Paper link: https://proceedings.mlr.press/v106/nestor19a.html

Description:
Thin dataset wrapper around MIMIC3Dataset for temporal robustness experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from pyhealth.datasets import MIMIC3Dataset


@dataclass
class TemporalSplit:
    """Container for temporal split indices."""
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]


class MIMIC3TemporalDataset(MIMIC3Dataset):
    """MIMIC-III dataset wrapper for temporal experiments.

    Args:
        root: Root path to the MIMIC-III files.
        tables: Tables to load.
        dataset_name: Optional dataset name.
        config_path: Optional config file path.
        **kwargs: Additional keyword arguments forwarded to MIMIC3Dataset.
    """

    REQUIRED_TABLES: Tuple[str, ...] = (
        "diagnoses_icd",
        "procedures_icd",
        "prescriptions",
    )

    def __init__(
        self,
        root: str,
        tables: Sequence[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._validate_tables(tables)
        super().__init__(
            root=root,
            tables=list(tables),
            dataset_name=dataset_name or "mimic3_temporal",
            config_path=config_path,
            **kwargs,
        )

    @classmethod
    def _validate_tables(cls, tables: Sequence[str]) -> None:
        """Checks that all required tables are present."""
        missing = [table for table in cls.REQUIRED_TABLES if table not in tables]
        if missing:
            raise ValueError(
                "MIMIC3TemporalDataset requires tables "
                f"{cls.REQUIRED_TABLES}, but missing {missing}."
            )

    @staticmethod
    def normalize_year(year: int, min_year: int = 2001, max_year: int = 2012) -> float:
        """Normalizes a year to [0, 1]."""
        if max_year <= min_year:
            raise ValueError("max_year must be greater than min_year.")
        clipped = min(max(year, min_year), max_year)
        return (clipped - min_year) / float(max_year - min_year)

    @staticmethod
    def temporal_split_from_years(
        years: Sequence[int],
        train_end_year: int,
        val_end_year: int,
    ) -> TemporalSplit:
        """Builds temporal train/val/test split indices."""
        train_idx, val_idx, test_idx = [], [], []

        for idx, year in enumerate(years):
            if year <= train_end_year:
                train_idx.append(idx)
            elif year <= val_end_year:
                val_idx.append(idx)
            else:
                test_idx.append(idx)

        if not train_idx or not val_idx or not test_idx:
            raise ValueError(
                "Temporal split produced an empty partition. "
                "Choose different year boundaries."
            )

        return TemporalSplit(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
