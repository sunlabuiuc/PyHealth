"""Shared fixtures for core tests.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pyhealth.datasets import SEERDataset


@pytest.fixture
def seer_dataset(tmp_path: Path) -> "SEERDataset":
    """Build the tiny event cache in-process without starting Dask workers."""
    from pyhealth.datasets import SEERDataset

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "seer"
    dataset = SEERDataset(fixture, cache_dir=tmp_path)
    dataset.load_data().to_parquet(
        dataset.cache_dir / "global_event_df.parquet",
        write_index=False,
        compute_kwargs={"scheduler": "synchronous"},
    )
    return dataset
