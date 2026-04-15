import pytest

from pyhealth.datasets.mimic3_temporal import MIMIC3TemporalDataset, TemporalSplit


def test_normalize_year() -> None:
    value = MIMIC3TemporalDataset.normalize_year(2006, min_year=2001, max_year=2012)
    assert 0.0 <= value <= 1.0


def test_temporal_split_from_years() -> None:
    split = MIMIC3TemporalDataset.temporal_split_from_years(
        years=[2001, 2002, 2008, 2010],
        train_end_year=2002,
        val_end_year=2008,
    )
    assert isinstance(split, TemporalSplit)
    assert split.train_idx == [0, 1]
    assert split.val_idx == [2]
    assert split.test_idx == [3]


def test_temporal_split_raises_on_empty_partition() -> None:
    with pytest.raises(ValueError):
        MIMIC3TemporalDataset.temporal_split_from_years(
            years=[2001, 2001, 2001],
            train_end_year=2001,
            val_end_year=2002,
        )
