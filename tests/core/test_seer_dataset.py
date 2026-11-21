"""Verify the SEER dataset with synthetic tumor records.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

from pathlib import Path

import pytest

from pyhealth.datasets import SEERDataset

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "seer"


def test_loads_tumor_events_by_patient(seer_dataset: SEERDataset) -> None:
    dataset = seer_dataset
    assert sorted(dataset.unique_patient_ids) == [
        "00001", "00002", "00003", "00004"
    ]
    events = dataset.get_patient("00001").get_events(event_type="tumor")
    assert [(event.record_number, event.survival_months) for event in events] == [
        ("01", 8),
        ("02", 12),
    ]
    assert events[0].cause_of_death == "cancer_death"
    assert events[0].diagnosis_year == 2004
    assert dataset.get_patient("00003").get_events()[0].cause_of_death == "unknown"
    assert dataset.get_patient("00004").get_events()[0].survival_months is None


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("Survival months", "Missing", "Missing SEER export columns"),
        ("00001,02", "00001,01", "Duplicate tumor key"),
        ("00001,02", ",02", "Incomplete tumor key"),
        (",8,Dead", ",-8,Dead", "Invalid survival months"),
        ("Dead (missing/unknown COD)", "Unrecognized", "Unrecognized cause of death"),
        (",65-69,", ",bad-age,", "Invalid SEER age_group"),
        (",8140,", ",bad-histology,", "Invalid SEER histology"),
        (",III,Left,", ",bad-stage,Left,", "Invalid SEER stage"),
    ],
)
def test_rejects_invalid_exports(
    tmp_path: Path, original: str, replacement: str, message: str
) -> None:
    filename = "seer_export.txt"
    contents = (FIXTURE / filename).read_text(encoding="ascii")
    (tmp_path / filename).write_text(
        contents.replace(original, replacement, 1), encoding="ascii"
    )
    dataset = SEERDataset(tmp_path, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match=message):
        dataset.load_table("tumor")
