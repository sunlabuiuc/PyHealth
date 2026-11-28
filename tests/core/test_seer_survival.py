"""Synthetic tests for the SEER task and training-only encoders.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

import pytest
import torch

from pyhealth.datasets import SEERDataset, SubjectSplit
from pyhealth.processors.seer_processor import SEERFeatureProcessor
from pyhealth.tasks import SEERSurvivalPrediction
from pyhealth.tasks.seer_survival import SEER_EXTENT_FEATURES, SEER_FEATURES


def test_labels_and_patient_split(seer_dataset: SEERDataset) -> None:
    dataset = seer_dataset
    task = SEERSurvivalPrediction()
    patient = dataset.get_patient("00001")
    samples = task(patient)
    assert [sample["label"] for sample in samples] == [0, 1]
    assert len({sample["record_id"] for sample in samples}) == 2
    assert set(samples[0]["features"]) == set(SEER_FEATURES)
    assert [sample["label"] for sample in SEERSurvivalPrediction(60)(patient)] == [0]
    assert [sample["label"] for sample in task(dataset.get_patient("00002"))] == [1]
    assert task(dataset.get_patient("00003")) == []
    assert task(dataset.get_patient("00004")) == []
    reduced = SEERSurvivalPrediction(include_extent=False)
    assert set(samples[0]) == {
        "patient_id", "record_id", "features", "label",
    }
    assert set(samples[0]["features"]) - set(reduced(patient)[0]["features"]) == set(
        SEER_EXTENT_FEATURES
    )

    split = SubjectSplit((("00001",), ("00002",)))
    encoded_task = SEERSurvivalPrediction(encoding="one_hot")
    train, test = dataset.set_task(encoded_task, split=split)
    try:
        assert len(train) == 2 and len(test) == 1
        assert train[0]["patient_id"] == train[1]["patient_id"] == "00001"
        assert test[0]["patient_id"] == "00002"
        processor = train.input_processors["features"]
        sex = processor.field_slices["sex"]
        assert train[0]["features"][sex].tolist() == [0.0, 0.0, 1.0]
        assert test[0]["features"][sex].tolist() == [0.0, 1.0, 0.0]
        year = processor.field_slices["diagnosis_year"]
        assert test[0]["features"][year][0].item() == pytest.approx(7 / 3)
    finally:
        train.close()
        test.close()


def test_one_hot_distinguishes_measurements_and_special_values() -> None:
    processor = SEERFeatureProcessor(("tumor_size",), "one_hot")
    processor.fit(
        [{"x": {"tumor_size": size}} for size in ("020", "040", "999")], "x"
    )
    assert processor.process({"tumor_size": "030"}).tolist() == [0, 0, 0, 0]
    assert processor.process({"tumor_size": "020"}).tolist() == [-1, 0, 0, 0]
    assert processor.process({"tumor_size": ""}).tolist() == [0, 1, 0, 0]
    assert processor.process({"tumor_size": "991"}).tolist() == [0, 0, 1, 0]
    assert processor.process({"tumor_size": "999"}).tolist() == [0, 0, 0, 1]


def test_numeric_codes_and_unseen_formatted_categories() -> None:
    processor = SEERFeatureProcessor(("histology", "stage"), "numeric")
    processor.fit(
        [
            {"x": {"histology": "8000", "stage": "II"}},
            {"x": {"histology": "8100", "stage": "I"}},
            {"x": {"histology": "8400", "stage": "III"}},
        ],
        "x",
    )
    vector = processor.process({"histology": "8300", "stage": "IV"})
    assert vector.dtype == torch.float32
    assert vector[0].item() == pytest.approx(4 / 26**0.5)
    assert vector[1].item() == 0
    assert processor.process({"histology": "8100", "stage": "I"})[1] < 0


def test_numeric_keeps_special_tumor_size_codes() -> None:
    processor = SEERFeatureProcessor(("tumor_size",), "numeric")
    processor.fit(
        [{"x": {"tumor_size": size}} for size in ("010", "030", "999")], "x"
    )
    center, scale = processor._scaling["tumor_size"]
    assert center == pytest.approx(1039 / 3)
    vector = processor.process({"tumor_size": "999"})
    assert vector.tolist() == [pytest.approx((999 - center) / scale)]


def test_reduced_task_encodes_without_extent(seer_dataset: SEERDataset) -> None:
    split = SubjectSplit((("00001",), ("00002",)))
    task = SEERSurvivalPrediction(encoding="numeric", include_extent=False)
    train, test = seer_dataset.set_task(task, split=split)
    try:
        processor = train.input_processors["features"]
        assert set(processor.field_slices).isdisjoint(SEER_EXTENT_FEATURES)
        assert train[0]["features"].shape == (len(processor.field_slices),)
    finally:
        train.close()
        test.close()


def test_processor_reports_flat_vector_shape() -> None:
    processor = SEERFeatureProcessor(("stage",), "one_hot")
    processor.fit([{"x": {"stage": "I"}}], "x")
    assert processor.dim() == (1,)
    assert processor.spatial() == (False,)


@pytest.mark.parametrize("horizon", [0, -1, 12.5, True])
def test_requires_a_positive_whole_month_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SEERSurvivalPrediction(horizon)
