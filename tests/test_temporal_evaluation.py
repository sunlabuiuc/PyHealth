import pytest

from pyhealth.tasks.temporal_evaluation import (
    generate_synthetic_temporal_shift_data,
    prepare_data,
    run_ablation,
    run_random_experiment,
    run_temporal_experiment,
    temporal_split,
    validate_dataset,
)


@pytest.fixture
def synthetic_data():
    return [
        {"patient_id": 1, "year": 2010, "features": [0.20, 0.10], "label": 0},
        {"patient_id": 2, "year": 2011, "features": [0.25, 0.15], "label": 0},
        {"patient_id": 3, "year": 2012, "features": [0.70, 0.60], "label": 1},
        {"patient_id": 4, "year": 2013, "features": [0.40, 0.35], "label": 0},
        {"patient_id": 5, "year": 2014, "features": [0.55, 0.45], "label": 1},
        {"patient_id": 6, "year": 2015, "features": [0.60, 0.50], "label": 1},
        {"patient_id": 7, "year": 2016, "features": [0.65, 0.55], "label": 1},
        {"patient_id": 8, "year": 2017, "features": [0.35, 0.25], "label": 0},
        {"patient_id": 9, "year": 2018, "features": [0.75, 0.60], "label": 1},
        {"patient_id": 10, "year": 2019, "features": [0.15, 0.10], "label": 0},
        {"patient_id": 11, "year": 2020, "features": [0.82, 0.70], "label": 1},
        {"patient_id": 12, "year": 2021, "features": [0.18, 0.12], "label": 0},
    ]


def test_validate_dataset_passes(synthetic_data):
    validate_dataset(synthetic_data)


def test_validate_dataset_empty():
    with pytest.raises(ValueError, match="Dataset must not be empty"):
        validate_dataset([])


def test_validate_dataset_missing_key():
    bad_data = [
        {"patient_id": 1, "year": 2010, "label": 0},
    ]
    with pytest.raises(ValueError, match="missing required keys"):
        validate_dataset(bad_data)


def test_prepare_data(synthetic_data):
    x, y = prepare_data(synthetic_data[:2])
    assert x == [[0.20, 0.10], [0.25, 0.15]]
    assert y == [0, 0]


def test_temporal_split(synthetic_data):
    train, test = temporal_split(synthetic_data, 2015)
    assert len(train) == 6
    assert len(test) == 6
    assert all(row["year"] <= 2015 for row in train)
    assert all(row["year"] > 2015 for row in test)


def test_temporal_split_empty_train(synthetic_data):
    with pytest.raises(ValueError, match="empty training set"):
        temporal_split(synthetic_data, 2000)


def test_temporal_split_empty_test(synthetic_data):
    with pytest.raises(ValueError, match="empty testing set"):
        temporal_split(synthetic_data, 2025)


def test_run_temporal_experiment(synthetic_data):
    result = run_temporal_experiment(synthetic_data, 2015)
    assert result.experiment_type == "temporal"
    assert result.split_year == 2015
    assert result.train_size == 6
    assert result.test_size == 6
    assert 0.0 <= result.accuracy <= 1.0
    assert result.auroc is None or 0.0 <= result.auroc <= 1.0
    assert result.auprc is None or 0.0 <= result.auprc <= 1.0
    assert result.brier is None or 0.0 <= result.brier <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_run_random_experiment(synthetic_data):
    result = run_random_experiment(synthetic_data, random_state=42)
    assert result.experiment_type == "random"
    assert result.split_year is None
    assert result.train_size + result.test_size == len(synthetic_data)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.auroc is None or 0.0 <= result.auroc <= 1.0
    assert result.auprc is None or 0.0 <= result.auprc <= 1.0
    assert result.brier is None or 0.0 <= result.brier <= 1.0
    assert 0.0 <= result.f1 <= 1.0


def test_run_ablation(synthetic_data):
    results = run_ablation(synthetic_data, split_years=[2013, 2015, 2017])
    assert len(results) == 3
    assert [result.split_year for result in results] == [2013, 2015, 2017]


def test_generate_synthetic_temporal_shift_data():
    data = generate_synthetic_temporal_shift_data(
        n_patients_per_year=5,
        start_year=2010,
        end_year=2012,
        seed=123,
    )
    assert len(data) == 15
    assert all("year" in row and "features" in row and "label" in row for row in data)
    assert all(len(row["features"]) == 4 for row in data)
    years = sorted(set(row["year"] for row in data))
    assert years == [2010, 2011, 2012]