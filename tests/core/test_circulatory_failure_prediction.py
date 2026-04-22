from pyhealth.tasks import CirculatoryFailurePredictionTask


def test_circulatory_failure_task_basic():
    task = CirculatoryFailurePredictionTask(prediction_window_hours=12)

    patient = {
        "patient_id": 1,
        "icustay_id": 1001,
        "gender": "F",
        "first_failure_time": "2150-01-01 11:00:00",
        "time_series": [
            {"charttime": "2150-01-01 00:00:00", "map": 80.0},
            {"charttime": "2150-01-01 01:00:00", "map": 78.0},
            {"charttime": "2150-01-01 10:00:00", "map": 70.0},
            {"charttime": "2150-01-01 11:00:00", "map": 60.0},
        ],
    }

    samples = task(patient)

    assert len(samples) == 4
    assert samples[0]["label"] == 1
    assert samples[1]["label"] == 1
    assert samples[2]["label"] == 1
    assert samples[3]["label"] == 0
    assert samples[0]["features"]["map"] == 80.0
    assert samples[0]["gender"] == "F"