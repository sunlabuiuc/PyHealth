import json
from califorest_tests.utils import create_temp_dataset

"""
Dataset tests using small synthetic EHR data.

These tests verify:
- JSON dataset loading
- Patient and visit structure
- Event field integrity

All tests use temporary directories and synthetic data to ensure
fast execution and full isolation.
"""

def test_dataset_loading():

    #Verify that synthetic dataset JSON can be loaded correctly.
    temp_dir, data_path = create_temp_dataset()

    with open(data_path) as f:
        data = json.load(f)

    assert len(data) > 0  # patients exist
    assert isinstance(data, list)  # (data integrity)

    temp_dir.cleanup()


def test_patient_structure():

    #Verify each patient contains required fields.#
    temp_dir, data_path = create_temp_dataset()

    with open(data_path) as f:
        patients = json.load(f)

    patient = patients[0]
    assert "patient_id" in patient
    assert "visits" in patient
    assert len(patient["visits"]) == 2

    temp_dir.cleanup()


def test_visit_structure():
    #Verify each visit contains required event fields.
    temp_dir, data_path = create_temp_dataset()

    with open(data_path) as f:
        patients = json.load(f)

    visit = patients[0]["visits"][0]

    assert "conditions" in visit
    assert "procedures" in visit
    assert "drugs" in visit
    assert "label" in visit

    temp_dir.cleanup()