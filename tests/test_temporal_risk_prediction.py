from pyhealth.tasks.temporal_risk_prediction import TemporalMortalityMIMIC4


class MockTimestamp:
    def __init__(self, year):
        self.year = year


class MockEvent:
    def __init__(
        self,
        timestamp=None,
        icd_code=None,
        drug=None,
        hospital_expire_flag="0",
    ):
        self.timestamp = timestamp
        self.icd_code = icd_code
        self.drug = drug
        self.hospital_expire_flag = hospital_expire_flag


class MockPatient:
    """Mock patient object for testing temporal task behavior."""
    def __init__(self, event_map):
        self.event_map = event_map

    def get_events(self, table, end=None):
        return self.event_map.get(table, [])


def test_temporal_task_builds_samples():
    """Tests that the task generates valid samples with features, year, and label."""
    patient = MockPatient(
        {
            "admissions": [
                MockEvent(
                    timestamp=MockTimestamp(2018),
                    hospital_expire_flag="1",
                ),
                MockEvent(
                    timestamp=MockTimestamp(2020),
                    hospital_expire_flag="0",
                ),
            ],
            "diagnoses_icd": [
                MockEvent(timestamp=MockTimestamp(2017), icd_code="A"),
                MockEvent(timestamp=MockTimestamp(2018), icd_code="B"),
            ],
            "procedures_icd": [
                MockEvent(timestamp=MockTimestamp(2018), icd_code="P1"),
            ],
            "prescriptions": [
                MockEvent(timestamp=MockTimestamp(2018), drug="drug_a"),
            ],
        }
    )

    task = TemporalMortalityMIMIC4()
    samples = task(patient)

    assert len(samples) == 2
    assert all("features" in s and "year" in s and "label" in s for s in samples)
    assert all(len(s["features"]) == 4 for s in samples)
    assert samples[0]["year"] == [2018.0]
    assert samples[0]["label"] in [0, 1]


def test_temporal_task_skips_missing_timestamp():
    patient = MockPatient(
        {
            "admissions": [
                MockEvent(timestamp=None, hospital_expire_flag="0"),
            ],
            "diagnoses_icd": [],
            "procedures_icd": [],
            "prescriptions": [],
        }
    )

    task = TemporalMortalityMIMIC4()
    assert task(patient) == []


def test_temporal_task_respects_min_history_events():
    patient = MockPatient(
        {
            "admissions": [
                MockEvent(timestamp=MockTimestamp(2018), hospital_expire_flag="0"),
            ],
            "diagnoses_icd": [],
            "procedures_icd": [],
            "prescriptions": [],
        }
    )

    task = TemporalMortalityMIMIC4(min_history_events=1)
    assert task(patient) == []