from pyhealth.patient_builder.patient_builder import PatientBuilder


class MockCondition:
    def __init__(self, code):
        self.condition_source_value = code


class MockPatient:
    def __init__(self, patient_id, conditions, visits):
        self.patient_id = patient_id
        self._conditions = conditions
        self._visits = visits

    def get_events(self, event_type):

        if event_type == "condition_occurrence":
            return self._conditions

        if event_type == "visit_occurrence":
            return self._visits

        return []


class MockDataset:
    def __init__(self):

        patient1 = MockPatient(
            patient_id="1",
            conditions=[
                MockCondition("E11.9"),
                MockCondition("I10"),
            ],
            visits=[1, 2, 3],
        )

        patient2 = MockPatient(
            patient_id="2",
            conditions=[
                MockCondition("I10"),
            ],
            visits=[1],
        )

        patient3 = MockPatient(
            patient_id="3",
            conditions=[
                MockCondition("J44.9"),
            ],
            visits=[1, 2],
        )

        self.patients = {
            "1": patient1,
            "2": patient2,
            "3": patient3,
        }

    def get_patient(self, patient_id):
        return self.patients[patient_id]

    def get_visit_count(self, patient):
        return len(
            patient.get_events(
                event_type="visit_occurrence"
            )
        )

    def get_condition_code(self, event):
        return event.condition_source_value


def test_builder_creation():

    dataset = MockDataset()

    builder = PatientBuilder(dataset)

    assert builder is not None


def test_build_without_filters():

    dataset = MockDataset()

    patients = (
        PatientBuilder(dataset)
        .build()
    )

    assert len(patients) == 3


def test_with_min_visits():

    dataset = MockDataset()

    patients = (
        PatientBuilder(dataset)
        .with_min_visits(2)
        .build()
    )

    assert len(patients) == 2

    patient_ids = {
        p.patient_id
        for p in patients
    }

    assert patient_ids == {"1", "3"}


def test_with_condition():

    dataset = MockDataset()

    patients = (
        PatientBuilder(dataset)
        .with_condition("E11.9")
        .build()
    )

    assert len(patients) == 1

    assert patients[0].patient_id == "1"


def test_chained_filters():

    dataset = MockDataset()

    patients = (
        PatientBuilder(dataset)
        .with_condition("E11.9")
        .with_min_visits(2)
        .build()
    )

    assert len(patients) == 1

    assert patients[0].patient_id == "1"


def test_no_matching_condition():

    dataset = MockDataset()

    patients = (
        PatientBuilder(dataset)
        .with_condition("UNKNOWN")
        .build()
    )

    assert len(patients) == 0


def test_with_min_visits_no_match():

    dataset = MockDataset()

    patients = (
        PatientBuilder(dataset)
        .with_min_visits(10)
        .build()
    )

    assert len(patients) == 0