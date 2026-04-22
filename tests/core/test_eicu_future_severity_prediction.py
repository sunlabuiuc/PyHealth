import unittest
from types import SimpleNamespace

from pyhealth.tasks import FutureSeverityPredictionEICU


class MockPatient:
    """Minimal mock patient for task testing."""

    def __init__(self, patient_id, stays, events_by_type):
        self.patient_id = patient_id
        self._stays = stays
        self._events_by_type = events_by_type

    def get_events(self, event_type, filters=None):
        if event_type == "patient":
            return self._stays

        events = list(self._events_by_type.get(event_type, []))
        if not filters:
            return events

        filtered_events = events
        for attr, op, value in filters:
            if op != "==":
                raise ValueError(f"Unsupported operator in test mock: {op}")
            filtered_events = [
                event
                for event in filtered_events
                if str(getattr(event, attr, "")) == str(value)
            ]
        return filtered_events


class TestEICUFutureSeverityPrediction(unittest.TestCase):
    """Tests for the eICU future severity prediction task."""

    def test_task_metadata(self):
        """Task exposes the expected schemas and task name."""
        task = FutureSeverityPredictionEICU()

        self.assertEqual(task.task_name, "FutureSeverityPredictionEICU")
        self.assertEqual(
            task.input_schema,
            {
                "conditions": "sequence",
                "procedures": "sequence",
                "drugs": "sequence",
            },
        )
        self.assertEqual(task.output_schema, {"future_severity": "multiclass"})

    def test_invalid_future_window_raises_error(self):
        """Task should reject non-positive future windows."""
        with self.assertRaises(ValueError):
            FutureSeverityPredictionEICU(future_window=0)

    def test_returns_empty_when_not_enough_future_stays(self):
        """Task should return no samples without enough future visits."""
        stays = [SimpleNamespace(patientunitstayid="stay1")]
        patient = MockPatient(
            patient_id="patient_1",
            stays=stays,
            events_by_type={},
        )

        task = FutureSeverityPredictionEICU(future_window=1)
        samples = task(patient)

        self.assertEqual(samples, [])

    def test_skips_stay_with_missing_modalities(self):
        """Task skips a sample when any required modality is empty."""
        stays = [
            SimpleNamespace(patientunitstayid="stay1"),
            SimpleNamespace(patientunitstayid="stay2"),
        ]
        patient = MockPatient(
            patient_id="patient_2",
            stays=stays,
            events_by_type={
                "diagnosis": [
                    SimpleNamespace(patientunitstayid="stay1", icd9code="250.00"),
                    SimpleNamespace(patientunitstayid="stay2", icd9code="401.9"),
                ],
                "physicalexam": [
                    SimpleNamespace(
                        patientunitstayid="stay2",
                        physicalexampath="Neuro/Eyes",
                    )
                ],
                "medication": [
                    SimpleNamespace(patientunitstayid="stay1", drugname="Aspirin"),
                    SimpleNamespace(patientunitstayid="stay2", drugname="Insulin"),
                ],
            },
        )

        task = FutureSeverityPredictionEICU(future_window=1)
        samples = task(patient)

        self.assertEqual(samples, [])

    def test_generates_expected_sample_and_low_severity_label(self):
        """Task builds the expected sample structure and low-risk label."""
        stays = [
            SimpleNamespace(patientunitstayid="stay1"),
            SimpleNamespace(patientunitstayid="stay2"),
        ]
        patient = MockPatient(
            patient_id="patient_3",
            stays=stays,
            events_by_type={
                "diagnosis": [
                    SimpleNamespace(patientunitstayid="stay1", icd9code="250.00"),
                    SimpleNamespace(patientunitstayid="stay2", icd9code="401.9"),
                ],
                "physicalexam": [
                    SimpleNamespace(
                        patientunitstayid="stay1",
                        physicalexampath="CVS/Heart",
                    )
                ],
                "medication": [
                    SimpleNamespace(patientunitstayid="stay1", drugname="Metformin"),
                ],
            },
        )

        task = FutureSeverityPredictionEICU(future_window=1)
        samples = task(patient)

        self.assertEqual(len(samples), 1)

        sample = samples[0]
        self.assertEqual(sample["visit_id"], "stay1")
        self.assertEqual(sample["patient_id"], "patient_3")
        self.assertEqual(sample["conditions"], ["250.00"])
        self.assertEqual(sample["procedures"], ["CVS/Heart"])
        self.assertEqual(sample["drugs"], ["Metformin"])
        self.assertEqual(sample["future_severity"], 0)

    def test_generates_medium_and_high_severity_labels(self):
        """Task assigns medium- and high-severity labels correctly."""
        stays = [
            SimpleNamespace(patientunitstayid="stay1"),
            SimpleNamespace(patientunitstayid="stay2"),
            SimpleNamespace(patientunitstayid="stay3"),
        ]
        patient = MockPatient(
            patient_id="patient_4",
            stays=stays,
            events_by_type={
                "diagnosis": [
                    SimpleNamespace(patientunitstayid="stay1", icd9code="111.1"),
                    SimpleNamespace(patientunitstayid="stay2", icd9code="222.2"),
                    SimpleNamespace(patientunitstayid="stay2", icd9code="333.3"),
                    SimpleNamespace(patientunitstayid="stay3", icd9code="444.4"),
                    SimpleNamespace(patientunitstayid="stay3", icd9code="555.5"),
                    SimpleNamespace(patientunitstayid="stay3", icd9code="666.6"),
                    SimpleNamespace(patientunitstayid="stay3", icd9code="777.7"),
                ],
                "physicalexam": [
                    SimpleNamespace(
                        patientunitstayid="stay1",
                        physicalexampath="Exam/A",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay2",
                        physicalexampath="Exam/B",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay2",
                        physicalexampath="Exam/C",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay2",
                        physicalexampath="Exam/D",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay3",
                        physicalexampath="Exam/E",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay3",
                        physicalexampath="Exam/F",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay3",
                        physicalexampath="Exam/G",
                    ),
                ],
                "medication": [
                    SimpleNamespace(patientunitstayid="stay1", drugname="DrugA"),
                    SimpleNamespace(patientunitstayid="stay2", drugname="DrugB"),
                    SimpleNamespace(patientunitstayid="stay2", drugname="DrugC"),
                    SimpleNamespace(patientunitstayid="stay3", drugname="DrugD"),
                    SimpleNamespace(patientunitstayid="stay3", drugname="DrugE"),
                    SimpleNamespace(patientunitstayid="stay3", drugname="DrugF"),
                ],
            },
        )

        task = FutureSeverityPredictionEICU(future_window=1)
        samples = task(patient)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["visit_id"], "stay1")
        self.assertEqual(samples[0]["future_severity"], 1)
        self.assertEqual(samples[1]["visit_id"], "stay2")
        self.assertEqual(samples[1]["future_severity"], 2)

    def test_future_window_changes_lookahead(self):
        """Task uses the configured future window for label generation."""
        stays = [
            SimpleNamespace(patientunitstayid="stay1"),
            SimpleNamespace(patientunitstayid="stay2"),
            SimpleNamespace(patientunitstayid="stay3"),
        ]
        patient = MockPatient(
            patient_id="patient_5",
            stays=stays,
            events_by_type={
                "diagnosis": [
                    SimpleNamespace(patientunitstayid="stay1", icd9code="100.0"),
                    SimpleNamespace(patientunitstayid="stay2", icd9code="200.0"),
                    SimpleNamespace(patientunitstayid="stay3", icd9code="300.0"),
                ],
                "physicalexam": [
                    SimpleNamespace(
                        patientunitstayid="stay1",
                        physicalexampath="Exam/One",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay2",
                        physicalexampath="Exam/Two",
                    ),
                    SimpleNamespace(
                        patientunitstayid="stay3",
                        physicalexampath="Exam/Three",
                    ),
                ],
                "medication": [
                    SimpleNamespace(patientunitstayid="stay1", drugname="Drug1"),
                    SimpleNamespace(patientunitstayid="stay2", drugname="Drug2"),
                ],
            },
        )

        task = FutureSeverityPredictionEICU(future_window=2)
        samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["visit_id"], "stay1")
        self.assertEqual(samples[0]["future_severity"], 0)


if __name__ == "__main__":
    unittest.main()