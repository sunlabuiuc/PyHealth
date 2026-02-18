import unittest
from datetime import datetime, timedelta

from pyhealth.tasks.ehr_foundational_model_mimic4 import EHRFoundationalModelMIMIC4


class TestEHRFoundationalModelMIMIC4(unittest.TestCase):
    """Tests for EHRFoundationalModelMIMIC4.

    Assumptions:
    - first_admission_time: We have a first admission time that comes from `patient.get_events(event_type="admissions")`
    
    """

    def setUp(self):
        self.task = EHRFoundationalModelMIMIC4()
        self.first_admission_time = datetime(2023, 1, 1, 0, 0, 0)

    # Tests for EHRFoundationalModelMIMIC4._compute_time_diffs
    def test_empty_list_returns_missing(self):
        texts, diffs = self.task._compute_time_diffs([], self.first_admission_time)
        self.assertEqual(texts, ["<missing>"])
        self.assertEqual(diffs, [0.0])

    def test_none_returns_missing(self):
        texts, diffs = self.task._compute_time_diffs(None, self.first_admission_time)
        self.assertEqual(texts, ["<missing>"])
        self.assertEqual(diffs, [0.0])

    def test_single_note_at_admission_time(self):
        notes = [("note A", self.first_admission_time)]
        texts, diffs = self.task._compute_time_diffs(notes, self.first_admission_time)
        self.assertEqual(texts, ["note A"])
        self.assertAlmostEqual(diffs[0], 0.0)

    def test_single_note_offset(self):
        notes = [("note A", self.first_admission_time + timedelta(hours=3))]
        texts, diffs = self.task._compute_time_diffs(notes, self.first_admission_time)
        self.assertEqual(texts, ["note A"])
        self.assertAlmostEqual(diffs[0], 3.0)

    def test_multiple_notes_unsorted(self):
        notes = [
            ("third", self.first_admission_time + timedelta(hours=10)),
            ("first", self.first_admission_time + timedelta(hours=1)),
            ("second", self.first_admission_time + timedelta(hours=5)),
        ]
        texts, diffs = self.task._compute_time_diffs(notes, self.first_admission_time)
        self.assertEqual(texts, ["first", "second", "third"])
        self.assertAlmostEqual(diffs[0], 1.0)
        self.assertAlmostEqual(diffs[1], 5.0)
        self.assertAlmostEqual(diffs[2], 10.0)


if __name__ == "__main__":
    unittest.main()
