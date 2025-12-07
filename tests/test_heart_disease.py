import unittest
from types import SimpleNamespace

from pyhealth.tasks.heart_disease_prediction import HeartDiseasePrediction


class TestHeartDiseasePrediction(unittest.TestCase):
    def setUp(self):
        self.task = HeartDiseasePrediction()
        self.fake_patient_dict = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
            "target": 1,
        }

        self.fake_patient_obj = SimpleNamespace(**self.fake_patient_dict)

    def test_dict_input(self):
        samples = self.task(self.fake_patient_dict)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        # Check keys
        for feature in self.task.input_schema.keys():
            self.assertIn(feature, sample)

        # Check output label
        self.assertIn("heart_disease", sample)
        self.assertEqual(sample["heart_disease"], 1)

    def test_object_input(self):
        samples = self.task(self.fake_patient_obj)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        # Check keys
        for feature in self.task.input_schema.keys():
            self.assertIn(feature, sample)

        # Check output label
        self.assertIn("heart_disease", sample)
        self.assertEqual(sample["heart_disease"], 1)


if __name__ == "__main__":
    unittest.main()
