import unittest

from pyhealth.tasks import ReadmissionPredictionMIMIC3

class TestReadmissionPredictionMIMIC3(unittest.TestCase):
    def test_task_schema(self):
        self.assertIn("task_name", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("input_schema", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("output_schema", vars(ReadmissionPredictionMIMIC3))

if __name__ == "__main__":
    unittest.main()
