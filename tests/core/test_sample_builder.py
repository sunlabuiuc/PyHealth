import unittest

from pyhealth.datasets.sample_dataset import SampleBuilder


class TestSampleBuilder(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"patient_id": "p1", "record_id": "r1", "feature": "a", "label": 1},
            {"patient_id": "p1", "record_id": "r2", "feature": "b", "label": 0},
            {"patient_id": "p2", "record_id": "r3", "feature": "c", "label": 1},
        ]
        self.input_schema = {"feature": "raw"}
        self.output_schema = {"label": "raw"}

    def test_fit_and_transform(self):
        builder = SampleBuilder(
            input_schema=self.input_schema, output_schema=self.output_schema
        )

        with self.assertRaises(RuntimeError):
            _ = builder.input_processors  # Access before fit should fail

        builder.fit(iter(self.samples))

        self.assertIn("feature", builder.input_processors)
        self.assertIn("label", builder.output_processors)

        self.assertEqual(builder.patient_to_index["p1"], [0, 1])
        self.assertEqual(builder.record_to_index["r3"], [2])

        transformed = builder.transform(self.samples[0])
        self.assertEqual(transformed["feature"], "a")
        self.assertEqual(transformed["label"], 1)
        self.assertEqual(transformed["patient_id"], "p1")

    def test_transform_requires_fit(self):
        builder = SampleBuilder(
            input_schema=self.input_schema, output_schema=self.output_schema
        )
        with self.assertRaises(RuntimeError):
            builder.transform(self.samples[0])


if __name__ == "__main__":
    unittest.main()
