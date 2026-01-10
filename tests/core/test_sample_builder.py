import os
import pickle
import tempfile
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

        builder.fit(self.samples)

        self.assertIn("feature", builder.input_processors)
        self.assertIn("label", builder.output_processors)

        self.assertEqual(builder.patient_to_index["p1"], [0, 1])
        self.assertEqual(builder.record_to_index["r3"], [2])

        transformed = builder.transform(builder.metadata(), {"sample": pickle.dumps(self.samples[0])})
        self.assertEqual(transformed["feature"], "a")
        self.assertEqual(transformed["label"], 1)
        self.assertEqual(transformed["patient_id"], "p1")

    def test_transform_requires_fit(self):
        builder = SampleBuilder(
            input_schema=self.input_schema, output_schema=self.output_schema
        )
        with self.assertRaises(RuntimeError):
            builder.transform(builder.metadata(), {"sample": pickle.dumps(self.samples[0])})

    def test_index_mappings(self):
        builder = SampleBuilder(
            input_schema=self.input_schema, output_schema=self.output_schema
        )
        builder.fit(self.samples)

        expected_patient = {"p1": [0, 1], "p2": [2]}
        expected_record = {"r1": [0], "r2": [1], "r3": [2]}
        self.assertEqual(builder.patient_to_index, expected_patient)
        self.assertEqual(builder.record_to_index, expected_record)

    def test_save_persists_fitted_state(self):
        builder = SampleBuilder(
            input_schema=self.input_schema, output_schema=self.output_schema
        )
        builder.fit(self.samples)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "schema.pkl")
            builder.save(path)

            with open(path, "rb") as f:
                metadata = pickle.load(f)

        self.assertEqual(metadata["input_schema"], self.input_schema)
        self.assertEqual(metadata["output_schema"], self.output_schema)
        self.assertEqual(metadata["patient_to_index"], builder.patient_to_index)
        self.assertEqual(metadata["record_to_index"], builder.record_to_index)

        feature_processor = metadata["input_processors"]["feature"]
        label_processor = metadata["output_processors"]["label"]
        self.assertEqual(feature_processor.process("foo"), "foo")
        self.assertEqual(label_processor.process(1), 1)


if __name__ == "__main__":
    unittest.main()
