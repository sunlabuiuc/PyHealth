import unittest

from pyhealth.datasets import (
    assert_patient_disjoint,
    check_patient_disjoint,
    create_sample_dataset,
    get_patient_ids,
    split_by_patient,
    split_by_patient_conformal,
    split_by_sample,
)


def _make_dataset(patient_counts):
    samples = []
    for patient_id, count in patient_counts:
        for index in range(count):
            samples.append(
                {
                    "patient_id": patient_id,
                    "record_id": f"{patient_id}-{index}",
                    "label": index % 2,
                }
            )
    return create_sample_dataset(
        samples=samples,
        input_schema={},
        output_schema={"label": "binary"},
        in_memory=True,
    )


class TestPatientDisjoint(unittest.TestCase):
    def test_split_by_patient_passes(self):
        dataset = _make_dataset([(f"p{i}", 2) for i in range(6)])
        train, val, test = split_by_patient(dataset, [0.5, 0.25, 0.25], seed=0)

        report = assert_patient_disjoint(
            train, val, test, names=["train", "val", "test"]
        )

        self.assertTrue(report["is_disjoint"])
        self.assertEqual(report["overlaps"], {})
        self.assertEqual(sum(report["counts"].values()), 6)

    def test_split_by_sample_can_fail_with_repeated_patients(self):
        dataset = _make_dataset([("shared", 6), ("other_a", 1), ("other_b", 1)])
        train, val, test = split_by_sample(dataset, [0.5, 0.25, 0.25], seed=0)

        report = check_patient_disjoint(
            train, val, test, names=["train", "val", "test"]
        )

        self.assertFalse(report["is_disjoint"])
        self.assertTrue(any("/" in key for key in report["overlaps"]))
        self.assertIn("shared", set().union(*report["overlaps"].values()))
        with self.assertRaisesRegex(
            AssertionError,
            "Patient overlap detected.*shared",
        ):
            assert_patient_disjoint(
                train, val, test, names=["train", "val", "test"]
            )

    def test_split_by_patient_conformal_passes(self):
        dataset = _make_dataset([(f"p{i}", 2) for i in range(8)])
        train, val, cal, test = split_by_patient_conformal(
            dataset, [0.25, 0.25, 0.25, 0.25], seed=0
        )

        report = assert_patient_disjoint(
            train, val, cal, test, names=["train", "val", "cal", "test"]
        )

        self.assertTrue(report["is_disjoint"])
        self.assertEqual(report["overlaps"], {})
        self.assertEqual(sum(report["counts"].values()), 8)

    def test_missing_patient_id_error_is_readable(self):
        with self.assertRaisesRegex(ValueError, "Sample 0.*patient_id"):
            get_patient_ids([{"record_id": "r0", "label": 0}])

        with self.assertRaisesRegex(ValueError, "Sample 1.*patient_id"):
            get_patient_ids(
                [
                    {"patient_id": "p0", "label": 0},
                    {"patient_id": None, "label": 1},
                ]
            )


if __name__ == "__main__":
    unittest.main()
