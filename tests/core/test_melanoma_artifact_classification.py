import unittest
from typing import Any

from pyhealth.tasks import MelanomaArtifactClassification


class FakePatient:
    def __init__(self, patient_id: str, image: Any = None, label: Any = None):
        self.patient_id = patient_id
        self.image = image
        self.label = label


class TestMelanomaArtifactClassification(unittest.TestCase):

    def setUp(self):
        self.task = MelanomaArtifactClassification(mode="whole")

    def test_valid_patient(self):
        patient = FakePatient(
            patient_id="p1",
            image="fake_image_path.jpg",
            label=1
        )

        output = self.task(patient)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["patient_id"], "p1")
        self.assertEqual(output[0]["image"], "fake_image_path.jpg")
        self.assertEqual(output[0]["label"], 1)
        self.assertEqual(output[0]["mode"], "whole")

    def test_missing_image(self):
        patient = FakePatient(
            patient_id="p2",
            image=None,
            label=1
        )

        output = self.task(patient)
        self.assertEqual(output, [])

    def test_missing_label(self):
        patient = FakePatient(
            patient_id="p3",
            image="fake.jpg",
            label=None
        )

        output = self.task(patient)
        self.assertEqual(output, [])

    def test_missing_patient_id(self):
        patient = FakePatient(
            patient_id=None,
            image="fake.jpg",
            label=1
        )

        output = self.task(patient)
        self.assertEqual(output, [])

    def test_different_mode(self):
        task = MelanomaArtifactClassification(mode="background")

        patient = FakePatient(
            patient_id="p4",
            image="fake.jpg",
            label=0
        )

        output = task(patient)

        self.assertEqual(output[0]["mode"], "background")

    def test_mode_changes_image(self):
        import torch

        patient = FakePatient(
            patient_id="p5",
            image=torch.ones(3, 10, 10),
            label=1
        )

        task = MelanomaArtifactClassification(mode="background")
        output = task(patient)
        img = output[0]["image"]
        self.assertTrue((img[:, 3:7, 3:7] == 0).all())

if __name__ == "__main__":
    unittest.main()