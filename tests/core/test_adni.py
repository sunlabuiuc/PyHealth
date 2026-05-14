"""
Unit tests for the ADNIDataset and AlzheimersDiseaseClassification task.

Components tested:

    Dataset: ADNIDataset
    Task:    AlzheimersDiseaseClassification
    Model:   AlzheimersDiseaseCNN

"""
import nibabel as nib
import numpy as np
import random
import shutil
import tempfile
import torch
import unittest

from pathlib import Path
from pyhealth.datasets import ADNIDataset
from pyhealth.tasks import AlzheimersDiseaseClassification
from pyhealth.models import AlzheimersDiseaseCNN
from torch.utils.data import DataLoader


def create_adni_image(adni_root_path, subject_id=None, group=None):
    """Create test ADNI directory structure populated with synthetic data.

    Creates a directory structure for one subject, with the same layout as 
    the one obtained by downloading actual ADNI data files.

    The directory structure has the following layout:

    - root
        - subject id
            - pre-processing transform
                - date acquired
                    - image uid
                        MRI image file
        metadata xml file

    Args:
        adni_root_path: Path in which to create the ADNI directory structure
        subject_id: Subject ID for this directory structure, if None then a 
            random Subject ID will be generated instead.
        group: Label to assign to this subject, if None then a label will be 
            randomly selected from the three valid choices (i.e. MCI, CN, AD).

    Returns:
        Dictionary containing the following values for later comparison:
        - subject_id: Subject ID of the patient.
        - group: Label assigned to the patient.
        - gender: Patient's randomly selected gender.
        - age: Patient's randomly selected age.
        - weight: Patient's randomly selected weight.
        - image_uid: Unique ID of the MRI image.
        - image_path: Path to the MRI image file.
    """

    if not subject_id:
        subject_id = f"002_S_{random.randint(0, 9999):04d}"
    if not group:
        group = random.choice(["CN", "MCI", "AD"])
    gender = random.choice(["M", "F"])
    age = round(random.uniform(40.0000, 85.0000), 4)
    weight = round(random.uniform(55.0, 120.0), 1)
    date_acquired = f"{random.randint(1950, 2000)}-03-15"

    xform_str = "MPR__GradWarp__B1_Correction__N3"
    date_dir = f"{date_acquired}_09_45_30.0"
    series_id = f"{random.randint(0, 99999):05d}"
    image_uid = f"{random.randint(0, 99999):05d}"

    # Create MRI image directory structure
    adni_image_dir = adni_root_path / subject_id / \
        xform_str / date_dir / f"I{image_uid}"
    adni_image_dir.mkdir(parents=True)

    # Generate test MRI image
    file_date_str = f"{date_acquired.replace("-", "")}{random.randint(100000000, 300000000):9d}"
    image_filepath = adni_image_dir / \
        f"ADNI_{subject_id}_MR_{xform_str}_Br_{file_date_str}_S{series_id}_I{image_uid}.nii"
    image_data = np.random.rand(121, 145, 121).astype(np.float32)
    mri_image = nib.Nifti1Image(image_data, affine=np.eye(4))
    nib.save(mri_image, image_filepath)

    # Generate metadata xml
    metadata_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <idaxs>
        <project>
            <projectIdentifier>ADNI</projectIdentifier>
            <siteKey>002</siteKey>
            <subject>
                <subjectIdentifier>{subject_id}</subjectIdentifier>
                <researchGroup>{group}</researchGroup>
                <subjectSex>{gender}</subjectSex>
                <study>
                    <subjectAge>{age}</subjectAge>
                    <weightKg>{weight}</weightKg>
                    <series>
                        <seriesIdentifier>{series_id}</seriesIdentifier>
                        <dateAcquired>{date_acquired}</dateAcquired>
                        <seriesLevelMeta>
                            <derivedProduct>
                                <imageUID>{image_uid}</imageUID>
                            </derivedProduct>
                        </seriesLevelMeta>
                    </series>
                </study>
            </subject>
        </project>
    </idaxs>
    """
    metadata_xml_filename = f"ADNI_{subject_id}_{xform_str}_S{series_id}_I{image_uid}.xml"
    metadata_xml_path = adni_root_path / metadata_xml_filename
    with open(metadata_xml_path, "w", encoding="utf-8") as f:
        f.write(metadata_xml)

    # Return test values for later comparison
    return {
        "subject_id": subject_id,
        "group": group,
        "gender": gender,
        "age": age,
        "weight": weight,
        "image_uid": image_uid,
        "image_path": image_filepath,
    }


class TestADNIDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create test ADNI directory with test images and metadata files.
        Note that we're creating this once to be shared amongst all tests.
        """
        # Create temporary directory for simulated ADNI files
        cls.temp_directory = tempfile.mkdtemp()
        cls.adni_root = Path(cls.temp_directory)

        # Create three MRI images and their metadata files
        cls.test_vals = []
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0001", "CN"))
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0010", "MCI"))
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0100", "AD"))

        # Create dataset
        cls.dataset = ADNIDataset(root=cls.temp_directory)

        # Get the first record
        patient_id = cls.test_vals[0]["subject_id"]
        cls.first_patient = cls.dataset.get_patient(patient_id)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory"""
        shutil.rmtree(cls.temp_directory)

    def test_dataset_initialization(self):
        """Dataset can be initialized"""
        self.assertIsNotNone(self.dataset)

    def test_patient_structure(self):
        """Patient object structure is valid"""
        self.assertIsNotNone(self.first_patient)

        patient_events = self.first_patient.get_events(event_type="adni")
        self.assertIsNotNone(patient_events)
        first_event = patient_events[0]

        self.assertEqual(first_event["gender"], self.test_vals[0]["gender"])
        self.assertEqual(first_event["age"], str(self.test_vals[0]["age"]))

    def test_raises_exception_for_invalid_roots(self):
        """Raises FileNotFoundError is root path is invalid"""
        with self.assertRaises(FileNotFoundError):
            ADNIDataset(root="/some/bogus/path")

    def test_raises_exception_when_no_images(self):
        """Raises FileNotFoundError when images are missing"""
        empty_directory = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                ADNIDataset(root=empty_directory)
        finally:
            shutil.rmtree(empty_directory)

    def test_patient_count(self):
        """Correct number of patients is created"""
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_patient_attributes_exist(self):
        """Correct patient attributes are created"""
        patient_events = self.first_patient.get_events(event_type="adni")
        self.assertIn("gender", patient_events[0].attr_dict)
        self.assertIn("age", patient_events[0].attr_dict)
        self.assertIn("weight", patient_events[0].attr_dict)
        self.assertIn("group", patient_events[0].attr_dict)
        self.assertIn("image_uid", patient_events[0].attr_dict)
        self.assertIn("image_path", patient_events[0].attr_dict)

    def test_patient_attribute_values(self):
        """Patient attributes have expected values"""
        patient_events = self.first_patient.get_events(event_type="adni")
        self.assertEqual(patient_events[0].attr_dict["gender"], str(
            self.test_vals[0]["gender"]))
        self.assertEqual(patient_events[0].attr_dict["age"], str(
            self.test_vals[0]["age"]))
        self.assertEqual(patient_events[0].attr_dict["weight"], str(
            self.test_vals[0]["weight"]))
        self.assertEqual(patient_events[0].attr_dict["group"], str(
            self.test_vals[0]["group"]))
        self.assertEqual(patient_events[0].attr_dict["image_path"], str(
            self.test_vals[0]["image_path"]))

class TestAlzheimersDiseaseClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create test ADNI directory with test images and metadata files.
        Note that we're creating this once to be shared amongst all tests.
        """
        # Create temporary directory for simulated ADNI files
        cls.temp_directory = tempfile.mkdtemp()
        cls.adni_root = Path(cls.temp_directory)

        # Create three MRI images and their metadata files, one for each group
        cls.test_vals = []
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0001", "CN"))
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0010", "MCI"))
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0100", "AD"))

        # Create dataset
        cls.dataset = ADNIDataset(root=cls.temp_directory)

        # Assign task
        cls.task = AlzheimersDiseaseClassification()
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory"""
        shutil.rmtree(cls.temp_directory)

    def test_attributes_exist(self):
        """Correct attributes are present"""
        self.assertIn("image", self.samples[0])
        self.assertIn("label", self.samples[0])
        self.assertIn("age", self.samples[0])
        self.assertIn("gender", self.samples[0])
        self.assertIn("weight", self.samples[0])

    def test_attribute_types(self):
        """Attributes have correct types"""
        sample = self.samples[0]
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertIsInstance(sample["label"], (int, torch.Tensor))
        self.assertIsInstance(sample["age"], (float, torch.Tensor))
        self.assertIsInstance(sample["gender"], (int, torch.Tensor))
        self.assertIsInstance(sample["weight"], (float, torch.Tensor))

    def test_attribute_values(self):
        """Attributes have expected values"""
        GROUP_MAPPINGS = {"CN": 0, "MCI": 1, "AD": 2}
        GENDER_MAPPINGS = {"M": 0, "F": 1}
        for i in range(len(self.dataset.unique_patient_ids)):
            patient = self.dataset.get_patient(self.test_vals[i]["subject_id"])
            patient_events = patient.get_events(event_type="adni")
            self.assertEqual(
                self.samples[i]["label"].item(), GROUP_MAPPINGS[patient_events[0].attr_dict["group"]])
            self.assertAlmostEqual(
                self.samples[i]["age"].item(), float(patient_events[0].attr_dict["age"]), places=4)
            self.assertEqual(
                int(self.samples[i]["gender"].item()), GENDER_MAPPINGS[patient_events[0].attr_dict["gender"]])
            self.assertAlmostEqual(
                self.samples[i]["weight"], float(str(patient_events[0].attr_dict["weight"])), places=4)

    def test_image_shape(self):
        """Output image has correct shape"""
        self.assertEqual(self.samples[0]["image"].shape, (1, 96, 96, 96))

    def test_image_normalization(self):
        """Output image has been correctly normalized"""
        image = self.samples[0]["image"]
        mean_val = image.mean().item()
        std_val = image.std().item()
        self.assertAlmostEqual(mean_val, 0.0, places=3)
        self.assertAlmostEqual(std_val, 1.0, places=3)


class TestAlzheimersDiseaseModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create test ADNI directory with test images and metadata files.
        Note that we're creating this once to be shared amongst all tests.
        """
        # Create temporary directory for simulated ADNI files
        cls.temp_directory = tempfile.mkdtemp()
        cls.adni_root = Path(cls.temp_directory)

        # Create three MRI images and their metadata files, one for each group
        cls.test_vals = []
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0001", "CN"))
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0010", "MCI"))
        cls.test_vals.append(create_adni_image(
            cls.adni_root, "002_S_0100", "AD"))

        # Create dataset
        cls.dataset = ADNIDataset(root=cls.temp_directory)

        # Assign task
        cls.task = AlzheimersDiseaseClassification()
        cls.samples = cls.dataset.set_task(cls.task)

        # Initialize model
        cls.model = AlzheimersDiseaseCNN(
            dataset=cls.samples,
            width_factor=4,
            use_age=True,
            use_gender=True,
            norm_method="instance"
        )

        # Initialize dataloader
        cls.loader = DataLoader(cls.samples, batch_size=1)
        cls.batch = next(iter(cls.loader))

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory"""
        shutil.rmtree(cls.temp_directory)

    def test_forward_pass(self):
        """Model can process a batch"""
        # Execute forward pass
        output = self.model(**self.batch)
        self.assertIn("y_prob", output)
        self.assertEqual(output["y_prob"].shape, (1, 3))
        self.assertIn("loss", output)

    def test_backward_pass(self):
        """Model can compute gradients"""
        # Execute forward pass
        output = self.model(**self.batch)
        loss = output["loss"]

        # Execute backward pass
        loss.backward()

        # Check model gradients
        self.assertIsNotNone(self.model.block1[0].weight.grad)
        self.assertIsNotNone(self.model.block2[0].weight.grad)
        self.assertIsNotNone(self.model.block3[0].weight.grad)
        self.assertIsNotNone(self.model.block4[0].weight.grad)
        self.assertIsNotNone(self.model.fc1.weight.grad)
        self.assertIsNotNone(self.model.fc2.weight.grad)
        self.assertIsNotNone(self.model.age_fc1.weight.grad)
        self.assertIsNotNone(self.model.age_norm.weight.grad)
        self.assertIsNotNone(self.model.age_fc2.weight.grad)
        self.assertIsNotNone(self.model.gender_embed.weight.grad)

if __name__ == '__main__':
    unittest.main()
