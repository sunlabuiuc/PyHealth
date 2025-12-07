"""Unit tests for OhioT1DMDataset and blood glucose prediction tasks.

Run with: pytest test_ohiot1dm.py -v

These tests create synthetic OhioT1DM data to verify:
1. Dataset loading works correctly
2. XML parsing is correct
3. Patient data structure is correct
4. Task functions create valid samples
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np


def create_synthetic_ohiot1dm_data(root_dir: str, num_subjects: int = 3) -> None:
    """Create synthetic OhioT1DM XML data for testing.
    
    Args:
        root_dir: Directory to create test data in.
        num_subjects: Number of subjects to create.
    """
    subject_ids = [540, 544, 552][:num_subjects]
    
    for sid in subject_ids:
        # Create training XML
        xml_content = create_xml_for_subject(sid, num_days=7)
        xml_path = os.path.join(root_dir, f"{sid}-ws-training.xml")
        with open(xml_path, "w") as f:
            f.write(xml_content)


def create_xml_for_subject(subject_id: int, num_days: int = 7) -> str:
    """Create synthetic XML content for a subject."""
    
    # Generate timestamps and glucose values
    start_date = datetime(2020, 1, 1, 0, 0, 0)
    
    glucose_events = []
    meal_events = []
    bolus_events = []
    basal_events = []
    
    # Generate 5-minute CGM readings
    num_readings = num_days * 24 * 12  # 12 readings per hour
    
    for i in range(num_readings):
        ts = start_date + timedelta(minutes=i * 5)
        ts_str = ts.strftime("%d-%m-%Y %H:%M:%S")
        
        # Simulate realistic glucose values (80-200 mg/dL with some variation)
        base_glucose = 120 + 30 * np.sin(i * 0.1) + np.random.normal(0, 10)
        base_glucose = np.clip(base_glucose, 50, 300)
        
        glucose_events.append(f'        <event ts="{ts_str}" value="{base_glucose:.1f}"/>')
    
    # Generate some meals (3 per day)
    for day in range(num_days):
        for hour in [8, 13, 19]:  # Breakfast, lunch, dinner
            ts = start_date + timedelta(days=day, hours=hour)
            ts_str = ts.strftime("%d-%m-%Y %H:%M:%S")
            carbs = np.random.randint(30, 80)
            meal_events.append(f'        <event ts="{ts_str}" type="meal" carbs="{carbs}"/>')
    
    # Generate bolus insulin (with meals)
    for day in range(num_days):
        for hour in [8, 13, 19]:
            ts = start_date + timedelta(days=day, hours=hour)
            ts_str = ts.strftime("%d-%m-%Y %H:%M:%S")
            dose = np.random.uniform(2, 8)
            bolus_events.append(f'        <event ts_begin="{ts_str}" ts_end="{ts_str}" dose="{dose:.1f}" type="normal"/>')
    
    # Generate basal rates
    basal_events.append(f'        <event ts="{start_date.strftime("%d-%m-%Y %H:%M:%S")}" value="0.8"/>')
    
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<patient id="{subject_id}" insulin_type="humalog" weight="99">
    <glucose_level>
{chr(10).join(glucose_events)}
    </glucose_level>
    <meal>
{chr(10).join(meal_events)}
    </meal>
    <bolus>
{chr(10).join(bolus_events)}
    </bolus>
    <basal>
{chr(10).join(basal_events)}
    </basal>
    <finger_stick>
    </finger_stick>
    <temp_basal>
    </temp_basal>
    <exercise>
    </exercise>
    <sleep>
    </sleep>
    <work>
    </work>
    <stressors>
    </stressors>
    <illness>
    </illness>
    <basis_heart_rate>
    </basis_heart_rate>
    <basis_gsr>
    </basis_gsr>
    <basis_skin_temperature>
    </basis_skin_temperature>
    <basis_air_temperature>
    </basis_air_temperature>
    <basis_steps>
    </basis_steps>
</patient>'''
    
    return xml_content


class TestOhioT1DMDataset(unittest.TestCase):
    """Test cases for OhioT1DMDataset."""

    @classmethod
    def setUpClass(cls):
        """Create test data directory."""
        cls.test_dir = tempfile.mkdtemp()
        create_synthetic_ohiot1dm_data(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.test_dir)

    def test_dataset_loading(self):
        """Test that dataset loads correctly."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        
        self.assertGreater(len(dataset), 0)
        self.assertIn("540", dataset.patients)

    def test_patient_data_structure(self):
        """Test patient data has correct structure."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        patient = dataset.get_patient("540")

        # Check keys exist
        self.assertIn("patient_id", patient)
        self.assertIn("glucose_level", patient)
        self.assertIn("meal", patient)
        self.assertIn("bolus", patient)
        self.assertIn("basal", patient)

        # Check glucose data
        self.assertGreater(len(patient["glucose_level"]), 0)
        self.assertIn("ts", patient["glucose_level"][0])
        self.assertIn("value", patient["glucose_level"][0])

    def test_glucose_values_range(self):
        """Test that glucose values are in reasonable range."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        patient = dataset.get_patient("540")
        
        glucose_values = [g["value"] for g in patient["glucose_level"]]
        
        self.assertTrue(all(40 <= v <= 400 for v in glucose_values))

    def test_dev_mode(self):
        """Test dev mode loads fewer subjects."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset

        dataset = OhioT1DMDataset(root=self.test_dir, dev=True, refresh_cache=True)
        self.assertLessEqual(len(dataset), 3)

    def test_get_patient_error(self):
        """Test get_patient raises error for invalid ID."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        
        with self.assertRaises(KeyError):
            dataset.get_patient("999")

    def test_iteration(self):
        """Test iteration over dataset."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        patients = list(dataset)
        
        self.assertGreater(len(patients), 0)


class TestBloodGlucosePredictionTask(unittest.TestCase):
    """Test cases for blood glucose prediction task functions."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        create_synthetic_ohiot1dm_data(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_glucose_prediction_task(self):
        """Test blood glucose prediction task."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset
        from pyhealth.tasks.blood_glucose_prediction_ohiot1dm import blood_glucose_prediction_30min_fn

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        dataset = dataset.set_task(blood_glucose_prediction_30min_fn)

        self.assertIsNotNone(dataset.samples)
        self.assertGreater(len(dataset.samples), 0)

        # Check sample structure
        sample = dataset.samples[0]
        self.assertIn("patient_id", sample)
        self.assertIn("record_id", sample)
        self.assertIn("glucose_history", sample)
        self.assertIn("glucose_target", sample)

        # Check glucose_history shape (60 min / 5 min = 12 points)
        self.assertEqual(sample["glucose_history"].shape, (12,))

    def test_hypoglycemia_detection_task(self):
        """Test hypoglycemia detection task."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset
        from pyhealth.tasks.blood_glucose_prediction_ohiot1dm import hypoglycemia_detection_ohiot1dm_fn

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        dataset = dataset.set_task(hypoglycemia_detection_ohiot1dm_fn)

        self.assertGreater(len(dataset.samples), 0)

        # Check label is binary
        sample = dataset.samples[0]
        self.assertIn(sample["label"], [0, 1])

    def test_glucose_range_classification_task(self):
        """Test glucose range classification task."""
        from pyhealth.datasets.ohiot1dm import OhioT1DMDataset
        from pyhealth.tasks.blood_glucose_prediction_ohiot1dm import glucose_range_classification_ohiot1dm_fn

        dataset = OhioT1DMDataset(root=self.test_dir, refresh_cache=True)
        dataset = dataset.set_task(glucose_range_classification_ohiot1dm_fn)

        self.assertGreater(len(dataset.samples), 0)

        # Check label is in valid range (0, 1, or 2)
        sample = dataset.samples[0]
        self.assertIn(sample["label"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
