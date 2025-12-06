"""
Unit tests for age prediction functionality.

Author:
    Tyler Cheng (tylerc10@illinois.edu)
"""
import os
import shutil
import unittest

import numpy as np
import torch
from PIL import Image

from pyhealth.datasets import ChestXray14Dataset
from pyhealth.tasks.age_prediction import AgePredictionTask
from pyhealth.models.age_predictor import ChestXrayAgePredictor


class TestAgePrediction(unittest.TestCase):
    """Minimal tests for age prediction."""
    
    def setUp(self):
        """Create minimal mock dataset."""
        if os.path.exists("test_age"):
            shutil.rmtree("test_age")
        os.makedirs("test_age/images")
        
        # Minimal CSV with correct column names
        lines = [
            "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Sex,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],",
            "00000001_000.png,No Finding,0,1,25,M,PA,2682,2749,0.143,0.143,",
            "00000002_000.png,Effusion,0,2,50,F,PA,2500,2048,0.168,0.168,",
            "00000003_000.png,Hernia,0,3,75,M,PA,2500,2048,0.168,0.168,",
        ]
        
        # Create mock grayscale images
        for line in lines[1:]:
            name = line.split(',')[0]
            img = Image.fromarray(
                np.random.randint(0, 256, (224, 224), dtype=np.uint8), 
                mode="L"
            )
            img.save(os.path.join("test_age/images", name))
        
        with open("test_age/Data_Entry_2017_v2020.csv", 'w') as f:
            f.write("\n".join(lines))
        
        self.dataset = ChestXray14Dataset(root="./test_age")
        self.task = AgePredictionTask()
    
    def tearDown(self):
        """Clean up mock data."""
        if os.path.exists("test_age"):
            shutil.rmtree("test_age")
    
    def test_task(self):
        """Test age prediction task."""
        # Test task creation
        self.assertEqual(self.task.task_name, "AgePrediction")
        self.assertIn("image", self.task.input_schema)
        self.assertIn("age", self.task.output_schema)
        
        # Test task __call__ directly on a patient
        patient = self.dataset.get_patient('1')
        samples = self.task(patient)
        
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["age"], 25.0)
        self.assertIn("image", samples[0])
        
        # Test evaluation
        y_true = torch.tensor([25.0, 50.0, 75.0])
        y_prob = torch.tensor([27.0, 48.0, 73.0])
        metrics = self.task.evaluate(y_true, y_prob)
        
        self.assertIn("mae", metrics)
        self.assertLess(metrics["mae"], 5.0)
    
    def test_model(self):
        """Test age prediction model."""
        model = ChestXrayAgePredictor()
        
        # Test forward pass with labels
        batch_size = 2
        image = torch.randn(batch_size, 1, 224, 224)
        age = torch.tensor([30.0, 60.0])
        
        outputs = model(image=image, age=age)
        
        self.assertIn('y_prob', outputs)
        self.assertIn('loss', outputs)
        self.assertEqual(outputs['y_prob'].shape, (batch_size,))
        self.assertGreater(outputs['loss'].item(), 0)
    
    def test_integration(self):
        """Test task and model integration."""
        # Test task generates samples correctly
        patient = self.dataset.get_patient('2')
        samples = self.task(patient)
        
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["age"], 50.0)
        
        # Test model can process data
        model = ChestXrayAgePredictor()
        model.eval()
        
        # Simulate what a sample would look like
        # (without going through set_task which requires processors)
        with torch.no_grad():
            image = torch.randn(1, 1, 224, 224)
            age = torch.tensor([50.0])
            outputs = model(image=image, age=age)
        
        self.assertIn('y_prob', outputs)
        self.assertEqual(outputs['y_prob'].shape, (1,))


if __name__ == "__main__":
    unittest.main()