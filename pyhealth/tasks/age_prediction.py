"""
Age prediction task for chest X-rays.
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask
from pyhealth.processors.base_processor import FeatureProcessor

logger = logging.getLogger(__name__)


class RegressionProcessor(FeatureProcessor):
    """Simple processor for continuous regression outputs."""

    def __init__(self):
        super().__init__()
        self.fitted = False

    def fit(self, samples, key):
        """No fitting needed for regression - just pass through."""
        self.fitted = True
        return self

    def process(self, sample):
        """Process method required by BaseProcessor - convert to float."""
        return float(sample)

    def __call__(self, sample):
        """Convert to float and return."""
        return self.process(sample)

    def inverse_transform(self, value):
        """Return the value as-is."""
        return float(value)

class AgePredictionTask(BaseTask):
    """
    A PyHealth task class for age prediction from chest X-rays.
    
    This task predicts patient age from chest X-ray images using regression.
    
    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.
        min_age (int): Minimum valid age.
        max_age (int): Maximum valid age.
    
    Examples:
        >>> from pyhealth.datasets import ChestXray14Dataset
        >>> from pyhealth.tasks import AgePredictionTask
        >>> 
        >>> dataset = ChestXray14Dataset(root="/data/chestxray14")
        >>> task = AgePredictionTask()
        >>> dataset = dataset.set_task(task)
    """
    
    task_name: str = "AgePrediction"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"age": "label"}
    
    def __init__(self, min_age: int = 1, max_age: int = 100) -> None:
        """
        Initializes the age prediction task.
        
        Args:
            min_age: Minimum valid age (default: 1)
            max_age: Maximum valid age (default: 100)
        """
        self.min_age = min_age
        self.max_age = max_age
    
    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generates age prediction samples for a single patient.
        
        Args:
            patient (Patient): A patient object containing chest X-ray events.
        
        Returns:
            List[Dict]: A list of dictionaries, each containing:
                - 'image': path to the chest X-ray image
                - 'age': patient age at the time of the image
                - 'patient_id': patient identifier
        """
        events: List[Event] = patient.get_events(event_type="chestxray14")
        
        samples = []
        for event in events:
            age = event.attr_dict.get("patient_age") or event.attr_dict.get("Patient Age")

            # Skip if no age
            if age is None:
                continue
            # Convert to float (handles both string and numeric types)
            try:
                age = float(age)
            except (ValueError, TypeError):
                logger.warning(f"Skipping sample with invalid age: {age}")
                continue
            # Skip if age out of valid range
            if age < self.min_age or age > self.max_age:
                logger.warning(f"Skipping sample with age {age} outside valid range [{self.min_age}, {self.max_age}]")
                continue
            
            # FIXED: Access path using attr_dict or direct attribute access
            image_path = event.attr_dict.get("path")
            if image_path is None:
                logger.warning(f"Skipping sample with no image path")
                continue
            
            sample = {
                "image": image_path,
                "age": age,
                "patient_id": patient.patient_id,
            }
            
            # Add optional fields if available - FIXED: Use attr_dict.get()
            gender = event.attr_dict.get("patient_sex") or event.attr_dict.get("Patient Sex")
            if gender is not None:
                sample["gender"] = gender
            
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def evaluate(
        y_true: torch.Tensor,
        y_prob: torch.Tensor,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate age prediction performance.
        
        Args:
            y_true: Ground truth ages (N,)
            y_prob: Predicted ages (N,)
            metrics: List of metrics to compute. If None, compute all.
                Available: mae, rmse, correlation, within_3, within_5, within_10
        
        Returns:
            Dictionary of metric names to values
        """
        if metrics is None:
            metrics = ["mae", "rmse", "correlation", "within_5", "within_10"]
        
        # Convert to numpy
        if torch.is_tensor(y_true):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = np.array(y_true)
        
        if torch.is_tensor(y_prob):
            y_prob_np = y_prob.cpu().numpy()
        else:
            y_prob_np = np.array(y_prob)
        
        # Clip predictions to reasonable range
        y_prob_np = np.clip(y_prob_np, 0, 120)
        
        results = {}
        
        if "mae" in metrics:
            results["mae"] = float(mean_absolute_error(y_true_np, y_prob_np))
        
        if "rmse" in metrics:
            results["rmse"] = float(np.sqrt(mean_squared_error(y_true_np, y_prob_np)))
        
        if "mse" in metrics:
            results["mse"] = float(mean_squared_error(y_true_np, y_prob_np))
        
        if "correlation" in metrics:
            if len(y_true_np) > 1:
                try:
                    corr, _ = pearsonr(y_true_np, y_prob_np)
                    results["correlation"] = float(corr)
                except Exception as e:
                    logger.warning(f"Could not compute correlation: {e}")
                    results["correlation"] = 0.0
        
        if "within_3" in metrics:
            results["within_3"] = float((np.abs(y_true_np - y_prob_np) <= 3).mean())
        
        if "within_5" in metrics:
            results["within_5"] = float((np.abs(y_true_np - y_prob_np) <= 5).mean())
        
        if "within_10" in metrics:
            results["within_10"] = float((np.abs(y_true_np - y_prob_np) <= 10).mean())
        
        return results