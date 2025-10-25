# -*- coding: utf-8 -*-
"""Daily and Sports Activities Dataset loader for PyHealth.

Authors: 
- Michael Quan (mdquan2)
- Daniel Valentine (dvt3)

Reference:
- "Daily Physical Activity Monitoring: Adaptive Learning from Multi-source Motion Sensor Data"
- Paper Link: https://arxiv.org/abs/2405.16395
- Original GitHub: https://github.com/Oceanjinghai/HealthTimeSerial/tree/main

Description: 
Implements loading and preprocessing for the Daily and Sports Activities dataset from UCI Machine Learning Repository.
Contains 19 activities performed by 8 subjects, recorded with 45 sensors across 5 body positions.
"""

import os
import numpy as np
from pyhealth.datasets import BaseDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Dict, Optional, List, Tuple

class DailySportsActivitiesDataset(BaseDataset):
    """Daily and Sports Activities Dataset loader for time series classification.
    
    The dataset contains 19 activities collected from 8 subjects, each performed for 5 minutes.
    Five Xsens MTx units with 9 sensors each (total 45 sensors) are placed on torso, arms, and legs.
    
    Args:
        dataset_name: Name for the dataset instance. Default: "DailySports".
        root: Path to directory containing raw data files. Required.
        dev: Whether to enable developer mode (use small subset). Default: False.
        refresh_cache: Whether to refresh cache. Default: False.
        
    Examples:
        >>> from pyhealth.datasets import DailySportsActivitiesDataset
        >>> dataset = DailySportsActivitiesDataset(
        >>>     root="data/Daily_and_Sports_Activity/data/",
        >>>     dev=True
        >>> )
        >>> dataset.stat()
    """
    
    def parse_tables(self) -> Dict[str, Dict]:
        """Parse raw data files into PyHealth's patient-visit structure.
        
        Returns:
            A dictionary of patients with visit records containing:
                - signal: np.array of shape (125 timesteps, 45 sensors)
                - label: string activity name
                - metadata: tuple (activity, subject, session)
                
        Process:
            1. Load and scale raw sensor segments
            2. Organize by subject (patient) and session (visit)
            3. Add metadata for each sample
        """
        segments, labels, metadata = self._load_raw_data(self.root)
        patients = {}
        
        for segment, label, meta in zip(segments, labels, metadata):
            patient_id = meta[1]  # Subject ID as patient_id
            visit_id = f"{meta[0]}_{meta[2].split('.')[0]}"  # Activity_session
            
            if patient_id not in patients:
                patients[patient_id] = {
                    "patient_id": patient_id,
                    "visits": {},
                    "other": {}
                }
                
            patients[patient_id]["visits"][visit_id] = {
                "signal": segment.T,  # (125, 45)
                "label": label,
                "metadata": meta
            }
            
        return patients

    def _load_raw_data(self, data_root: str) -> Tuple[List[np.ndarray], List[str], List[tuple]]:
        """Load and preprocess raw sensor data from text files.
        
        Args:
            data_root: Root directory containing activity/subject folders
            
        Returns:
            Tuple of:
                - List of sensor segments (125x45 arrays)
                - List of activity labels
                - List of metadata tuples (activity, subject, session)
        """
        segments = []
        labels = []
        metadata = []
        
        for activity in sorted(os.listdir(data_root)):
            activity_path = os.path.join(data_root, activity)
            if not os.path.isdir(activity_path):
                continue
                
            for subject in sorted(os.listdir(activity_path)):
                subject_path = os.path.join(activity_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                
                for session in sorted(os.listdir(subject_path)):
                    if session.endswith('.txt'):
                        file_path = os.path.join(subject_path, session)
                        try:
                            raw = np.loadtxt(file_path, delimiter=",")
                            scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(raw)
                            segments.append(scaled)
                            labels.append(activity)
                            metadata.append((activity, subject, session))
                        except Exception as e:
                            raise RuntimeError(f"Error loading {file_path}: {str(e)}")
                            
        return segments, labels, metadata

    def set_task(self, task_fn: Optional[callable] = None, **kwargs):
        """Create task-specific dataset for activity recognition.
        
        Args:
            task_fn: Function to process patient data into samples.
                     Uses default activity recognition if None.
            kwargs: Additional arguments for task function
            
        Returns:
            TaskDataset with samples containing:
                - signal: np.array of shape (125, 45)
                - label: int encoded activity class
        """
        if task_fn is None:
            def default_task_fn(patient: Dict):
                samples = []
                label_encoder = LabelEncoder()
                label_encoder.fit(list(self.label_dict.values()))
                
                for visit_id, visit in patient["visits"].items():
                    samples.append({
                        "signal": visit["signal"],
                        "label": label_encoder.transform([visit["label"]])[0],
                        "metadata": visit["metadata"]
                    })
                return samples
            task_fn = default_task_fn
            
        return super().set_task(task_fn=task_fn, **kwargs)
