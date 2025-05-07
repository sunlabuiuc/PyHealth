from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from ..data import Patient, Visit


class BioimpedanceDataset(Dataset):
    """Bioimpedance Dataset class for handling bioimpedance signals and blood pressure data.
    
    This dataset is designed to work with wrist-worn bioimpedance measurements and 
    corresponding blood pressure values (systolic and diastolic). It supports both
    real data loading and synthetic data generation.
    
    Attributes:
        root (str): Root directory of the dataset.
        subjects (List[str], optional): List of subject IDs to include. If None, all subjects are included.
        signal_length (int, optional): Length of bioimpedance signal segments. Defaults to 1000.
        overlap (float, optional): Overlap between consecutive signal segments (0.0-1.0). Defaults to 0.5.
        normalize (bool, optional): Whether to normalize bioimpedance signals. Defaults to True.
        synthetic (bool, optional): Whether to use synthetic data. Defaults to False.
        num_synthetic_subjects (int, optional): Number of synthetic subjects to generate. Defaults to 10.
        sampling_rate (int, optional): Sampling rate of bioimpedance signals in Hz. Defaults to 100.
    """
    
    def __init__(
        self,
        root,
        dataset_name=None,
        subjects=None,
        signal_length=1000,
        overlap=0.5,
        normalize=True,
        synthetic=False,
        num_synthetic_subjects=10,
        sampling_rate=100,
        dev=False,
        refresh_cache=False,
        **kwargs
    ):
        """Initialize the BioimpedanceDataset.
        
        Args:
            root: Root directory of the dataset.
            dataset_name: Name of the dataset. Defaults to "bioimpedance".
            subjects: List of subject IDs to include. If None, all subjects are included.
            signal_length: Length of bioimpedance signal segments. Defaults to 1000.
            overlap: Overlap between consecutive signal segments (0.0-1.0). Defaults to 0.5.
            normalize: Whether to normalize bioimpedance signals. Defaults to True.
            synthetic: Whether to use synthetic data. Defaults to False.
            num_synthetic_subjects: Number of synthetic subjects to generate. Defaults to 10.
            sampling_rate: Sampling rate of bioimpedance signals in Hz. Defaults to 100.
            dev: Whether to enable dev mode (only use a small subset of the data). Defaults to False.
            refresh_cache: Whether to refresh the cache. Defaults to False.
        """
        # Dataset attributes
        self.task = None
        self.samples = []
        self.patient_to_index = {}
        self.record_to_index = {}
        
        # Bioimpedance-specific attributes
        self.root = root
        self.subjects = subjects
        self.signal_length = signal_length
        self.overlap = overlap
        self.normalize = normalize
        self.synthetic = synthetic
        self.num_synthetic_subjects = num_synthetic_subjects
        self.sampling_rate = sampling_rate
        
        # Store other attributes
        self.dataset_name = dataset_name or "bioimpedance"
        self.dev = dev
        self.refresh_cache = refresh_cache
        
        # Load data
        self.patients, self.visits = self._load_data()
    
    def _load_data(self) -> Tuple[Dict[str, Patient], Dict[str, Visit]]:
        """Load bioimpedance data and create Patient and Visit objects.
        
        This method either loads real bioimpedance data from files or generates
        synthetic data if synthetic=True.
        
        Returns:
            Tuple[Dict[str, Patient], Dict[str, Visit]]: A tuple containing:
                - patients: Dictionary mapping patient_id to Patient objects
                - visits: Dictionary mapping visit_id to Visit objects
        """
        print("Loading bioimpedance data...")
        
        # Initialize dictionaries for patients and visits
        patients = {}
        visits = {}
        
        if self.synthetic:
            # Generate synthetic data
            patient_data = self._generate_synthetic_data()
        else:
            # Load real data from files
            patient_data = self._load_real_data()
        
        # Process patient data and create Patient and Visit objects
        for subject_id, data in patient_data.items():
            # Create Patient object
            if subject_id not in patients:
                patient = Patient(
                    patient_id=subject_id,
                    gender=data.get("gender", "unknown"),
                    # Include other patient attributes if available
                )
                patients[subject_id] = patient
            
            # Create Visit objects for each recording session
            for session_idx, session_data in enumerate(data["sessions"]):
                visit_id = f"{subject_id}_session_{session_idx}"
                
                # Create Visit object
                visit = Visit(
                    visit_id=visit_id,
                    patient_id=subject_id,
                    encounter_time=None,  # Set to actual timestamp if available
                    discharge_time=None,  # Set to actual timestamp if available
                )
                
                # Add bioimpedance measurements as "procedure"
                for segment_idx, (bio_segment, sbp, dbp) in enumerate(zip(
                    session_data["bioimpedance_segments"],
                    session_data["sbp_values"],
                    session_data["dbp_values"]
                )):
                    # Store bioimpedance segment as a procedure
                    visit.add_procedure(
                        code=f"bioimp_segment_{segment_idx}",
                        code_system="bioimpedance",
                        timestamp=None,  # Set to actual timestamp if available
                        attrs={
                            "bioimpedance": bio_segment,
                            "sbp": sbp,
                            "dbp": dbp,
                            "segment_idx": segment_idx,
                        }
                    )
                
                # Add the visit to the patient and visits dictionary
                patients[subject_id].add_visit(visit)
                visits[visit_id] = visit
        
        print(f"Loaded data for {len(patients)} patients with {len(visits)} visits.")
        return patients, visits
    
    def _load_real_data(self) -> Dict:
        """Load real bioimpedance data from files.
        
        Returns:
            Dict: Dictionary containing patient data organized by subject ID.
        """
        patient_data = {}
        
        # Get list of all subject directories
        subject_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        
        # Filter subjects if self.subjects is provided
        if self.subjects is not None:
            subject_dirs = [d for d in subject_dirs if d in self.subjects]
        
        # Process each subject's data
        for subject_dir in tqdm(subject_dirs, desc="Loading subject data"):
            subject_id = subject_dir
            subject_path = os.path.join(self.root, subject_dir)
            
            # Initialize subject data structure
            patient_data[subject_id] = {
                "sessions": []
            }
            
            # Find all recording sessions for this subject
            session_files = [f for f in os.listdir(subject_path) if f.endswith(".csv") or f.endswith(".txt")]
            
            for session_file in session_files:
                session_path = os.path.join(subject_path, session_file)
                
                try:
                    # Load session data (adapt this to the actual file format)
                    session_data = pd.read_csv(session_path)
                    
                    # Extract bioimpedance signals and blood pressure values
                    bioimpedance_signal = session_data["bioimpedance"].values
                    sbp_values = session_data["sbp"].values
                    dbp_values = session_data["dbp"].values
                    
                    # Segment the signals with overlap
                    bioimpedance_segments, sbp_segments, dbp_segments = self._segment_signals(
                        bioimpedance_signal, sbp_values, dbp_values
                    )
                    
                    # Store segments and corresponding BP values
                    patient_data[subject_id]["sessions"].append({
                        "bioimpedance_segments": bioimpedance_segments,
                        "sbp_values": sbp_segments,
                        "dbp_values": dbp_segments,
                    })
                    
                except Exception as e:
                    print(f"Error loading session file {session_file}: {e}")
        
        return patient_data
    
    def _segment_signals(
        self, 
        bioimpedance_signal, 
        sbp_values, 
        dbp_values
    ):
        """Segment bioimpedance signals and corresponding BP values.
        
        Args:
            bioimpedance_signal: Raw bioimpedance signal.
            sbp_values: Systolic blood pressure values.
            dbp_values: Diastolic blood pressure values.
            
        Returns:
            Tuple[List[np.ndarray], List[float], List[float]]: Segmented signals and BP values.
        """
        # Calculate step size based on overlap
        step_size = int(self.signal_length * (1 - self.overlap))
        
        # Initialize lists for segments
        bioimpedance_segments = []
        sbp_segments = []
        dbp_segments = []
        
        # Generate segments
        for i in range(0, len(bioimpedance_signal) - self.signal_length + 1, step_size):
            segment = bioimpedance_signal[i:i + self.signal_length]
            
            # Normalize if required
            if self.normalize:
                segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-10)
            
            # Calculate corresponding BP values (average over segment)
            start_idx = i // len(sbp_values) if len(sbp_values) > 0 else 0
            end_idx = (i + self.signal_length) // len(sbp_values) if len(sbp_values) > 0 else 0
            end_idx = max(end_idx, start_idx + 1)
            
            sbp_segment = np.mean(sbp_values[start_idx:end_idx])
            dbp_segment = np.mean(dbp_values[start_idx:end_idx])
            
            # Add to lists
            bioimpedance_segments.append(segment)
            sbp_segments.append(sbp_segment)
            dbp_segments.append(dbp_segment)
        
        return bioimpedance_segments, sbp_segments, dbp_segments
    
    def _generate_synthetic_data(self) -> Dict:
        """Generate synthetic bioimpedance and blood pressure data.
        
        Returns:
            Dict: Dictionary containing synthetic patient data organized by subject ID.
        """
        print("Generating synthetic bioimpedance data...")
        patient_data = {}
        
        # Define number of subjects
        num_subjects = self.num_synthetic_subjects
        
        for subject_idx in range(num_subjects):
            subject_id = f"synthetic_subject_{subject_idx}"
            
            # Initialize subject data
            patient_data[subject_id] = {
                "gender": "male" if np.random.rand() > 0.5 else "female",
                "sessions": []
            }
            
            # Generate between 1-5 sessions per subject
            num_sessions = np.random.randint(1, 6)
            
            for session_idx in range(num_sessions):
                # Generate synthetic bioimpedance signal (e.g., sinusoidal with noise)
                session_length = np.random.randint(60, 301) * self.sampling_rate  # 1-5 minutes at 100Hz
                
                # Base frequency components
                t = np.arange(session_length) / self.sampling_rate
                base_freq = np.random.uniform(0.8, 1.2)  # Base heart rate frequency around 1 Hz
                amplitude = np.random.uniform(0.8, 1.2)
                
                # Generate basic bioimpedance waveform (combination of sinusoids + noise)
                bioimpedance_signal = amplitude * np.sin(2 * np.pi * base_freq * t)
                bioimpedance_signal += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)  # First harmonic
                bioimpedance_signal += 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)  # Second harmonic
                bioimpedance_signal += 0.05 * np.random.randn(len(t))  # Noise
                
                # Randomly vary the amplitude over time to simulate blood volume changes
                amplitude_mod = 1 + 0.2 * np.sin(2 * np.pi * 0.05 * t) + 0.05 * np.random.randn(len(t))
                bioimpedance_signal *= amplitude_mod
                
                # Generate corresponding BP values with fewer samples
                # (typically BP is measured less frequently than bioimpedance)
                num_bp_samples = max(10, session_length // (self.sampling_rate * 10))  # One sample every 10 seconds
                
                # Generate base BP values for this subject
                base_sbp = np.random.randint(100, 160)
                base_dbp = np.random.randint(60, 100)
                
                # Generate BP with some random fluctuation
                sbp_values = np.array([base_sbp + np.random.randint(-10, 11) for _ in range(num_bp_samples)])
                dbp_values = np.array([base_dbp + np.random.randint(-10, 11) for _ in range(num_bp_samples)])
                
                # Ensure systolic is always higher than diastolic
                for i in range(len(sbp_values)):
                    if sbp_values[i] <= dbp_values[i]:
                        sbp_values[i] = dbp_values[i] + np.random.randint(10, 41)
                
                # Segment the signals
                bioimpedance_segments, sbp_segments, dbp_segments = self._segment_signals(
                    bioimpedance_signal, sbp_values, dbp_values
                )
                
                # Store segments and corresponding BP values
                patient_data[subject_id]["sessions"].append({
                    "bioimpedance_segments": bioimpedance_segments,
                    "sbp_values": sbp_segments,
                    "dbp_values": dbp_segments,
                })
        
        return patient_data
    
    def set_task(
        self, 
        task_fn, 
        task_params=None
    ):
        """Set the task for this dataset.
        
        This method processes the dataset to create samples for the specified task.
        
        Args:
            task_fn: The task function to apply to each patient/visit.
            task_params: Parameters for the task function. Defaults to None.
            
        Returns:
            BioimpedanceDataset: The dataset itself with processed samples.
        """
        self.samples = []
        self.patient_to_index = {}
        self.record_to_index = {}
        task_params = task_params or {}
        
        print("Creating samples for task...")
        for patient in tqdm(self.patients.values()):
            patient_samples = task_fn(patient, **task_params)
            if patient_samples:
                if isinstance(patient_samples, list):
                    start_idx = len(self.samples)
                    self.samples.extend(patient_samples)
                    
                    # Create patient_to_index mapping
                    patient_id = patient.patient_id
                    if patient_id not in self.patient_to_index:
                        self.patient_to_index[patient_id] = []
                    
                    # Add indices for all samples from this patient
                    for i in range(start_idx, len(self.samples)):
                        self.patient_to_index[patient_id].append(i)
                        
                        # Create record_to_index mapping
                        sample = self.samples[i]
                        record_id = sample.get('record_id', sample.get('visit_id'))
                        if record_id is not None:
                            if record_id not in self.record_to_index:
                                self.record_to_index[record_id] = []
                            self.record_to_index[record_id].append(i)
                else:
                    idx = len(self.samples)
                    self.samples.append(patient_samples)
                    
                    # Create patient_to_index mapping
                    patient_id = patient.patient_id
                    if patient_id not in self.patient_to_index:
                        self.patient_to_index[patient_id] = []
                    self.patient_to_index[patient_id].append(idx)
                    
                    # Create record_to_index mapping
                    record_id = patient_samples.get('record_id', patient_samples.get('visit_id'))
                    if record_id is not None:
                        if record_id not in self.record_to_index:
                            self.record_to_index[record_id] = []
                        self.record_to_index[record_id].append(idx)
        
        print(f"Created {len(self.samples)} samples.")
        return self

    def info(self):
        """Print information about the dataset."""
        print(f"Bioimpedance Dataset: {self.dataset_name}")
        print(f"Number of patients: {len(self.patients)}")
        print(f"Number of visits: {len(self.visits)}")
        print(f"Number of samples: {len(self.samples)}")
        print(f"Synthetic data: {'Yes' if self.synthetic else 'No'}")
        print(f"Signal length: {self.signal_length}")
        print(f"Overlap: {self.overlap}")
        print(f"Normalize: {self.normalize}")
        print(f"Sampling rate: {self.sampling_rate} Hz")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        """Returns a sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict: A dictionary containing bioimpedance signal and blood pressure data.
        """
        if not self.samples:
            raise ValueError("Dataset samples are not initialized. Call set_task() first.")
        return self.samples[index]