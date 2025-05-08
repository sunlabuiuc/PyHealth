import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
#from pyhealth.tasks import BaseTask
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("./PyHealth")


class ICUStayPredictionTask(BaseTask):
    """
    A task to predict ICU stay outcomes (ICU admission or ICU length of stay)
    using structured data (demographics), time-series vital signs, and unstructured clinical notes.
    """

    def __init__(self, data_dir: str, tokenizer_name: str, time_window: int = 48,
                 outcome_label: str = 'ICU_admission', max_length: int = 512, batch_size: int = 32):
        """
        Args:
            data_dir (str): Directory containing MIMIC-III data files (CSV for structured data, TXT for notes).
            tokenizer_name (str): Pre-trained tokenizer (e.g., 'bert-base-uncased') for clinical notes.
            time_window (int): Number of time steps (e.g., hours) to consider for time-series data.
            outcome_label (str): Column name in structured data indicating ICU-related outcome.
            max_length (int): Maximum sequence length for tokenization (for clinical notes).
            batch_size (int): Batch size for training.
        """
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.time_window = time_window
        self.outcome_label = outcome_label
        self.max_length = max_length
        self.batch_size = batch_size

        # Load structured data (demographics, diagnoses, etc.)
        self.patients_df = pd.read_csv(os.path.join(data_dir, 'patients.csv'))

        # Load vital signs time-series data (heart rate, blood pressure, etc.)
        self.vital_signs_df = pd.read_csv(os.path.join(data_dir, 'vital_signs.csv'))

        # Directory for clinical notes (e.g., discharge summaries)
        self.notes_dir = os.path.join(data_dir, 'notes')
        self.notes_files = os.listdir(self.notes_dir)

        # Preprocessing scaler for structured data
        self.scaler = StandardScaler()
        self.scaler.fit(self.patients_df[['age', 'height', 'weight']])  # Standardize demographics

        # Load dataset into memory for training
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dict[str, Any]:
        """Preprocesses and combines all relevant data for prediction."""
        dataset = {
            'patient_ids': [],
            'time_series_data': [],
            'structured_data': [],
            'notes_tokens': [],
            'outcomes': []
        }

        for _, row in self.patients_df.iterrows():
            patient_id = row['patient_id']
            outcome = row[self.outcome_label]

            # Process time-series vital signs data
            vital_signs = self.vital_signs_df[self.vital_signs_df['patient_id'] == patient_id]
            time_series_data = self._process_time_series(vital_signs)

            # Process clinical notes
            notes = self._process_notes(patient_id)

            # Process structured data (e.g., demographics)
            structured_data = self._process_structured_data(row)

            # Append data to the dataset
            dataset['patient_ids'].append(patient_id)
            dataset['time_series_data'].append(time_series_data)
            dataset['structured_data'].append(structured_data)
            dataset['notes_tokens'].append(notes)
            dataset['outcomes'].append(outcome)

        return dataset

    def _process_time_series(self, vital_signs: pd.DataFrame) -> np.ndarray:
        """Process time-series vital signs (e.g., heart rate, blood pressure)"""
        vital_signs = vital_signs[['time', 'heart_rate', 'systolic_bp', 'diastolic_bp']]

        # Resample time-series data to a consistent time window (e.g., hourly)
        vital_signs = vital_signs.set_index('time').resample('H').mean().fillna(0)

        # Normalize the vital signs data
        normalized_data = self.scaler.transform(vital_signs[['heart_rate', 'systolic_bp', 'diastolic_bp']].values)

        # Select the last `time_window` hours of data
        time_series_data = normalized_data[-self.time_window:]
        return time_series_data

    def _process_notes(self, patient_id: str) -> str:
        """Process and retrieve the clinical notes for a given patient."""
        note_files = [f for f in self.notes_files if f.startswith(patient_id)]
        notes_text = ""
        for file in note_files:
            with open(os.path.join(self.notes_dir, file), 'r') as f:
                notes_text += f.read()
        # Tokenize the clinical notes using pre-trained tokenizer
        return self.tokenizer(notes_text, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')

    def _process_structured_data(self, row: pd.Series) -> np.ndarray:
        """Process structured data (e.g., patient demographics, diagnoses)"""
        demographics = np.array([row['age'], row['height'], row['weight']])
        demographics = self.scaler.transform([demographics])[0]
        return demographics

    def get_data_loader(self) -> DataLoader:
        """Returns a DataLoader for the dataset."""
        # Convert data to tensors
        time_series_data = torch.tensor(self.dataset['time_series_data'], dtype=torch.float32)
        structured_data = torch.tensor(self.dataset['structured_data'], dtype=torch.float32)
        outcomes = torch.tensor(self.dataset['outcomes'], dtype=torch.float32)

        # Create a PyTorch DataLoader
        data = torch.utils.data.TensorDataset(time_series_data, structured_data, outcomes)
        return DataLoader(data, batch_size=self.batch_size, shuffle=True)

    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader) -> float:
        """Evaluates the model on the dataset and returns the accuracy (or other metrics)."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for time_series, structured_data, outcomes in data_loader:
                outputs = model(time_series, structured_data)
                predictions = (outputs > 0.5).float()  # Binary classification
                correct += (predictions == outcomes).sum().item()
                total += outcomes.size(0)

        accuracy = correct / total
        return accuracy

# Example Usage
if __name__ == '__main__':
    dataset = ICUStayPredictionTask(data_dir='data/', text_encoder=lambda x: x[:100])
    print("Total samples:", len(dataset))

    sample_data, sample_label = dataset[0]
    print("Sample Data:", sample_data)
    print("Sample Label:", sample_label)