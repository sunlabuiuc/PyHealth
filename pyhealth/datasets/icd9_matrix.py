"""
Simple ICD-9 matrix utilities for MIMIC-III data.
No conversions - keeps original ICD-9 format.
"""

import os
import numpy as np
import torch
from typing import Dict, Tuple
from torch.utils.data import Dataset

from .base_dataset import BaseDataset


def convert_to_3digit_icd9(dxStr):
    """Convert ICD-9 to 3-digit format"""
    if dxStr.startswith('E'):
        if len(dxStr) > 4: 
            return dxStr[:4]
        else: 
            return dxStr
    else:
        if len(dxStr) > 3: 
            return dxStr[:3]
        else: 
            return dxStr


def create_icd9_matrix(dataset: BaseDataset, output_path: str = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Create ICD-9 binary matrix from MIMIC3Dataset.
    
    Args:
        dataset: MIMIC3Dataset instance
        output_path: Optional path to save matrix
        
    Returns:
        Tuple of (matrix, icd9_code_to_index_mapping)
    """
    print("Processing ICD-9 codes...")
    
    # Collect all ICD codes from patients
    icd_codes = set()
    patient_icd_map = {}
    
    for patient in dataset.iter_patients():
        patient_codes = set()
        
        # Get events from the diagnoses_icd table
        events = patient.get_events(event_type="diagnoses_icd")
        
        for event in events:
            # Check if this is an ICD-9 diagnosis
            if "icd9_code" in event.attr_dict and event.attr_dict["icd9_code"]:
                # Use 3-digit truncation
                code = convert_to_3digit_icd9(event.attr_dict["icd9_code"])
                icd_codes.add(code)
                patient_codes.add(code)
        
        if patient_codes:
            patient_icd_map[patient.patient_id] = patient_codes
    
    # Create ICD-9 matrix
    icd_codes_list = sorted(list(icd_codes))
    icd9_types = {code: idx for idx, code in enumerate(icd_codes_list)}
    
    num_patients = len(patient_icd_map)
    num_codes = len(icd9_types)
    
    print(f"Creating ICD-9 matrix: {num_patients} patients x {num_codes} codes")
    
    icd9_matrix = np.zeros((num_patients, num_codes), dtype=np.float32)
    
    for i, (patient_id, codes) in enumerate(patient_icd_map.items()):
        for code in codes:
            if code in icd9_types:
                icd9_matrix[i, icd9_types[code]] = 1.0
    
    # Save matrix if output path provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, "icd9_matrix.npy"), icd9_matrix)
        print(f"Saved ICD-9 matrix to {output_path}/icd9_matrix.npy")
    
    print(f"Final matrix shape: {icd9_matrix.shape}")
    print(f"Sparsity: {1.0 - np.mean(icd9_matrix):.3f}")
    
    return icd9_matrix, icd9_types


class ICD9MatrixDataset(Dataset):
    """Simple dataset wrapper for ICD-9 matrix"""
    
    def __init__(self, icd9_matrix: np.ndarray):
        self.icd9_matrix = icd9_matrix
    
    def __len__(self):
        return self.icd9_matrix.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.icd9_matrix[idx], dtype=torch.float32)