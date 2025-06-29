"""
Phecode dataset with proper ICD-9 to ICD-10 to PhecodeX transformations.
Based on SynthEHRella's approach for medical coding standardization.
"""

import os
import pickle
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .base_dataset import BaseDataset


def convert_to_3digit_icd9(dxStr):
    """convert ICD-9 to 3-digit format"""
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


class PhecodeTransformer:
    """transformer for ICD-9 to ICD-10 to PhecodeX conversions"""
    
    def __init__(self, mapping_dir: Optional[str] = None):
        if mapping_dir is None:
            # use default mapping directory
            mapping_dir = os.path.join(os.path.dirname(__file__), "phecode_mappings")
        
        self.mapping_dir = mapping_dir
        self._load_mappings()
    
    def _load_mappings(self):
        """load all mapping files"""
        # ICD-9 to ICD-10 mapping
        with open(os.path.join(self.mapping_dir, "ICD9toICD10Mapping.json"), 'r') as f:
            self.icd9_to_icd10_mapping = json.load(f)
        
        # ICD-10 types
        with open(os.path.join(self.mapping_dir, "ICD10types.json"), 'r') as f:
            self.icd10_types_dict = json.load(f)
        
        # ICD-10 to PhecodeX mapping
        with open(os.path.join(self.mapping_dir, "icd10_to_phecodex_mapping.json"), 'r') as f:
            self.icd10_to_phecodex_mapping = json.load(f)
        
        # PhecodeX types
        with open(os.path.join(self.mapping_dir, "phecodex_types.json"), 'r') as f:
            self.phecodex_types_dict = json.load(f)
    
    def icd9_to_icd10_matrix(self, icd9_matrix: np.ndarray, icd9_types: Dict[str, int]) -> np.ndarray:
        """convert ICD-9 matrix to ICD-10 matrix like SynthEHRella"""
        n, p = icd9_matrix.shape
        
        # initialize ICD-10 matrix
        icd10_matrix = np.zeros((n, len(self.icd10_types_dict)), dtype=int)
        
        # create reverse mapping from index to ICD-9 code
        idx_to_icd9 = {idx: code for code, idx in icd9_types.items()}
        
        # create reverse mapping from integer index to string ICD-10 code
        idx_to_icd10_string = {idx: code for code, idx in self.icd10_types_dict.items()}
        
        # create mapping from ICD-9 indices to ICD-10 codes
        icd9_to_icd10_mappings = []
        for j in range(p):
            icd9_code = idx_to_icd9[j]  # Get the actual ICD-9 code for this index
            if icd9_code in self.icd9_to_icd10_mapping:
                for icd10_idx_int in self.icd9_to_icd10_mapping[icd9_code]:
                    # Convert integer index to string ICD-10 code
                    if icd10_idx_int in idx_to_icd10_string:
                        icd10_code_string = idx_to_icd10_string[icd10_idx_int]
                        # Get the final integer index for this ICD-10 code
                        if icd10_code_string in self.icd10_types_dict:
                            final_icd10_idx = self.icd10_types_dict[icd10_code_string]
                            icd9_to_icd10_mappings.append((j, final_icd10_idx))
        
        # apply mappings
        for icd9_idx, icd10_idx in icd9_to_icd10_mappings:
            icd10_matrix[:, icd10_idx] = np.maximum(icd10_matrix[:, icd10_idx], icd9_matrix[:, icd9_idx])
        
        return icd10_matrix
    
    def icd10_to_phecodex(self, icd10_matrix: np.ndarray) -> np.ndarray:
        """convert ICD-10 matrix to PhecodeX matrix"""
        n, p = icd10_matrix.shape
        
        # initialize phecodex matrix
        phecodex_matrix = np.zeros((n, len(self.phecodex_types_dict)), dtype=int)
        
        # create reverse mapping from integer index to string PhecodeX code
        idx_to_phecodex_string = {idx: code for code, idx in self.phecodex_types_dict.items()}
        
        # apply mappings
        for icd10_idx_int in range(p):
            # Convert integer index to string for mapping lookup
            icd10_idx_str = str(icd10_idx_int)
            
            if icd10_idx_str in self.icd10_to_phecodex_mapping:
                for phecodex_idx_int in self.icd10_to_phecodex_mapping[icd10_idx_str]:
                    if phecodex_idx_int in idx_to_phecodex_string:
                        phecodex_code_string = idx_to_phecodex_string[phecodex_idx_int]
                        if phecodex_code_string in self.phecodex_types_dict:
                            phecodex_idx = self.phecodex_types_dict[phecodex_code_string]
                        phecodex_matrix[:, phecodex_idx] = np.maximum(
                            phecodex_matrix[:, phecodex_idx], 
                                icd10_matrix[:, icd10_idx_int]
                        )
        
        return phecodex_matrix


class PhecodeDataset:
    """phecode dataset with proper ICD transformations"""
    
    def __init__(
        self,
        base_dataset: BaseDataset,
        output_path: str = "./phecode_data",
        use_phecode_mapping: bool = True,
        **kwargs
    ):
        # Store the base dataset and its config
        self.base_dataset = base_dataset
        self.config = base_dataset.config
        self.dataset_name = base_dataset.dataset_name
        self.root = base_dataset.root
        self.tables = base_dataset.tables
        self.dev = base_dataset.dev
        
        self.output_path = output_path
        self.use_phecode_mapping = use_phecode_mapping
        
        # create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # initialize transformer if using phecode mapping
        if self.use_phecode_mapping:
            self.transformer = PhecodeTransformer()
        
        # process data
        self._process_data()
    
    def _process_data(self):
        """process data to create phecode matrix"""
        print("Processing ICD codes...")
        
        # collect all ICD codes from patients
        icd_codes = set()
        patient_icd_map = {}
        
        for patient in self.base_dataset.iter_patients():
            patient_codes = set()
            
            # Get events from the diagnoses_icd table
            events = patient.get_events(event_type="diagnoses_icd")
            
            for event in events:
                # Check if this is an ICD-9 diagnosis
                if "icd9_code" in event.attr_dict and event.attr_dict["icd9_code"]:
                    # use 3-digit truncation like SynthEHRella
                    code = convert_to_3digit_icd9(event.attr_dict["icd9_code"])
                    icd_codes.add(code)
                    patient_codes.add(code)
            
            if patient_codes:
                patient_icd_map[patient.patient_id] = patient_codes
        
        # create ICD-9 matrix
        icd_codes_list = sorted(list(icd_codes))
        self.icd9_types = {code: idx for idx, code in enumerate(icd_codes_list)}
        
        num_patients = len(patient_icd_map)
        num_codes = len(self.icd9_types)
        
        print(f"Creating ICD-9 matrix: {num_patients} patients x {num_codes} codes")
        
        icd9_matrix = np.zeros((num_patients, num_codes), dtype=np.float32)
        
        for i, (patient_id, codes) in enumerate(patient_icd_map.items()):
            for code in codes:
                if code in self.icd9_types:
                    icd9_matrix[i, self.icd9_types[code]] = 1.0
        
        # save ICD-9 matrix
        np.save(os.path.join(self.output_path, "icd9_matrix.npy"), icd9_matrix)
        
        if self.use_phecode_mapping:
            # transform to ICD-10
            print("Transforming ICD-9 to ICD-10...")
            icd10_matrix = self.transformer.icd9_to_icd10_matrix(icd9_matrix, self.icd9_types)
            np.save(os.path.join(self.output_path, "icd10_matrix.npy"), icd10_matrix)
            
            # transform to PhecodeX
            print("Transforming ICD-10 to PhecodeX...")
            phecodex_matrix = self.transformer.icd10_to_phecodex(icd10_matrix)
            np.save(os.path.join(self.output_path, "phecodex_matrix.npy"), phecodex_matrix)
            
            self.phecode_matrix = phecodex_matrix
        else:
            # use raw ICD-9 matrix
            self.phecode_matrix = icd9_matrix
        
        print(f"Final matrix shape: {self.phecode_matrix.shape}")
        print(f"Sparsity: {1.0 - np.mean(self.phecode_matrix):.3f}")
    
    def get_phecode_matrix(self) -> np.ndarray:
        """get the phecode matrix"""
        return self.phecode_matrix
    
    def get_phecode_mapping(self) -> Dict:
        """get the phecode mapping information"""
        mapping_info = {
            'matrix_shape': self.phecode_matrix.shape,
            'use_phecode_mapping': self.use_phecode_mapping,
            'output_path': self.output_path
        }
        
        if self.use_phecode_mapping:
            # Add transformer mapping info
            mapping_info.update({
                'icd10_types_count': len(self.transformer.icd10_types_dict),
                'phecodex_types_count': len(self.transformer.phecodex_types_dict),
                'mapping_dir': self.transformer.mapping_dir
            })
            
            # Add the actual mapping dictionaries for postprocessing
            # These need to be in the format expected by the postprocessing function
            # where keys are string indices and values are lists of target codes
            
            # ICD-9 to ICD-10 mapping (index-based)
            icd9_to_icd10 = {}
            # We need to use the icd9_types that were created during data processing
            # This contains the actual ICD-9 codes found in the data and their indices
            for icd9_code, icd9_idx in self.icd9_types.items():
                # Convert the ICD-9 code to match the mapping file format
                # The mapping file uses numeric keys (e.g., "8") while data has string codes (e.g., "008")
                # Remove leading zeros to match the mapping format
                icd9_code_numeric = str(int(icd9_code)) if icd9_code.isdigit() else icd9_code
                
                if icd9_code_numeric in self.transformer.icd9_to_icd10_mapping:
                    # Convert ICD-10 codes to indices
                    icd10_indices = []
                    for icd10_code in self.transformer.icd9_to_icd10_mapping[icd9_code_numeric]:
                        if icd10_code in self.transformer.icd10_types_dict:
                            icd10_indices.append(self.transformer.icd10_types_dict[icd10_code])
                    if icd10_indices:
                        icd9_to_icd10[str(icd9_idx)] = icd10_indices
            
            # ICD-10 to PhecodeX mapping (index-based)
            icd10_to_phecodex = {}
            for icd10_code, icd10_idx in self.transformer.icd10_types_dict.items():
                icd10_idx_str = str(icd10_idx)
                if icd10_idx_str in self.transformer.icd10_to_phecodex_mapping:
                    phecodex_indices = []
                    for phecodex_idx_int in self.transformer.icd10_to_phecodex_mapping[icd10_idx_str]:
                        # Convert integer index to string code, then to final index
                        for phecodex_code, phecodex_idx in self.transformer.phecodex_types_dict.items():
                            if phecodex_idx == phecodex_idx_int:
                                phecodex_indices.append(phecodex_idx)
                                break
                    if phecodex_indices:
                        icd10_to_phecodex[str(icd10_idx)] = phecodex_indices
            
            # PhecodeX to PhecodeXM mapping (index-based)
            # Load the phecodex_to_phecodexm mapping file
            phecodex_to_phecodexm_path = os.path.join(self.transformer.mapping_dir, "phecodex_to_phecodexm_mapping.json")
            phecodexm_types_path = os.path.join(self.transformer.mapping_dir, "phecodexm_types.json")
            
            with open(phecodex_to_phecodexm_path, 'r') as f:
                phecodex_to_phecodexm_mapping = json.load(f)
            
            with open(phecodexm_types_path, 'r') as f:
                phecodexm_types_dict = json.load(f)
            
            phecodex_to_phecodexm = {}
            for phecodex_code, phecodex_idx in self.transformer.phecodex_types_dict.items():
                phecodex_idx_str = str(phecodex_idx)
                if phecodex_idx_str in phecodex_to_phecodexm_mapping:
                    phecodexm_idx = phecodex_to_phecodexm_mapping[phecodex_idx_str]
                    # FIXED: phecodexm_idx is already the correct index, no need to check against phecodexm_types_dict
                    # The mapping file contains PhecodeXM indices directly
                    phecodex_to_phecodexm[str(phecodex_idx)] = [phecodexm_idx]
            
            # Add all the mapping dictionaries
            mapping_info.update({
                'icd9_to_icd10': icd9_to_icd10,
                'icd10_to_phecodex': icd10_to_phecodex,
                'phecodex_to_phecodexm': phecodex_to_phecodexm,
                'icd10_types': self.transformer.icd10_types_dict,
                'phecodex_types': self.transformer.phecodex_types_dict,
                'phecodexm_types': phecodexm_types_dict
            })
        
        return mapping_info


class PhecodeMatrixDataset:
    """dataset wrapper for phecode matrix"""
    
    def __init__(self, phecode_matrix: np.ndarray):
        self.phecode_matrix = phecode_matrix
    
    def __len__(self):
        return self.phecode_matrix.shape[0]
    
    def __getitem__(self, idx):
        return torch.tensor(self.phecode_matrix[idx], dtype=torch.float32) 