import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from pyhealth.datasets import BaseDataset
from pyhealth.datasets.utils import strptime


class CorGANDataset(BaseDataset):
    """
    Dataset class for CorGAN that preprocesses medical data into binary matrices.
    
    This dataset converts medical records (like MIMIC-III/IV) into binary matrices
    suitable for training CorGAN, similar to the synthEHRella preprocessing pipeline.
    
    Args:
        dataset_name: Name of the dataset (e.g., "mimic3", "mimic4")
        root: Root directory of the dataset
        tables: List of table names to use
        code_mapping: Dictionary mapping table names to code columns
        visit_mapping: Dictionary mapping table names to visit ID columns
        patient_mapping: Dictionary mapping table names to patient ID columns
        timestamp_mapping: Dictionary mapping table names to timestamp columns
        **kwargs: Additional arguments passed to BaseDataset
    
    Examples:
        >>> dataset = CorGANDataset(
        ...     dataset_name="mimic3",
        ...     root="path/to/mimic3",
        ...     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD"],
        ...     code_mapping={"DIAGNOSES_ICD": "ICD9_CODE", "PROCEDURES_ICD": "ICD9_CODE"},
        ...     visit_mapping={"DIAGNOSES_ICD": "HADM_ID", "PROCEDURES_ICD": "HADM_ID"},
        ...     patient_mapping={"DIAGNOSES_ICD": "SUBJECT_ID", "PROCEDURES_ICD": "SUBJECT_ID"},
        ...     timestamp_mapping={"DIAGNOSES_ICD": "CHARTTIME", "PROCEDURES_ICD": "CHARTTIME"}
        ... )
        >>> binary_matrix = dataset.get_binary_matrix()
    """
    
    def __init__(
        self,
        dataset_name: str,
        root: str,
        tables: List[str],
        code_mapping: Dict[str, str],
        visit_mapping: Dict[str, str],
        patient_mapping: Dict[str, str],
        timestamp_mapping: Dict[str, str],
        **kwargs
    ):
        super(CorGANDataset, self).__init__(dataset_name=dataset_name, root=root, **kwargs)
        
        self.tables = tables
        self.code_mapping = code_mapping
        self.visit_mapping = visit_mapping
        self.patient_mapping = patient_mapping
        self.timestamp_mapping = timestamp_mapping
        
        # load and preprocess data
        self._load_data()
        self._preprocess_data()
    
    def _load_data(self):
        """Load data from CSV files"""
        self.data = {}
        
        for table in self.tables:
            file_path = os.path.join(self.root, f"{table}.csv")
            if os.path.exists(file_path):
                self.data[table] = pd.read_csv(file_path)
                print(f"Loaded {table}: {len(self.data[table])} records")
            else:
                print(f"Warning: {file_path} not found")
    
    def _preprocess_data(self):
        """Preprocess data into patient-visit-code format"""
        print("Preprocessing data...")
        
        # build patient-admission mapping
        pid_adm_map = {}
        adm_date_map = {}
        
        # assume ADMISSIONS table exists for visit timestamps
        if "ADMISSIONS" in self.data:
            for _, row in self.data["ADMISSIONS"].iterrows():
                pid = row.get("SUBJECT_ID", row.get("patient_id"))
                adm_id = row.get("HADM_ID", row.get("visit_id"))
                adm_time = strptime(row.get("ADMITTIME", row.get("timestamp")))
                
                if pd.notna(pid) and pd.notna(adm_id) and adm_time is not None:
                    adm_date_map[adm_id] = adm_time
                    if pid in pid_adm_map:
                        pid_adm_map[pid].append(adm_id)
                    else:
                        pid_adm_map[pid] = [adm_id]
        
        # build admission-code mapping
        adm_code_map = {}
        self.code_types = {}  # mapping from original codes to processed codes
        
        for table in self.tables:
            if table not in self.data:
                continue
                
            code_col = self.code_mapping.get(table, "code")
            visit_col = self.visit_mapping.get(table, "visit_id")
            
            for _, row in self.data[table].iterrows():
                adm_id = row.get(visit_col)
                code = row.get(code_col)
                
                if pd.notna(adm_id) and pd.notna(code):
                    # convert to 3-digit ICD9 if needed
                    processed_code = self._convert_to_3digit_icd9(str(code))
                    code_str = f"D_{processed_code}"
                    
                    # track code mapping
                    if code not in self.code_types:
                        self.code_types[code] = processed_code
                    
                    if adm_id in adm_code_map:
                        adm_code_map[adm_id].append(code_str)
                    else:
                        adm_code_map[adm_id] = [code_str]
        
        # build patient-sorted visits mapping
        pid_seq_map = {}
        for pid, adm_list in pid_adm_map.items():
            sorted_list = []
            for adm_id in adm_list:
                if adm_id in adm_date_map and adm_id in adm_code_map:
                    sorted_list.append((adm_date_map[adm_id], adm_code_map[adm_id]))
            
            if sorted_list:
                sorted_list.sort(key=lambda x: x[0])  # sort by date
                pid_seq_map[pid] = sorted_list
        
        # convert to sequences
        self.pids = []
        self.dates = []
        self.seqs = []
        
        for pid, visits in pid_seq_map.items():
            self.pids.append(pid)
            seq = []
            date = []
            for visit in visits:
                date.append(visit[0])
                seq.append(visit[1])
            self.dates.append(date)
            self.seqs.append(seq)
        
        # convert string sequences to integer sequences
        self.types = {}
        self.new_seqs = []
        
        for patient in self.seqs:
            new_patient = []
            for visit in patient:
                new_visit = []
                for code in visit:
                    if code in self.types:
                        new_visit.append(self.types[code])
                    else:
                        self.types[code] = len(self.types)
                        new_visit.append(self.types[code])
                new_patient.append(new_visit)
            self.new_seqs.append(new_patient)
        
        print(f"Preprocessing complete: {len(self.pids)} patients, {len(self.types)} unique codes")
    
    def _convert_to_3digit_icd9(self, dx_str: str) -> str:
        """Convert ICD9 code to 3-digit format"""
        if dx_str.startswith('E'):
            if len(dx_str) > 4:
                return dx_str[:4]
            else:
                return dx_str
        else:
            if len(dx_str) > 3:
                return dx_str[:3]
            else:
                return dx_str
    
    def get_binary_matrix(self) -> np.ndarray:
        """Get binary matrix representation of the data"""
        num_patients = len(self.new_seqs)
        num_codes = len(self.types)
        
        matrix = np.zeros((num_patients, num_codes), dtype=np.float32)
        
        for i, patient in enumerate(self.new_seqs):
            for visit in patient:
                for code in visit:
                    matrix[i][code] = 1.0
        
        return matrix
    
    def get_count_matrix(self) -> np.ndarray:
        """Get count matrix representation of the data"""
        num_patients = len(self.new_seqs)
        num_codes = len(self.types)
        
        matrix = np.zeros((num_patients, num_codes), dtype=np.float32)
        
        for i, patient in enumerate(self.new_seqs):
            for visit in patient:
                for code in visit:
                    matrix[i][code] += 1.0
        
        return matrix
    
    def save_processed_data(self, output_path: str, matrix_type: str = "binary"):
        """Save processed data to files"""
        os.makedirs(output_path, exist_ok=True)
        
        # save patient IDs
        with open(os.path.join(output_path, "processed.pids"), "wb") as f:
            pickle.dump(self.pids, f, -1)
        
        # save code types mapping
        with open(os.path.join(output_path, "processed.types"), "wb") as f:
            pickle.dump(self.types, f, -1)
        
        # save matrix
        if matrix_type == "binary":
            matrix = self.get_binary_matrix()
        elif matrix_type == "count":
            matrix = self.get_count_matrix()
        else:
            raise ValueError("matrix_type must be 'binary' or 'count'")
        
        with open(os.path.join(output_path, "processed.matrix"), "wb") as f:
            pickle.dump(matrix, f, -1)
        
        # also save as numpy array for convenience
        np.save(os.path.join(output_path, "processed_matrix.npy"), matrix)
        
        print(f"Saved processed data to {output_path}")
        print(f"Matrix shape: {matrix.shape}")
        print(f"Matrix type: {matrix_type}")
    
    def get_patient_visits(self, patient_id: str) -> List[List[str]]:
        """Get visits for a specific patient"""
        if patient_id not in self.pids:
            return []
        
        idx = self.pids.index(patient_id)
        return self.seqs[idx]
    
    def get_code_mapping(self) -> Dict[str, int]:
        """Get mapping from code strings to integer indices"""
        return self.types.copy()
    
    def get_reverse_code_mapping(self) -> Dict[int, str]:
        """Get mapping from integer indices to code strings"""
        return {v: k for k, v in self.types.items()}
    
    def get_patient_ids(self) -> List[str]:
        """Get list of patient IDs"""
        return self.pids.copy()
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        matrix = self.get_binary_matrix()
        
        stats = {
            "num_patients": len(self.pids),
            "num_codes": len(self.types),
            "num_visits": sum(len(patient) for patient in self.seqs),
            "avg_visits_per_patient": np.mean([len(patient) for patient in self.seqs]),
            "avg_codes_per_visit": np.mean([len(code) for patient in self.seqs for code in patient]),
            "sparsity": 1.0 - np.mean(matrix),
            "matrix_shape": matrix.shape,
        }
        
        return stats 