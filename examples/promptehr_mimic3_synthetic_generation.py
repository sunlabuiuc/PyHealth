"""
PromptEHR MIMIC-III Synthetic Data Generation Pipeline

This script implements the complete pipeline for generating synthetic MIMIC-III EHR data
using our restored PyHealth PromptEHR implementation, following the synthEHRella approach
but with full training capability.

Pipeline:
1. MIMIC-III Data Preprocessing (PyHealth)
2. PromptEHR Training (Restored Training Pipeline)
3. Synthetic Data Generation 
4. Format Conversion (Compatible with MedGAN/CorGAN evaluations)

Usage:
    # Full pipeline (training + generation)
    python promptehr_mimic3_synthetic_generation.py --mode train_and_generate --mimic_root ./data_files --output_dir ./promptehr_synthetic

    # Generation only (using pretrained model)  
    python promptehr_mimic3_synthetic_generation.py --mode generate_only --model_path ./trained_promptehr --output_dir ./promptehr_synthetic

    # Preprocessing only
    python promptehr_mimic3_synthetic_generation.py --mode preprocess_only --mimic_root ./data_files --output_dir ./promptehr_preprocessed
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import json
from tqdm import tqdm

# Add PyHealth to path
sys.path.append(str(Path(__file__).parent))
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyhealth', 'models', 'generators'))
from promptehr import PromptEHR
from pyhealth.datasets import SampleDataset, MIMIC3Dataset
from pyhealth.tasks import BaseTask

warnings.filterwarnings('ignore')

class MIMIC3PromptEHRTask(BaseTask):
    
    def __init__(
        self,
        max_visits_per_patient: int = 20,
        min_visits_per_patient: int = 2,
        include_procedures: bool = True,
        include_medications: bool = True,
        code_vocab_threshold: int = 5,
        convert_to_3digit_icd9: bool = True
    ):
        super().__init__()
        self.task_name = "MIMIC3_PromptEHR"
        self.input_schema = {}
        self.output_schema = {}
        self.max_visits_per_patient = max_visits_per_patient
        self.min_visits_per_patient = min_visits_per_patient
        self.include_procedures = include_procedures
        self.include_medications = include_medications
        self.code_vocab_threshold = code_vocab_threshold
        self.convert_to_3digit_icd9 = convert_to_3digit_icd9
    
    def _convert_to_3digit_icd9(self, dx_str: str) -> str:
        if dx_str.startswith('E'):
            if len(dx_str) > 1:
                numeric_part = dx_str[1:]  # Remove 'E' prefix
                if len(numeric_part) > 3:
                    numeric_part = numeric_part[:3]
                try:
                    num = int(numeric_part)
                    return str(800 + (num % 200))
                except:
                    return '800'
            else:
                return '800'
        elif dx_str.startswith('V'):
            if len(dx_str) > 1:
                numeric_part = dx_str[1:]  # Remove 'V' prefix
                if len(numeric_part) > 2:
                    numeric_part = numeric_part[:2]
                try:
                    num = int(numeric_part)
                    return str(700 + (num % 100))
                except:
                    return '700'
            else:
                return '700'
        else:
            if len(dx_str) > 3: 
                return dx_str[:3]
            else: 
                return dx_str
    
    def __call__(self, patient) -> List[Dict]:
        
        # Get patient admissions
        admissions = patient.get_events(event_type="admissions")
        
        if len(admissions) < self.min_visits_per_patient:
            return []
        
        all_diagnoses = patient.get_events(event_type="diagnoses_icd")
        all_procedures = patient.get_events(event_type="procedures_icd") if self.include_procedures else []
        all_medications = patient.get_events(event_type="prescriptions") if self.include_medications else []
        
        diag_codes = []
        for diagnosis in all_diagnoses:
            icd9_code = diagnosis.attr_dict.get('icd9_code')
            if icd9_code:
                code = str(icd9_code).strip()
                if self.convert_to_3digit_icd9:
                    code = self._convert_to_3digit_icd9(code)
                diag_codes.append(code)
        
        proc_codes = []
        if self.include_procedures:
            for procedure in all_procedures:
                icd9_code = procedure.attr_dict.get('icd9_code')
                if icd9_code:
                    proc_codes.append(str(icd9_code).strip())
        
        med_codes = []
        if self.include_medications:
            for prescription in all_medications:
                drug = prescription.attr_dict.get('drug')
                if drug:
                    drug_name = str(drug).strip()
                    import hashlib
                    drug_hash = hashlib.md5(drug_name.encode()).hexdigest()
                    drug_id = abs(int(drug_hash[:8], 16)) % 100000
                    med_codes.append(str(drug_id))
        
        if not diag_codes and not proc_codes and not med_codes:
            return []
        
        num_visits = min(len(admissions), self.max_visits_per_patient)
        visits = []
        
        for i in range(num_visits):
            if i == 0:
                visit_data = {
                    'diag': diag_codes,
                    'proc': proc_codes,
                    'med': med_codes
                }
            else:
                visit_data = {
                    'diag': [],
                    'proc': [],
                    'med': []
                }
            visits.append(visit_data)
        
        diag_visits = [visit.get('diag', []) for visit in visits]
        proc_visits = [visit.get('proc', []) for visit in visits] if self.include_procedures else [[] for _ in visits]
        med_visits = [visit.get('med', []) for visit in visits] if self.include_medications else [[] for _ in visits]
        
        baseline_features = self._extract_baseline_features(patient, admissions[0])
        
        sample = {
            'patient_id': patient.patient_id,
            'v': {
                'diag': diag_visits,
                'proc': proc_visits,
                'med': med_visits
            },
            'x': baseline_features,
            'num_visits': num_visits
        }
        
        return [sample]
    
    def _process_visit(self, patient, admission) -> Dict[str, List[str]]:
        
        visit_codes = {'diag': [], 'proc': [], 'med': []}
        
        from datetime import timedelta
        start_time = admission.timestamp
        discharge_time = admission.attr_dict.get('dischtime')
        if discharge_time:
            try:
                from datetime import datetime
                if isinstance(discharge_time, str):
                    end_time = datetime.strptime(discharge_time, '%Y-%m-%d %H:%M:%S')
                else:
                    end_time = discharge_time
                end_time = end_time + timedelta(hours=24)
            except:
                end_time = admission.timestamp + timedelta(days=30)
        else:
            end_time = admission.timestamp + timedelta(days=30)
        
        try:
            all_diagnoses = patient.get_events(event_type="diagnoses_icd")
            admission_id = admission.attr_dict.get('hadm_id')
            diagnoses = []
            if admission_id:
                for diag in all_diagnoses:
                    if diag.attr_dict.get('hadm_id') == admission_id:
                        diagnoses.append(diag)
            for diagnosis in diagnoses:
                icd9_code = diagnosis.attr_dict.get('icd9_code')
                if icd9_code:
                    code = str(icd9_code).strip()
                    if self.convert_to_3digit_icd9:
                        code = self.convert_to_3digit_icd9(code)
                    visit_codes['diag'].append(code)
        except Exception:
            pass
        
        if self.include_procedures:
            try:
                all_procedures = patient.get_events(event_type="procedures_icd")
                procedures = []
                if admission_id:
                    for proc in all_procedures:
                        if proc.attr_dict.get('hadm_id') == admission_id:
                            procedures.append(proc)
                for procedure in procedures:
                    icd9_code = procedure.attr_dict.get('icd9_code')
                    if icd9_code:
                        visit_codes['proc'].append(str(icd9_code).strip())
            except Exception:
                pass
        
        if self.include_medications:
            try:
                all_prescriptions = patient.get_events(event_type="prescriptions")
                prescriptions = []
                if admission_id:
                    for pres in all_prescriptions:
                        if pres.attr_dict.get('hadm_id') == admission_id:
                            prescriptions.append(pres)
                for prescription in prescriptions:
                    drug = prescription.attr_dict.get('drug')
                    if drug:
                        visit_codes['med'].append(str(drug).strip())
            except Exception:
                pass
        
        for code_type in visit_codes:
            visit_codes[code_type] = list(dict.fromkeys(visit_codes[code_type]))
        
        return visit_codes
    
    def _extract_baseline_features(self, patient, first_admission) -> List[float]:
        
        features = []
        
        age = first_admission.attr_dict.get('age', 65.0)
        if age is None:
            age = 65.0
        features.append(min(float(age) / 100.0, 1.0))
        
        gender = first_admission.attr_dict.get('gender', 'F')
        if gender is None:
            gender = 'F'
        features.append(1.0 if str(gender).upper() == 'M' else 0.0)
        
        admission_type = first_admission.attr_dict.get('admission_type', '').upper()
        if 'EMERGENCY' in admission_type:
            adm_type_val = 1.0
        elif 'ELECTIVE' in admission_type:
            adm_type_val = 0.5
        elif 'URGENT' in admission_type:
            adm_type_val = 0.75
        else:
            adm_type_val = 0.25
        features.append(adm_type_val)
        
        insurance = str(first_admission.attr_dict.get('insurance', '')).upper()
        if 'MEDICARE' in insurance or 'MEDICAID' in insurance:
            ins_val = 1.0
        elif 'PRIVATE' in insurance:
            ins_val = 0.5
        elif 'SELF' in insurance:
            ins_val = 0.25
        else:
            ins_val = 0.0
        features.append(ins_val)
        
        ethnicity = str(first_admission.attr_dict.get('ethnicity', '')).upper()
        features.append(1.0 if 'WHITE' in ethnicity else 0.0)
        features.append(1.0 if 'BLACK' in ethnicity or 'AFRICAN' in ethnicity else 0.0)
        features.append(1.0 if 'HISPANIC' in ethnicity or 'LATINO' in ethnicity else 0.0)
        features.append(1.0 if 'ASIAN' in ethnicity else 0.0)
        
        marital = str(first_admission.attr_dict.get('marital_status', '')).upper()
        features.append(1.0 if 'MARRIED' in marital else 0.0)
        
        language = str(first_admission.attr_dict.get('language', '')).upper()
        features.append(1.0 if language == 'ENGL' or language == 'ENGLISH' or language == '' else 0.0)
        
        return features

def preprocess_mimic3_data(mimic_root: str, output_dir: str, args) -> str:
    print("Preprocessing MIMIC-III data...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading MIMIC-III data from {mimic_root}")
    tables = ["ADMISSIONS", "DIAGNOSES_ICD"]
    if args.include_procedures:
        tables.append("PROCEDURES_ICD")
    if args.include_medications:
        tables.append("PRESCRIPTIONS")
    
    try:
        dataset = MIMIC3Dataset(root=mimic_root, tables=tables)
        print(f"Loaded {len(tables)} tables")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None
    
    print("Applying preprocessing...")
    task = MIMIC3PromptEHRTask(
        max_visits_per_patient=args.max_visits,
        min_visits_per_patient=args.min_visits,
        include_procedures=args.include_procedures,
        include_medications=args.include_medications,
        code_vocab_threshold=args.code_vocab_threshold
    )
    
    sample_dataset = dataset.set_task(task)
    
    if len(sample_dataset.samples) == 0:
        print("No samples generated")
        return None
    
    print(f"Processed {len(sample_dataset.samples)} patients")
    
    print("Building vocabulary...")
    vocab_stats = defaultdict(Counter)
    for sample in sample_dataset.samples:
        for code_type, visits in sample['v'].items():
            for visit_codes in visits:
                vocab_stats[code_type].update(visit_codes)
    
    filtered_vocab = {}
    for code_type, counter in vocab_stats.items():
        filtered_vocab[code_type] = [code for code, count in counter.items() if count >= args.code_vocab_threshold]
        print(f"  {code_type}: {len(filtered_vocab[code_type])} codes (min_freq={args.code_vocab_threshold})")
    
    print("Splitting data")
    np.random.seed(42)
    indices = np.random.permutation(len(sample_dataset.samples))
    split_idx = int(len(sample_dataset.samples) * args.train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_samples = [sample_dataset.samples[i] for i in train_indices]
    val_samples = [sample_dataset.samples[i] for i in val_indices]
    
    print(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
    
    print("Saving data...")
    
    with open(output_path / "train_samples.pkl", "wb") as f:
        pickle.dump(train_samples, f)
    
    with open(output_path / "val_samples.pkl", "wb") as f:
        pickle.dump(val_samples, f)
    
    with open(output_path / "vocabulary.pkl", "wb") as f:
        pickle.dump(filtered_vocab, f)
    
    metadata = {
        'total_patients': len(sample_dataset.samples),
        'train_patients': len(train_samples),
        'val_patients': len(val_samples),
        'vocabulary_sizes': {k: len(v) for k, v in filtered_vocab.items()},
        'code_types': list(filtered_vocab.keys()),
        'preprocessing_args': vars(args)
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved to {output_path}")
    print(f"Total patients: {metadata['total_patients']}")
    print(f"Vocabulary sizes: {metadata['vocabulary_sizes']}")
    
    return str(output_path)

def create_promptehr_dataset(samples: List[Dict]) -> SampleDataset:
    
    dataset = SampleDataset(
        samples=samples,
        input_schema={"v": "raw", "x": "raw"},
        output_schema={}
    )
    
    dataset.metadata = {
        'visit': {'mode': 'dense'},
        'voc': {},
        'max_visit': max(s['num_visits'] for s in samples)
    }
    
    return dataset

def train_promptehr_model(preprocess_dir: str, output_dir: str, args) -> str:
    print("Training PromptEHR model...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocess_path = Path(preprocess_dir)
    
    print("Loading data...")
    with open(preprocess_path / "train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)
    
    with open(preprocess_path / "val_samples.pkl", "rb") as f:
        val_samples = pickle.load(f)
    
    with open(preprocess_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")
    
    train_dataset = create_promptehr_dataset(train_samples)
    val_dataset = create_promptehr_dataset(val_samples)
    
    print("Initializing model...")
    n_features = len(train_samples[0]['x']) if train_samples[0]['x'] else 0
    model = PromptEHR(
        code_type=metadata['code_types'],
        n_num_feature=n_features,
        cat_cardinalities=None,
        epoch=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        output_dir=str(output_path / "training_logs"),
        device=args.device
    )
    
    print(f"Code types: {model.config['code_type']}")
    print(f"Epochs: {model.config['epoch']}")
    print(f"Batch size: {model.config['batch_size']}")
    
    print("Starting training...")
    try:
        model.fit(train_data=train_dataset, val_data=val_dataset)
        print("Training completed")
    except Exception as e:
        print(f"Training failed: {e}")
        return None
    
    model_path = output_path / "trained_model"
    print(f"Saving model to {model_path}")
    model.save_model(str(model_path))
    
    print(f"Model saved to {model_path}")
    return str(model_path)

def generate_synthetic_data(model_path: str, output_dir: str, args) -> str:
    print("Generating synthetic data...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path}")
    try:
        model = PromptEHR()
        model.load_model(model_path)
        print("Model loaded")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
    
    print("Creating seed data...")
    seed_samples = []
    for i in range(min(args.n_seed_samples, 100)):
        seed_sample = {
            'patient_id': f'seed_{i}',
            'v': {
                'diag': [['401', '250'], ['414', '428']],  # Common diagnosis patterns
                'proc': [[], []],
                'med': [[], []]
            },
            'x': np.random.randn(7).tolist()
        }
        seed_samples.append(seed_sample)
    
    seed_dataset = create_promptehr_dataset(seed_samples)
    
    print(f"Generating {args.n_synthetic} samples...")
    try:
        synthetic_results = model.predict(
            test_data=seed_dataset,
            n=args.n_synthetic,
            n_per_sample=args.n_per_sample,
            sample_config={'temperature': args.temperature},
            verbose=True
        )
        print("Generation completed")
    except Exception as e:
        print(f"Generation failed: {e}")
        return None
    
    raw_output_path = output_path / "promptehr_synthetic_raw.pkl"
    with open(raw_output_path, "wb") as f:
        pickle.dump(synthetic_results, f)
    
    print(f"Raw data saved to {raw_output_path}")
    
    print("Converting to binary matrix...")
    binary_matrix = convert_to_binary_matrix(synthetic_results, output_path)
    
    if binary_matrix is not None:
        print(f"Binary matrix saved with shape: {binary_matrix.shape}")
        return str(output_path)
    else:
        return None

def convert_to_binary_matrix(synthetic_results: Dict, output_dir: Path) -> Optional[np.ndarray]:
    
    print("Extracting diagnosis codes...")
    all_diag_codes = set()
    patient_diagnoses = []
    
    for i, patient_visits in enumerate(synthetic_results['visit']):
        patient_diags = set()
        for visit in patient_visits:
            if visit and len(visit) > 0 and len(visit[0]) > 0:
                for diag_code in visit[0]:
                    if isinstance(diag_code, (int, str)):
                        code_str = str(diag_code)
                        if code_str.startswith('E'):
                            if len(code_str) > 4: 
                                code_str = code_str[:4]
                        else:
                            if len(code_str) > 3: 
                                code_str = code_str[:3]
                        
                        patient_diags.add(f'D_{code_str}')
                        all_diag_codes.add(f'D_{code_str}')
        
        patient_diagnoses.append(patient_diags)
    
    if not all_diag_codes:
        print("No diagnosis codes found")
        return None
    
    sorted_codes = sorted(list(all_diag_codes))
    code_to_idx = {code: idx for idx, code in enumerate(sorted_codes)}
    
    print(f"Found {len(sorted_codes)} unique diagnosis codes")
    
    n_patients = len(patient_diagnoses)
    n_features = len(sorted_codes)
    binary_matrix = np.zeros((n_patients, n_features), dtype=np.float32)
    
    for i, patient_diags in enumerate(patient_diagnoses):
        for diag_code in patient_diags:
            if diag_code in code_to_idx:
                binary_matrix[i, code_to_idx[diag_code]] = 1.0
    
    matrix_path = output_dir / "promptehr_synthetic_binary.npy"
    np.save(matrix_path, binary_matrix)
    
    mapping_path = output_dir / "code_mapping.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump({
            'code_to_idx': code_to_idx,
            'idx_to_code': {v: k for k, v in code_to_idx.items()},
            'sorted_codes': sorted_codes
        }, f)
    
    stats = {
        'n_patients': int(n_patients),
        'n_features': int(n_features),
        'sparsity': float(1.0 - (np.count_nonzero(binary_matrix) / binary_matrix.size)),
        'avg_codes_per_patient': float(np.mean(np.sum(binary_matrix, axis=1))),
        'total_unique_codes': len(sorted_codes),
        'timestamp': pd.Timestamp.now().isoformat(),
        'generation_method': 'PromptEHR'
    }
    
    stats_path = output_dir / "generation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print("Creating CSV output...")
    
    patient_data = []
    for i, patient_diags in enumerate(patient_diagnoses):
        patient_data.append({
            'patient_id': f'synthetic_{i:06d}',
            'num_diagnosis_codes': len(patient_diags),
            'diagnosis_codes': ';'.join(sorted(list(patient_diags))) if patient_diags else '',
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'generation_method': 'PromptEHR'
        })
    
    patient_df = pd.DataFrame(patient_data)
    patient_csv_path = output_dir / "synthetic_patients_summary.csv"
    patient_df.to_csv(patient_csv_path, index=False)
    print(f"Patient summary saved to {patient_csv_path}")
    
    code_freq_data = []
    for code in sorted_codes:
        freq = np.sum(binary_matrix[:, code_to_idx[code]])
        code_freq_data.append({
            'diagnosis_code': code,
            'frequency': int(freq),
            'prevalence': freq / n_patients,
            'code_type': 'ICD9_diagnosis'
        })
    
    freq_df = pd.DataFrame(code_freq_data)
    freq_csv_path = output_dir / "synthetic_code_frequencies.csv"
    freq_df.to_csv(freq_csv_path, index=False)
    print(f"Code frequencies saved to {freq_csv_path}")
    
    print("Creating sparse CSV...")
    sparse_data = []
    for i in range(n_patients):
        for j in range(n_features):
            if binary_matrix[i, j] == 1:
                sparse_data.append({
                    'patient_id': f'synthetic_{i:06d}',
                    'diagnosis_code': sorted_codes[j],
                    'present': 1
                })
    
    if sparse_data:
        sparse_df = pd.DataFrame(sparse_data)
        sparse_csv_path = output_dir / "synthetic_patient_diagnoses_sparse.csv"
        sparse_df.to_csv(sparse_csv_path, index=False)
        print(f"Sparse matrix saved to {sparse_csv_path}")
    
    print(f"Shape: {binary_matrix.shape}")
    print(f"Sparsity: {stats['sparsity']:.3f}")
    print(f"Avg codes per patient: {stats['avg_codes_per_patient']:.1f}")
    
    return binary_matrix

def main():
    parser = argparse.ArgumentParser(description="PromptEHR MIMIC-III Synthetic Data Generation")
    
    parser.add_argument("--mode", type=str, choices=['train_and_generate', 'generate_only', 'preprocess_only'], 
                       default='train_and_generate', help="Pipeline mode")
    
    parser.add_argument("--mimic_root", type=str, default="./data_files", help="MIMIC-III root directory")
    parser.add_argument("--output_dir", type=str, default="./promptehr_synthetic", help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to trained model (for generate_only mode)")
    parser.add_argument("--preprocess_dir", type=str, help="Path to preprocessed data")
    
    parser.add_argument("--max_visits", type=int, default=20, help="Max visits per patient")
    parser.add_argument("--min_visits", type=int, default=2, help="Min visits per patient")
    parser.add_argument("--include_procedures", action="store_true", help="Include procedure codes")
    parser.add_argument("--include_medications", action="store_true", help="Include medication codes")
    parser.add_argument("--code_vocab_threshold", type=int, default=5, help="Minimum code frequency")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training data ratio")
    
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    
    parser.add_argument("--n_synthetic", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--n_per_sample", type=int, default=1, help="Samples per seed patient")
    parser.add_argument("--n_seed_samples", type=int, default=100, help="Number of seed samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")
    
    args = parser.parse_args()
    
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    
    if args.mode == 'preprocess_only':
        preprocess_dir = preprocess_mimic3_data(args.mimic_root, args.output_dir, args)
        if preprocess_dir:
            print(f"Preprocessing completed")
            print(f"Data saved to: {preprocess_dir}")
    
    elif args.mode == 'generate_only':
        if not args.model_path:
            print("--model_path required for generate_only mode")
            return
        
        synthetic_dir = generate_synthetic_data(args.model_path, args.output_dir, args)
        if synthetic_dir:
            print(f"Generation completed")
            print(f"Data saved to: {synthetic_dir}")
    
    elif args.mode == 'train_and_generate':
        
        if args.preprocess_dir:
            preprocess_dir = args.preprocess_dir
            print(f"Using existing preprocessed data: {preprocess_dir}")
        else:
            preprocess_dir = preprocess_mimic3_data(args.mimic_root, 
                                                   str(Path(args.output_dir) / "preprocessed"), args)
            if not preprocess_dir:
                print("Preprocessing failed")
                return
        
        model_path = train_promptehr_model(preprocess_dir, 
                                         str(Path(args.output_dir) / "model"), args)
        if not model_path:
            print("Training failed")
            return
        
        synthetic_dir = generate_synthetic_data(model_path, 
                                              str(Path(args.output_dir) / "synthetic"), args)
        if synthetic_dir:
            print(f"Pipeline completed")
            print(f"Data saved to: {synthetic_dir}")

if __name__ == "__main__":
    main()