#!/usr/bin/env python3
"""
CorGAN Example Script

This script demonstrates how to use CorGAN for generating synthetic medical data.
It follows the synthEHRella pipeline: preprocessing -> training -> generation -> postprocessing.

Usage:
    python corgan_example.py --data_path /path/to/mimic3 --output_dir ./corgan_output
"""

import os
import argparse
import numpy as np
import pickle
from datetime import datetime
from typing import Union, Dict, Any

from pyhealth.datasets.corgan_dataset import CorGANDataset
from pyhealth.datasets import BaseDataset
from pyhealth.models.generators.corgan import CorGAN


class DummyDataset(BaseDataset):
    """Dummy dataset for loading existing processed data"""
    def __init__(self, types: Dict[str, int]):
        super().__init__(dataset_name="dummy", root="", tables=["dummy"])
        self.patients = {"dummy": {"conditions": [["dummy"]]}}
        self.types = types
    
    def get_reverse_code_mapping(self) -> Dict[int, str]:
        """Get reverse code mapping"""
        return {v: k for k, v in self.types.items()}
    
    def load_data(self):
        """Override load_data to avoid loading actual files"""
        return None


def preprocess_data(data_path: str, output_dir: str):
    """
    Preprocess MIMIC-III data into binary matrix format
    
    Args:
        data_path: Path to MIMIC-III CSV files
        output_dir: Directory to save processed data
    """
    print("=== Preprocessing Data ===")
    
    # create dataset
    dataset = CorGANDataset(
        dataset_name="mimic3",
        root=data_path,
        tables=["ADMISSIONS", "DIAGNOSES_ICD"],
        code_mapping={"DIAGNOSES_ICD": "ICD9_CODE"},
        visit_mapping={"DIAGNOSES_ICD": "HADM_ID"},
        patient_mapping={"DIAGNOSES_ICD": "SUBJECT_ID"},
        timestamp_mapping={"DIAGNOSES_ICD": "CHARTTIME"}
    )
    
    # save processed data
    dataset.save_processed_data(output_dir, matrix_type="binary")
    
    # print statistics
    stats = dataset.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return dataset


def train_corgan(dataset: Union[CorGANDataset, DummyDataset], output_dir: str, model_params: dict):
    """
    Train CorGAN model
    
    Args:
        dataset: Preprocessed dataset
        output_dir: Directory to save model
        model_params: Model hyperparameters
    """
    print("=== Training CorGAN ===")
    
    # create model
    model = CorGAN(
        dataset=dataset,
        feature_keys=["conditions"],  # dummy key, not used in this implementation
        label_key="label",  # dummy key, not used in this implementation
        **model_params
    )
    
    # train model
    model.fit()
    
    # save model
    model_path = os.path.join(output_dir, "corgan_model.pth")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return model


def generate_synthetic_data(model: CorGAN, output_dir: str, n_samples: int):
    """
    Generate synthetic data using trained CorGAN
    
    Args:
        model: Trained CorGAN model
        output_dir: Directory to save synthetic data
        n_samples: Number of synthetic samples to generate
    """
    print("=== Generating Synthetic Data ===")
    
    # generate synthetic data
    synthetic_data = model.generate(n_samples=n_samples)
    
    # save synthetic data
    synthetic_path = os.path.join(output_dir, f"synthetic-{n_samples}.npy")
    np.save(synthetic_path, synthetic_data.numpy())
    print(f"Synthetic data saved to {synthetic_path}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    return synthetic_data


def postprocess_data(synthetic_data: np.ndarray, dataset: Union[CorGANDataset, DummyDataset], output_dir: str):
    """
    Postprocess synthetic data (convert back to medical codes)
    
    Args:
        synthetic_data: Generated synthetic data
        dataset: Original dataset for code mapping
        output_dir: Directory to save postprocessed data
    """
    print("=== Postprocessing Data ===")
    
    # get code mapping
    reverse_mapping = dataset.get_reverse_code_mapping()
    
    # convert binary matrix back to medical codes
    postprocessed_data = []
    
    for i in range(synthetic_data.shape[0]):
        patient_codes = []
        for j in range(synthetic_data.shape[1]):
            if synthetic_data[i, j] == 1.0:
                if j in reverse_mapping:
                    patient_codes.append(reverse_mapping[j])
        
        postprocessed_data.append(patient_codes)
    
    # save postprocessed data
    postprocessed_path = os.path.join(output_dir, "synthetic_postprocessed.pkl")
    with open(postprocessed_path, "wb") as f:
        pickle.dump(postprocessed_data, f)
    
    print(f"Postprocessed data saved to {postprocessed_path}")
    print(f"Number of synthetic patients: {len(postprocessed_data)}")
    
    # print some statistics
    code_counts = {}
    for patient_codes in postprocessed_data:
        for code in patient_codes:
            code_counts[code] = code_counts.get(code, 0) + 1
    
    print(f"Number of unique codes in synthetic data: {len(code_counts)}")
    print(f"Average codes per patient: {np.mean([len(codes) for codes in postprocessed_data]):.2f}")
    
    return postprocessed_data


def evaluate_synthetic_data(real_data: np.ndarray, synthetic_data: np.ndarray):
    """
    Basic evaluation of synthetic data quality
    
    Args:
        real_data: Real data matrix
        synthetic_data: Synthetic data matrix
    """
    print("=== Evaluating Synthetic Data ===")
    
    # calculate basic statistics
    real_sparsity = 1.0 - np.mean(real_data)
    synthetic_sparsity = 1.0 - np.mean(synthetic_data)
    
    real_code_prevalence = np.mean(real_data, axis=0)
    synthetic_code_prevalence = np.mean(synthetic_data, axis=0)
    
    # correlation between real and synthetic prevalence
    correlation = np.corrcoef(real_code_prevalence, synthetic_code_prevalence)[0, 1]
    
    print(f"Real data sparsity: {real_sparsity:.4f}")
    print(f"Synthetic data sparsity: {synthetic_sparsity:.4f}")
    print(f"Prevalence correlation: {correlation:.4f}")
    
    # calculate L1 distance between prevalence distributions
    l1_distance = np.mean(np.abs(real_code_prevalence - synthetic_code_prevalence))
    print(f"L1 distance: {l1_distance:.4f}")
    
    return {
        "real_sparsity": real_sparsity,
        "synthetic_sparsity": synthetic_sparsity,
        "prevalence_correlation": correlation,
        "l1_distance": l1_distance
    }


def main():
    parser = argparse.ArgumentParser(description="CorGAN Example Script")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to MIMIC-III CSV files")
    parser.add_argument("--output_dir", type=str, default="./corgan_output",
                       help="Output directory for results")
    parser.add_argument("--n_samples", type=int, default=1000,
                       help="Number of synthetic samples to generate")
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Training batch size")
    parser.add_argument("--n_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--latent_dim", type=int, default=128,
                       help="Latent dimension")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip preprocessing if data already exists")
    
    args = parser.parse_args()
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # set model parameters
    model_params = {
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "n_epochs_pretrain": 1,
    }
    
    # preprocessing
    if not args.skip_preprocessing:
        dataset = preprocess_data(args.data_path, args.output_dir)
    else:
        # load existing processed data
        print("Loading existing processed data...")
        with open(os.path.join(args.output_dir, "processed.types"), "rb") as f:
            types = pickle.load(f)
        
        dataset = DummyDataset(types)
    
    # training
    model = train_corgan(dataset, args.output_dir, model_params)
    
    # generation
    synthetic_data = generate_synthetic_data(model, args.output_dir, args.n_samples)
    
    # postprocessing
    if not args.skip_preprocessing:
        postprocessed_data = postprocess_data(synthetic_data, dataset, args.output_dir)
    
    # evaluation
    if not args.skip_preprocessing:
        real_data = np.load(os.path.join(args.output_dir, "processed_matrix.npy"))
        evaluation_results = evaluate_synthetic_data(real_data, synthetic_data)
        
        # save evaluation results
        eval_path = os.path.join(args.output_dir, "evaluation_results.txt")
        with open(eval_path, "w") as f:
            f.write("CorGAN Evaluation Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Real data sparsity: {evaluation_results['real_sparsity']:.4f}\n")
            f.write(f"Synthetic data sparsity: {evaluation_results['synthetic_sparsity']:.4f}\n")
            f.write(f"Prevalence correlation: {evaluation_results['prevalence_correlation']:.4f}\n")
            f.write(f"L1 distance: {evaluation_results['l1_distance']:.4f}\n")
        
        print(f"Evaluation results saved to {eval_path}")
    
    print("=== CorGAN Pipeline Complete ===")
    print(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 