"""
Example of using AdaCare for mortality prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the MortalityPredictionMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Training an AdaCare model for binary classification
5. Evaluating with PR-AUC, ROC-AUC, accuracy, and F1 metrics

Model: AdaCare (Adaptive Feature Calibration for Healthcare)
    Paper: Liantao Ma et al. AdaCare: Explainable Clinical Health Status
    Representation Learning via Scale-Adaptive Feature Extraction and
    Recalibration. AAAI 2020.

    - Uses dilated convolutions to capture multi-scale temporal patterns
    - Applies squeeze-and-excitation blocks for feature recalibration
    - Provides interpretable attention weights for clinical features

Task: Mortality Prediction
    - Input: Patient visit history with conditions, procedures, and drugs
    - Output: Binary label indicating in-hospital mortality
        - Schema: conditions, procedures, drugs (nested_sequence)
            -> mortality (binary)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.datasets.utils import load_processors, save_processors
from pyhealth.models import AdaCare
from pyhealth.tasks import MortalityPredictionMIMIC4
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="AdaCare for Mortality Prediction on MIMIC-IV"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:4",
        help="Device to use for training (default: cuda:4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--root_output",
        type=str,
        default="/shared/eng/pyhealth_agent",
        help="Root directory for runs/processors/cache",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Dev mode: run quick smoke test (1 epoch, small subset, cpu)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of runs for computing mean and std (default: 3)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    task_name = "mp"
    model_name = "adacare"
    root = Path(args.root_output)
    run_root = root / "runs" / task_name / model_name
    # Use task-level cache and processors (shared across all models for this task)
    processor_dir = root / "processors" / task_name
    cache_dir = root / "cache" / task_name

    # Use separate cache for dev mode
    if args.dev:
        cache_dir = root / "cache_dev" / task_name
        processor_dir = root / "processors_dev" / task_name

    exp_name = "optuna_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # Dev mode overrides for quick smoke testing
    if args.dev:
        print("[DEV MODE] Running quick smoke test...")
        args.epochs = 1
        args.device = "cpu"

    # STEP 1: Load MIMIC-IV base dataset
    print("Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
        dev=args.dev,
    )

    # STEP 2: Apply mortality prediction task
    print("Applying MortalityPredictionMIMIC4 task...")

    input_processors = None
    output_processors = None
    if (processor_dir / "input_processors.pkl").exists() and (
        processor_dir / "output_processors.pkl"
    ).exists():
        print(f"Loading cached processors from: {processor_dir}")
        input_processors, output_processors = load_processors(str(processor_dir))
    else:
        print(f"No cached processors found; will save to: {processor_dir}")

    sample_dataset = base_dataset.set_task(
        MortalityPredictionMIMIC4(),
        num_workers=4,
        cache_dir=str(cache_dir),
        input_processors=input_processors,
        output_processors=output_processors,
    )

    if input_processors is None and output_processors is None:
        save_processors(sample_dataset, str(processor_dir))

    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    # Inspect a sample
    sample = sample_dataset.samples[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Visit ID: {sample['visit_id']}")
    print(f"  Conditions: {len(sample['conditions'])} visits")
    print(f"  Procedures: {len(sample['procedures'])} visits")
    print(f"  Drugs: {len(sample['drugs'])} visits")
    print(f"  Mortality label: {sample['mortality']}")

    # STEP 3: Split dataset
    print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # STEP 4-6: Run training and evaluation multiple times
    print(f"\nRunning {args.n_runs} independent training runs...")
    all_results = []
    
    def objective(trial: optuna.Trial) -> float:
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])
        compression_ratio = trial.suggest_categorical("compression_ratio", [2, 4, 8])
        
        print("=" * 60)
        print(f"Trial {trial.number} hyperparameters:")
        print(f"  batch_size: {batch_size}")
        print(f"  lr: {lr}")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  dropout: {dropout}")
        print(f"  kernel_size: {kernel_size}")
        print(f"  compression_ratio: {compression_ratio}")
        
        run_exp_name = f"{exp_name}_trial{trial.number + 1}"
        
        # Create dataloaders
        train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print("\nInitializing AdaCare model...")
        model = AdaCare(
            dataset=sample_dataset,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            r_v=compression_ratio,
            r_c=compression_ratio,
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {num_params:,} parameters")
        print(f"Feature keys: {model.feature_keys}")
        print(f"Label keys: {model.label_keys}")
        
        trainer = Trainer(
            model=model,
            device=args.device,
            metrics=["roc_auc", "f1", "pr_auc", "accuracy"],
            output_path=str(run_root),
            exp_name=run_exp_name,
        )

        print(f"\nStarting training on {args.device}...")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            monitor="roc_auc",
            patience=args.patience,
            optimizer_params={"lr": args.lr},
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = trainer.evaluate(test_loader)
        all_results.append(results)

        print(f"\nTrial {trial.number} Test Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
            
        return float(results["loss"])

    study = optuna.create_study(storage=f"sqlite:///{run_root}/optuna.sqlite3")
    study.optimize(objective, n_trials=30)

if __name__ == "__main__":
    main()
