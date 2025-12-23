"""
Example of using MoleRec for drug recommendation on MIMIC-IV with Optuna tuning.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the DrugRecommendationMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Tuning a MoleRec model for multi-label classification using Optuna
5. Evaluating with PR-AUC, F1, and Jaccard metrics

Model: MoleRec (Molecular Recommendation)
    Paper: Nianzu Yang et al. MoleRec: Combinatorial Drug Recommendation with
    Substructure-Aware Molecular Representation Learning. WWW 2023.

        - Uses graph neural networks to learn molecular substructure
            representations
    - Models drug-drug interactions at the molecular level
    - Balances recommendation accuracy with DDI rate control

    IMPORTANT CONSTRAINTS:
    - Only accepts ATC level 3 medication codes
    - Requires the 'rdkit' package for molecular processing
    - Install with: pip install rdkit

Task: Drug Recommendation
        - Input: Patient visit history with conditions, procedures, and
            drug history
    - Output: Multi-label prediction of drugs to recommend
        - Schema: conditions, procedures, drugs_hist (nested_sequence)
            -> drugs (multilabel)
"""

import argparse
from datetime import datetime
from pathlib import Path

import optuna
import numpy as np

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.datasets.utils import load_processors, save_processors
from pyhealth.models import MoleRec
from pyhealth.tasks import DrugRecommendationMIMIC4
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoleRec for Drug Recommendation on MIMIC-IV with Optuna"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:4",
        help="Device to use for training (default: cuda:4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
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
        "--n_trials",
        type=int,
        default=30,
        help="Number of optuna trials (default: 30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    task_name = "dr"
    model_name = "molerec"
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

    # STEP 2: Apply drug recommendation task
    # Note: DrugRecommendationMIMIC4 task converts medications to ATC level 3
    # codes
    # which is required by MoleRec for molecular representation learning
    print("Applying DrugRecommendationMIMIC4 task...")

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
        DrugRecommendationMIMIC4(),
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
    print(f"  Conditions (history): {len(sample['conditions'])} visits")
    print(f"  Procedures (history): {len(sample['procedures'])} visits")
    print(f"  Drugs history: {len(sample['drugs_hist'])} visits")
    print(f"  Target drugs: {len(sample['drugs'])} drugs")

    # STEP 3: Split dataset
    print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # STEP 4: Run optuna trials
    print(f"\nRunning {args.n_trials} independent optuna trials...")

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256, step=32)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 6)
        target_ddi = trial.suggest_float("target_ddi", 0.01, 0.15)

        print("=" * 60)
        print(f"Trial {trial.number} hyperparameters:")
        print(f"  lr: {lr}")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  dropout: {dropout}")
        print(f"  num_gnn_layers: {num_gnn_layers}")
        print(f"  target_ddi: {target_ddi}")

        run_exp_name = f"{exp_name}_trial{trial.number + 1}"

        # Create dataloaders
        train_loader = get_dataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = get_dataloader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
        test_loader = get_dataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Initialize model
        print("\nInitializing MoleRec model...")
        model = MoleRec(
            dataset=sample_dataset,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            target_ddi=target_ddi,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {num_params:,} parameters")

        # Train the model
        trainer = Trainer(
            model=model,
            device=args.device,
            metrics=["f1_micro", "pr_auc_samples", "jaccard_samples", "ddi"],
            output_path=str(run_root),
            exp_name=run_exp_name,
        )

        print(f"\nStarting training on {args.device}...")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            monitor="f1_micro",
            patience=args.patience,
            optimizer_params={"lr": lr},
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = trainer.evaluate(test_loader)

        print(f"\nTrial {trial.number} Test Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        loss = results["loss"]
        roc_auc = results["roc_auc"]
        pr_auc = results["pr_auc"]
            
        return float(loss), float(roc_auc), float(pr_auc)

    study = optuna.create_study(
        storage=f"sqlite:///{run_root}/optuna.sqlite3",
        directions=["minimize", "maximize", "maximize"],
    )
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
