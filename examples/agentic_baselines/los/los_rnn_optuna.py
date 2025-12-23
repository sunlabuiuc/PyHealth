"""
Example of using RNN for length of stay prediction on MIMIC-IV with Optuna tuning.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the LengthOfStayPredictionMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Tuning an RNN model for multi-class classification using Optuna
5. Evaluating with accuracy and F1 metrics

Model: RNN (Recurrent Neural Network)
    - Uses GRU/LSTM cells to capture temporal dependencies in patient visits
    - Processes sequences of medical codes (diagnoses, procedures, drugs)
    - Outputs multi-class length of stay prediction

Task: Length of Stay Prediction
    - Input: Patient visit history with conditions, procedures, and drugs
    - Output: Multi-class label for length of stay category (10 categories)
    - Categories: <1 day, 1-2 days, 2-3 days, ..., >7 days
        - Schema: conditions, procedures, drugs (nested_sequence)
            -> los (multiclass)
"""

import argparse
from datetime import datetime
from pathlib import Path

import optuna
import numpy as np

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.datasets.utils import load_processors, save_processors
from pyhealth.models import RNN
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="RNN for Length of Stay Prediction on MIMIC-IV with Optuna"
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
        "--n_trials",
        type=int,
        default=20,
        help="Number of optuna trials (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    task_name = "los"
    model_name = "rnn"
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

    # STEP 2: Apply length of stay prediction task
    print("Applying LengthOfStayPredictionMIMIC4 task...")

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
        LengthOfStayPredictionMIMIC4(),
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
    print(f"  Length of Stay label: {sample['los']}")

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

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        embedding_dim = trial.suggest_int("embedding_dim", 32, 256, step=32)
        hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        rnn_type = trial.suggest_categorical("rnn_type", ["RNN", "LSTM", "GRU"])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        print("=" * 60)
        print(f"Trial {trial.number} hyperparameters:")
        print(f"  lr: {lr}")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  dropout: {dropout}")
        print(f"  rnn_type: {rnn_type}")
        print(f"  num_layers: {num_layers}")
        print(f"  bidirectional: {bidirectional}")
        print(f"  weight_decay: {weight_decay}")

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
        print("\nInitializing RNN model...")
        model = RNN(
            dataset=sample_dataset,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {num_params:,} parameters")

        # Train the model
        trainer = Trainer(
            model=model,
            device=args.device,
            metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
            output_path=str(run_root),
            exp_name=run_exp_name,
        )

        print(f"\nStarting training on {args.device}...")
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            monitor="f1_macro",
            patience=args.patience,
            weight_decay=weight_decay,
            optimizer_params={"lr": lr},
        )

        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = trainer.evaluate(test_loader)

        print(f"\nTrial {trial.number} Test Results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")

        loss = results["loss"]
        f1_macro = results["f1_macro"]

        return float(loss), float(f1_macro)

    study = optuna.create_study(
        storage=f"sqlite:///{run_root}/optuna.sqlite3",
        directions=["minimize", "maximize"],
    )
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
