"""
Example of using RNN for mortality prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the MortalityPredictionMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Training an RNN model for binary classification
5. Evaluating with PR-AUC, ROC-AUC, accuracy, and F1 metrics

Model: RNN (Recurrent Neural Network)
    - Uses GRU/LSTM cells to capture temporal dependencies in patient visits
    - Processes sequences of medical codes (diagnoses, procedures, drugs)
    - Outputs binary mortality prediction

Task: Mortality Prediction
    - Input: Patient visit history with conditions, procedures, and drugs
    - Output: Binary label indicating in-hospital mortality
        - Schema: conditions, procedures, drugs (nested_sequence)
            -> mortality (binary)
"""

import argparse
from datetime import datetime
from pathlib import Path

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.datasets.utils import load_processors, save_processors
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionMIMIC4
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="RNN for Mortality Prediction on MIMIC-IV"
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
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
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
        "--embedding_dim",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Dev mode: run quick smoke test (1 epoch, small subset, cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    task_name = "mp"
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

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")

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

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # STEP 4: Initialize RNN model
    print("\nInitializing RNN model...")
    model = RNN(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    print(f"Feature keys: {model.feature_keys}")
    print(f"Label keys: {model.label_keys}")

    # STEP 5: Train the model
    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["roc_auc", "f1", "pr_auc", "accuracy"],
        output_path=str(run_root),
        exp_name=exp_name,
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

    # STEP 6: Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
