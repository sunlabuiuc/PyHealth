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
    - Schema: conditions, procedures, drugs (nested_sequence) -> mortality (binary)
"""

import argparse

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
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
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
    )

    # STEP 2: Apply mortality prediction task
    print("Applying MortalityPredictionMIMIC4 task...")
    sample_dataset = base_dataset.set_task(
        MortalityPredictionMIMIC4(),
        num_workers=4,
        cache_dir="../../../caches/mp_cache",
    )

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

    # STEP 4: Initialize AdaCare model
    print("\nInitializing AdaCare model...")
    model = AdaCare(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    print(f"Feature keys: {model.feature_keys}")
    print(f"Label key: {model.label_key}")

    # STEP 5: Train the model
    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )

    print(f"\nStarting training on {args.device}...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="pr_auc",
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
