"""
Example of using RNN for length of stay prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the LengthOfStayPredictionMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Training an RNN model for multi-class classification
5. Evaluating with accuracy and F1 metrics (weighted, macro, micro)

Model: RNN (Recurrent Neural Network)
    - Uses GRU/LSTM cells to capture temporal dependencies in patient visits
    - Processes sequences of medical codes (diagnoses, procedures, drugs)
    - Outputs multi-class length of stay prediction

Task: Length of Stay Prediction
    - Input: Patient visit history with conditions, procedures, and drugs
    - Output: Multi-class label for length of stay category (10 categories)
    - Categories: <1 day, 1-2 days, 2-3 days, ..., >7 days
    - Schema: conditions, procedures, drugs (nested_sequence) -> los (multiclass)
"""

import argparse

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="RNN for Length of Stay Prediction on MIMIC-IV"
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

    # STEP 2: Apply length of stay prediction task
    print("Applying LengthOfStayPredictionMIMIC4 task...")
    sample_dataset = base_dataset.set_task(
        LengthOfStayPredictionMIMIC4(),
        num_workers=4,
        cache_dir="../../../caches/los_cache",
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
    print(f"  Length of Stay label: {sample['los']}")

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
    print(f"Label key: {model.label_key}")

    # STEP 5: Train the model
    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
    )

    print(f"\nStarting training on {args.device}...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="accuracy",
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
