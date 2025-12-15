"""
Example of using MoleRec for drug recommendation on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the DrugRecommendationMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Training a MoleRec model for multi-label classification
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

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.datasets.utils import load_processors, save_processors
from pyhealth.models import MoleRec
from pyhealth.tasks import DrugRecommendationMIMIC4
from pyhealth.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoleRec for Drug Recommendation on MIMIC-IV"
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
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.7,
        help="Dropout rate (default: 0.7)",
    )
    parser.add_argument(
        "--num_gnn_layers",
        type=int,
        default=4,
        help="Number of GNN layers (default: 4)",
    )
    parser.add_argument(
        "--target_ddi",
        type=float,
        default=0.08,
        help="Target DDI rate (default: 0.08)",
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

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # STEP 4: Initialize MoleRec model
    # MoleRec uses molecular substructure representations for drug modeling
    # and requires rdkit for molecular processing
    print("\nInitializing MoleRec model...")
    model = MoleRec(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        target_ddi=args.target_ddi,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    print(f"Feature keys: {model.feature_keys}")
    print(f"Label keys: {model.label_keys}")

    # STEP 5: Train the model
    trainer = Trainer(
        model=model,
        device=args.device,
        metrics=["f1_micro", "pr_auc_samples", "jaccard_samples", "ddi"],
        output_path=str(run_root),
        exp_name=exp_name,
    )

    print(f"\nStarting training on {args.device}...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="f1_micro",
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
