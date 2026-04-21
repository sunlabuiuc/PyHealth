"""Synthetic ablation study for RNNAttention model.

Contributor: Abdullah Rehman, Benjamin Yang, Leny Pan
NetID: arehman3;bhyang2;lenypan2
Paper: Predicting utilization of healthcare services from individual disease 
    trajectories using RNNs with multi-headed attention
Paper link: https://github.com/ykumards/pummel-regression

Description: Lightweight synthetic ablation study for the RNNAttention model
    in PyHealth. Compares RNNAttention against baseline models (standard RNN
    and Logistic Regression) with varying hyperparameters.

This example demonstrates:
- RNNAttention model with multi-headed attention for sequence prediction
- Comparison against built-in RNN and LogisticRegression baselines
- Hyperparameter ablation (embedding dimension, num attention heads)
- Fast evaluation on synthetic data suitable for PR inclusion
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LogisticRegression, RNN, RNN_attention


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute AUROC score."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    # Ensure binary classification
    if len(np.unique(y_true)) < 2:
        return {"auroc": 0.0}

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.0

    return {"auroc": auroc}


def create_synthetic_dataset(
    num_patients: int = 50,
    max_visits: int = 5,
    num_conditions: int = 20,
    num_procedures: int = 15,
    dataset_name: str = "synthetic_readmission",
) -> tuple:
    """Create a synthetic EHR dataset with visit sequences.

    Args:
        num_patients: number of synthetic patients
        max_visits: maximum number of visits per patient
        num_conditions: size of condition vocabulary
        num_procedures: size of procedure vocabulary
        dataset_name: name for the dataset

    Returns:
        train_dataset, test_dataset, vocab_size
    """
    samples = []
    np.random.seed(42)

    for patient_id in range(num_patients):
        num_visits = np.random.randint(2, max_visits + 1)

        # Create visit sequences
        conditions_seq = []
        procedures_seq = []

        for visit_idx in range(num_visits):
            # Random number of conditions and procedures per visit
            num_conds_visit = np.random.randint(1, 5)
            num_procs_visit = np.random.randint(0, 4)

            conditions_visit = [
                f"cond-{np.random.randint(0, num_conditions)}"
                for _ in range(num_conds_visit)
            ]
            procedures_visit = [
                f"proc-{np.random.randint(0, num_procedures)}"
                for _ in range(num_procs_visit)
            ]

            conditions_seq.append(conditions_visit)
            procedures_seq.append(procedures_visit)

        # Create label (binary): whether patient has readmission/high resource use
        label = np.random.randint(0, 2)

        samples.append(
            {
                "patient_id": f"patient-{patient_id}",
                "visit_id": f"visit-{patient_id}",
                "conditions": conditions_seq,
                "procedures": procedures_seq,
                "label": label,
            }
        )

    # Split into train/test
    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    test_samples = samples[split:]

    # Create datasets
    train_dataset = create_sample_dataset(
        samples=train_samples,
        input_schema={
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        },
        output_schema={"label": "binary"},
        dataset_name=f"{dataset_name}_train",
    )

    test_dataset = create_sample_dataset(
        samples=test_samples,
        input_schema={
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        },
        output_schema={"label": "binary"},
        dataset_name=f"{dataset_name}_test",
    )

    return train_dataset, test_dataset


def train_model(
    model,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
) -> float:
    """Train model for a few epochs and return final loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            output = model(**batch)
            loss = output["loss"]

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        final_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch + 1}: loss = {final_loss:.5f}")

    return final_loss


def evaluate_model(
    model,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            output = model(**batch)
            y_true = output["y_true"].detach().cpu().numpy()
            y_prob = output["y_prob"].detach().cpu().numpy()

            y_true_list.extend(y_true.flatten())
            y_prob_list.extend(y_prob.flatten())

    return evaluate(np.array(y_true_list), np.array(y_prob_list))


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model variants."""
    name: str
    embedding_dim: int
    attention_heads: int = None  # Only for RNNAttention


def run_ablation(
    config: ModelConfig,
    train_dataset,
    test_dataset,
    model_class,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float | str]:
    """Train and evaluate a single model configuration."""
    print(f"\n{'='*80}")
    print(f"Config: {config.name}")
    print(f"  Embedding Dimension: {config.embedding_dim}")
    if config.attention_heads is not None:
        print(f"  Attention Heads: {config.attention_heads}")
    print(f"{'='*80}")

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Initialize model
    try:
        if model_class == RNN_attention:
            model = model_class(
                dataset=train_dataset,
                embedding_dim=config.embedding_dim,
                h=config.attention_heads,
            ).to(device)
        else:
            model = model_class(
                dataset=train_dataset,
                embedding_dim=config.embedding_dim,
            ).to(device)
    except Exception as e:
        print(f"  Failed to initialize model: {e}")
        return {
            "config": config.name,
            "model_type": model_class.__name__,
            "embedding_dim": config.embedding_dim,
            "train_loss": -1.0,
            "test_auroc": -1.0,
            "status": "failed",
        }

    # Train
    print("  Training...")
    try:
        train_loss = train_model(
            model, train_loader, device, epochs=args.epochs)
    except Exception as e:
        print(f"  Training failed: {e}")
        return {
            "config": config.name,
            "model_type": model_class.__name__,
            "embedding_dim": config.embedding_dim,
            "train_loss": -1.0,
            "test_auroc": -1.0,
            "status": "failed",
        }

    # Evaluate
    print("  Evaluating...")
    try:
        metrics = evaluate_model(model, test_loader, device)
        test_auroc = metrics["auroc"]
    except Exception as e:
        print(f"  Evaluation failed: {e}")
        test_auroc = -1.0

    return {
        "config": config.name,
        "model_type": model_class.__name__,
        "embedding_dim": config.embedding_dim,
        "attention_heads": config.attention_heads,
        "train_loss": train_loss,
        "test_auroc": test_auroc,
        "status": "ok",
    }


def print_results_table(results: List[Dict]) -> None:
    """Print comparison table of results."""
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    print(
        "Model Type        | Config              | Emb Dim | Attn Heads | "
        "Train Loss | Test AUROC | Status"
    )
    print("-" * 100)

    for result in results:
        model_type = result["model_type"]
        config = result["config"]
        emb_dim = result["embedding_dim"]
        attn_heads = result.get("attention_heads", "-")
        train_loss = result["train_loss"]
        test_auroc = result["test_auroc"]
        status = result["status"]

        print(
            f"{model_type:<17} | {config:<19} | {emb_dim:<7} | "
            f"{str(attn_heads):<10} | {train_loss:>9.5f} | {test_auroc:>10.4f} | {status:<6}"
        )

    print("="*100)

    # Find best performer
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        best = max(ok_results, key=lambda x: x["test_auroc"])
        print(f"\nBest Configuration: {best['config']} ({best['model_type']})")
        print(f"  Test AUROC: {best['test_auroc']:.4f}")

    print("\nObservations:")
    print("- This is a lightweight synthetic ablation study for PR inclusion.")
    print("- Real-world performance may differ significantly with real EHR data.")
    print("- The paper demonstrates that RNNAttention outperforms baseline models")
    print("  on real MIMIC-III data with healthcare resource allocation tasks.")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for RNNAttention model"
    )
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,
                        default=16, help="Batch size")
    parser.add_argument("--device", type=str,
                        default="cpu", help="Device to use")
    parser.add_argument(
        "--num-patients", type=int, default=50, help="Number of synthetic patients"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("="*80)
    print("RNNAttention Ablation Study on Synthetic Readmission Task")
    print("="*80)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Synthetic patients: {args.num_patients}")

    # Create datasets
    print("\n" + "="*80)
    print("Creating synthetic datasets...")
    print("="*80)
    train_dataset, test_dataset = create_synthetic_dataset(
        num_patients=args.num_patients,
        max_visits=5,
        num_conditions=20,
        num_procedures=15,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Define configurations for ablation
    configs = [
        # RNNAttention variants
        ModelConfig(
            name="RNNAttn-emb64-h2",
            embedding_dim=64,
            attention_heads=2,
        ),
        ModelConfig(
            name="RNNAttn-emb128-h4",
            embedding_dim=128,
            attention_heads=4,
        ),
        ModelConfig(
            name="RNNAttn-emb256-h8",
            embedding_dim=256,
            attention_heads=8,
        ),
        # Standard RNN baseline
        ModelConfig(
            name="RNN-emb64",
            embedding_dim=64,
            attention_heads=None,
        ),
        ModelConfig(
            name="RNN-emb128",
            embedding_dim=128,
            attention_heads=None,
        ),
    ]

    # Run ablation studies
    results = []

    # Test RNNAttention configs
    print("\n" + "="*80)
    print("Testing RNNAttention Configurations")
    print("="*80)
    for config in configs[:3]:
        result = run_ablation(
            config=config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_class=RNN_attention,
            device=device,
            args=args,
        )
        results.append(result)

    # Test standard RNN baseline
    print("\n" + "="*80)
    print("Testing Standard RNN Baseline")
    print("="*80)
    for config in configs[3:]:
        result = run_ablation(
            config=config,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_class=RNN,
            device=device,
            args=args,
        )
        results.append(result)

    # Test LogisticRegression baseline
    print("\n" + "="*80)
    print("Testing Logistic Regression Baseline")
    print("="*80)
    lr_config = ModelConfig(
        name="LogReg-emb128",
        embedding_dim=128,
        attention_heads=None,
    )
    result = run_ablation(
        config=lr_config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_class=LogisticRegression,
        device=device,
        args=args,
    )
    results.append(result)

    # Print results
    print_results_table(results)


if __name__ == "__main__":
    main()
