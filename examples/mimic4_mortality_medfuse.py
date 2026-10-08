"""MedFuse Ablation Study for In-Hospital Mortality Prediction.

This script demonstrates MedFuse with varying hyperparameters on synthetic
mortality-like data to validate model behavior quickly.

Paper:
    Hayat et al. "MedFuse: Multi-modal fusion with clinical time-series data
    and chest X-ray images." MLHC 2022.

Ablations:
    1. EHR hidden dim: 64, 128, 256, 512
    2. Fusion hidden dim: 128, 256, 512
    3. Dropout: 0.0, 0.3, 0.5, 0.7
    4. Learning rate: 1e-5, 1e-4, 1e-3
    5. Modality: EHR-only vs EHR+CXR

The script is intentionally lightweight by default:
    - Synthetic data only
    - Small dataset
    - Few epochs
"""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MedFuse


def set_seed(seed: int) -> None:
    """Sets all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_samples(
    num_samples: int,
    seq_len: int,
    ehr_dim: int,
    image_size: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Builds synthetic samples with weakly learnable mortality signal."""
    generator = torch.Generator().manual_seed(seed)
    samples: List[Dict[str, object]] = []

    for index in range(num_samples):
        ehr = torch.randn(seq_len, ehr_dim, generator=generator)
        cxr = torch.randn(3, image_size, image_size, generator=generator)

        score = 0.7 * float(ehr[:, 0].mean())
        score += 0.3 * float(cxr.mean())
        score += float(torch.randn(1, generator=generator)) * 0.1
        label = 1 if score > 0 else 0

        samples.append(
            {
                "patient_id": f"patient-{index}",
                "visit_id": f"visit-{index}",
                "ehr": ehr.tolist(),
                "cxr": cxr.tolist(),
                "label": label,
            }
        )

    return samples


def build_datasets(
    train_size: int,
    val_size: int,
    seq_len: int,
    ehr_dim: int,
    image_size: int,
    seed: int,
):
    """Creates synthetic train/validation datasets."""
    train_samples = make_synthetic_samples(
        num_samples=train_size,
        seq_len=seq_len,
        ehr_dim=ehr_dim,
        image_size=image_size,
        seed=seed,
    )
    val_samples = make_synthetic_samples(
        num_samples=val_size,
        seq_len=seq_len,
        ehr_dim=ehr_dim,
        image_size=image_size,
        seed=seed + 1,
    )

    input_schema = {"ehr": "tensor", "cxr": "tensor"}
    output_schema = {"label": "binary"}

    train_dataset = create_sample_dataset(
        samples=train_samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="medfuse_ablation_train",
    )
    val_dataset = create_sample_dataset(
        samples=val_samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="medfuse_ablation_val",
    )
    return train_dataset, val_dataset


def compute_binary_metrics(
    y_true: List[float],
    y_prob: List[float],
) -> Dict[str, float]:
    """Computes AUROC and AUPRC with robust fallbacks."""
    metrics = {
        "auroc": math.nan,
        "auprc": math.nan,
    }

    if len(set(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    if any(y_true):
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))

    return metrics


def train_and_evaluate(
    train_dataset,
    val_dataset,
    use_cxr: bool,
    ehr_hidden_dim: int,
    fusion_hidden_dim: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    cxr_backbone: str,
) -> Dict[str, float]:
    """Trains one MedFuse configuration and reports validation metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedFuse(
        dataset=train_dataset,
        ehr_feature_key="ehr",
        cxr_feature_key="cxr",
        cxr_mask_key="cxr_mask",
        ehr_hidden_dim=ehr_hidden_dim,
        ehr_num_layers=2,
        cxr_backbone=cxr_backbone,
        cxr_pretrained=False,
        fusion_hidden_dim=fusion_hidden_dim,
        projection_dim=ehr_hidden_dim,
        dropout=dropout,
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            model_inputs = {
                "ehr": batch["ehr"].to(device),
                "label": batch["label"].to(device),
            }
            if use_cxr:
                model_inputs["cxr"] = batch["cxr"].to(device)

            outputs = model(**model_inputs)
            outputs["loss"].backward()
            optimizer.step()

    model.eval()
    all_probs: List[float] = []
    all_true: List[float] = []

    with torch.no_grad():
        for batch in val_loader:
            model_inputs = {"ehr": batch["ehr"].to(device)}
            if use_cxr:
                model_inputs["cxr"] = batch["cxr"].to(device)

            outputs = model(**model_inputs)
            all_probs.extend(outputs["y_prob"].view(-1).cpu().tolist())
            all_true.extend(batch["label"].view(-1).cpu().tolist())

    return compute_binary_metrics(all_true, all_probs)


def best_config(rows: List[Tuple[str, Dict[str, float]]]) -> Tuple[str, float]:
    """Returns the setting name and AUROC of the best configuration."""
    best_name = rows[0][0]
    best_auroc = -1.0
    for name, metrics in rows:
        auroc = metrics["auroc"]
        if not math.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_name = name
    return best_name, best_auroc


def format_metric(value: float) -> str:
    """Formats metric values for table output."""
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def print_table(title: str, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    """Prints a compact ablation result table."""
    print(f"\n{title}")
    print("| setting | AUROC | AUPRC |")
    print("|---------|-------|-------|")
    for setting, metrics in rows:
        auroc = format_metric(metrics["auroc"])
        auprc = format_metric(metrics["auprc"])
        print(f"| {setting} | {auroc} | {auprc} |")


def run_ablation_study(args: argparse.Namespace) -> None:
    """Runs all required MedFuse ablations."""
    set_seed(args.seed)
    train_dataset, val_dataset = build_datasets(
        train_size=args.train_size,
        val_size=args.val_size,
        seq_len=args.seq_len,
        ehr_dim=args.ehr_dim,
        image_size=args.image_size,
        seed=args.seed,
    )

    base_config = {
        "use_cxr": True,
        "ehr_hidden_dim": 128,
        "fusion_hidden_dim": 256,
        "dropout": 0.3,
        "learning_rate": 1e-4,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cxr_backbone": args.cxr_backbone,
    }

    hidden_results: List[Tuple[str, Dict[str, float]]] = []
    for hidden_dim in [64, 128, 256, 512]:
        metrics = train_and_evaluate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            use_cxr=True,
            ehr_hidden_dim=hidden_dim,
            fusion_hidden_dim=base_config["fusion_hidden_dim"],
            dropout=base_config["dropout"],
            learning_rate=base_config["learning_rate"],
            epochs=base_config["epochs"],
            batch_size=base_config["batch_size"],
            cxr_backbone=base_config["cxr_backbone"],
        )
        hidden_results.append((f"ehr_hidden_dim={hidden_dim}", metrics))

    fusion_results: List[Tuple[str, Dict[str, float]]] = []
    for fusion_dim in [128, 256, 512]:
        metrics = train_and_evaluate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            use_cxr=True,
            ehr_hidden_dim=base_config["ehr_hidden_dim"],
            fusion_hidden_dim=fusion_dim,
            dropout=base_config["dropout"],
            learning_rate=base_config["learning_rate"],
            epochs=base_config["epochs"],
            batch_size=base_config["batch_size"],
            cxr_backbone=base_config["cxr_backbone"],
        )
        fusion_results.append((f"fusion_hidden_dim={fusion_dim}", metrics))

    dropout_results: List[Tuple[str, Dict[str, float]]] = []
    for dropout_value in [0.0, 0.3, 0.5, 0.7]:
        metrics = train_and_evaluate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            use_cxr=True,
            ehr_hidden_dim=base_config["ehr_hidden_dim"],
            fusion_hidden_dim=base_config["fusion_hidden_dim"],
            dropout=dropout_value,
            learning_rate=base_config["learning_rate"],
            epochs=base_config["epochs"],
            batch_size=base_config["batch_size"],
            cxr_backbone=base_config["cxr_backbone"],
        )
        dropout_results.append((f"dropout={dropout_value}", metrics))

    lr_results: List[Tuple[str, Dict[str, float]]] = []
    for lr in [1e-5, 1e-4, 1e-3]:
        metrics = train_and_evaluate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            use_cxr=True,
            ehr_hidden_dim=base_config["ehr_hidden_dim"],
            fusion_hidden_dim=base_config["fusion_hidden_dim"],
            dropout=base_config["dropout"],
            learning_rate=lr,
            epochs=base_config["epochs"],
            batch_size=base_config["batch_size"],
            cxr_backbone=base_config["cxr_backbone"],
        )
        lr_results.append((f"learning_rate={lr:.0e}", metrics))

    modality_results: List[Tuple[str, Dict[str, float]]] = []
    for use_cxr in [False, True]:
        metrics = train_and_evaluate(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            use_cxr=use_cxr,
            ehr_hidden_dim=base_config["ehr_hidden_dim"],
            fusion_hidden_dim=base_config["fusion_hidden_dim"],
            dropout=base_config["dropout"],
            learning_rate=base_config["learning_rate"],
            epochs=base_config["epochs"],
            batch_size=base_config["batch_size"],
            cxr_backbone=base_config["cxr_backbone"],
        )
        name = "EHR+CXR" if use_cxr else "EHR-only"
        modality_results.append((name, metrics))

    all_ablations = [
        ("EHR Hidden Dim Ablation", hidden_results),
        ("Fusion Hidden Dim Ablation", fusion_results),
        ("Dropout Ablation", dropout_results),
        ("Learning Rate Ablation", lr_results),
        ("Modality Ablation", modality_results),
    ]

    for title, rows in all_ablations:
        print_table(title, rows)

    print("\nFindings (auto-generated from results above):")
    for title, rows in all_ablations:
        name, auroc = best_config(rows)
        print(f"- {title}: best config = {name} (AUROC={format_metric(auroc)})")

    print(
        "\nNote: These results are on small synthetic data with minimal training "
        "(default 32 train samples, 1 epoch). They validate that the model is "
        "sensitive to hyperparameter choices but should not be interpreted as "
        "representative of real-world performance. For reference, Hayat et al. "
        "(2022) report the following for in-hospital mortality on MIMIC-IV + "
        "MIMIC-CXR:\n"
        "  - Table 2 (paired EHR+CXR test set): MedFuse (OPTIMAL) 0.865 AUROC "
        "/ 0.594 AUPRC, vs. Unified (Hayat et al., 2021a) 0.835 / 0.495.\n"
        "  - Table 3 (partial test set): MedFuse (OPTIMAL) 0.874 AUROC / 0.567 "
        "AUPRC."
    )


def parse_args() -> argparse.Namespace:
    """Parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run MedFuse synthetic mortality ablation study."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--ehr-dim", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--cxr-backbone", type=str, default="resnet18")
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation_study(parse_args())