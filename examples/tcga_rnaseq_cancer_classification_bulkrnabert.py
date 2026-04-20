"""End-to-end cancer type classification example with TCGA RNA-seq and BulkRNABert.

This example demonstrates three ablation configurations for cancer type
classification on the TCGARNASeqDataset:

1. Frozen encoder + MLP head (linear probing baseline)
2. IA3 fine-tuning + MLP head
3. Full fine-tuning

The script supports a synthetic mode so the full pipeline can be demonstrated
without downloading TCGA data.

Example usage:

Synthetic demo:
    python examples/tcga_rnaseq_cancer_classification_bulkrnabert.py \
        --synthetic --epochs 2 --batch_size 4

Real data:
    python examples/tcga_rnaseq_cancer_classification_bulkrnabert.py \
        --data_dir /path/to/tcga_phase1_out --epochs 5 --batch_size 16

Expected files in --data_dir:
    gene_expression.csv
    clinical.csv

Notes:
- This script is intentionally lightweight and designed for reproducibility and
  demonstration.
- The paper reports strong pan-cancer performance for frozen and IA3-style
  settings. Exact numbers are not expected here because this PyHealth project
  uses a simplified BulkRNABert implementation and local training setup.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pyhealth.datasets import TCGARNASeqDataset
from pyhealth.models import BulkRNABert
from pyhealth.tasks import TCGARNASeqCancerTypeClassification

def _get_embedding_fixed(self, gene_expression):
    if self._encoder_is_scratch:
        projected = self.input_proj(gene_expression)
        hidden = self.encoder(projected.unsqueeze(1))
        embedding = hidden.squeeze(1)
    else:
        import torch
        gene_bins = (gene_expression * 63).long().clamp(0, 63)
        enc = self.encoder
        x = enc.expression_embedding_layer(gene_bins)
        if enc.config.use_gene_embedding:
            gene_indices = torch.arange(enc.config.n_genes, device=x.device)
            gene_embedding = enc.gene_embedding_layer(gene_indices)
            if enc.config.project_gene_embedding:
                gene_embedding = enc.fc_gene_embedding(gene_embedding)
            x = x + gene_embedding
        batch_size, seq_length = gene_bins.shape
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length),
            device=gene_bins.device,
            dtype=torch.bool,
        )
        for transformer in enc.transformer_layers:
            output = transformer(x, attention_mask=attention_mask)
            x = output["embeddings"]
        embedding = x.mean(dim=1)
    return embedding

BulkRNABert._get_embedding = _get_embedding_fixed

DEFAULT_NUM_GENES = 19062
DEFAULT_NUM_CLASSES = 33
DEFAULT_SYNTHETIC_SAMPLES = 96
DEFAULT_SEED = 42

COHORTS = [
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
]


@dataclass
class ExperimentConfig:
    name: str
    freeze_encoder: bool
    use_ia3: bool


class ClassificationSampleDataset(Dataset):
    """Minimal torch dataset wrapper for classification samples."""

    def __init__(self, samples: Sequence[Dict]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        return {
            "patient_id": sample["patient_id"],
            "gene_expression": torch.as_tensor(
                sample["gene_expression"], dtype=torch.float32
            ),
            "label": int(sample["label"]),
        }


class SyntheticTCGADataset:
    """Small synthetic stand-in with the minimal interface needed for the task.

    Person 1's task expects a list of patient-like dicts. Each patient contains
    one RNA-seq event with a gene expression vector and cohort label.
    """

    def __init__(
        self,
        num_samples: int,
        num_genes: int,
        cohorts: Sequence[str],
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.patients: List[Dict] = []

        for i in range(num_samples):
            cohort = cohorts[i % len(cohorts)]
            patient_id = f"SYN-{i:04d}"
            sample_id = f"SYN-{i:04d}-RNA"

            # Create mildly class-structured features so the demo learns something.
            class_index = i % len(cohorts)
            gene_expression = rng.lognormal(mean=0.0, sigma=0.5, size=num_genes)
            start = (class_index * 17) % max(1, num_genes - 32)
            gene_expression[start : start + 32] += 2.5

            self.patients.append(
                {
                    "patient_id": patient_id,
                    "cohort": cohort,
                    "events": [
                        {
                            "sample_id": sample_id,
                            "gene_expression": gene_expression.astype(np.float32),
                            "cohort": cohort,
                        }
                    ],
                }
            )


class SyntheticTCGARNASeqCancerTypeClassification:
    """Synthetic replacement for Person 1's task in demo mode."""

    def __init__(self, cohorts: Sequence[str]) -> None:
        self.cohorts = list(cohorts)
        self.label_map = {cohort: idx for idx, cohort in enumerate(self.cohorts)}

    def __call__(self, dataset: SyntheticTCGADataset) -> List[Dict]:
        samples: List[Dict] = []
        for patient in dataset.patients:
            for event in patient["events"]:
                samples.append(
                    {
                        "patient_id": patient["patient_id"],
                        "gene_expression": event["gene_expression"],
                        "label": self.label_map[event["cohort"]],
                    }
                )
        return samples


class MetricsLogger:
    def __init__(self) -> None:
        self.rows: List[Dict[str, float | str]] = []

    def add(self, row: Dict[str, float | str]) -> None:
        self.rows.append(row)

    def print_table(self) -> None:
        if not self.rows:
            return
        headers = list(self.rows[0].keys())
        widths = {
            key: max(len(key), max(len(f"{row[key]}") for row in self.rows))
            for key in headers
        }
        header_line = " | ".join(key.ljust(widths[key]) for key in headers)
        sep_line = "-+-".join("-" * widths[key] for key in headers)
        print("\nResults")
        print(header_line)
        print(sep_line)
        for row in self.rows:
            print(
                " | ".join(str(row[key]).ljust(widths[key]) for key in headers)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TCGA RNA-seq cancer classification with BulkRNABert ablations"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on synthetic data instead of real TCGA CSV files.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing gene_expression.csv and clinical.csv.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num_workers", type=int, default=0, help="DataLoader worker count."
    )
    parser.add_argument(
        "--synthetic_num_samples",
        type=int,
        default=DEFAULT_SYNTHETIC_SAMPLES,
    )
    parser.add_argument(
        "--synthetic_num_genes",
        type=int,
        default=DEFAULT_NUM_GENES,
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save the ablation summary as CSV.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["frozen_encoder_mlp", "ia3_finetuning", "full_finetuning"],
        help="Run only one training mode. If not set, runs all three.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_real_data_dir(data_dir: Optional[str]) -> Path:
    if data_dir is None:
        raise ValueError("--data_dir is required when not using --synthetic.")
    path = Path(data_dir)
    gene_path = path / "gene_expression.csv"
    clinical_path = path / "clinical.csv"
    if not gene_path.exists() or not clinical_path.exists():
        raise FileNotFoundError(
            f"Expected {gene_path} and {clinical_path} to exist."
        )
    return path


def stratified_split_by_patient(
    samples: Sequence[Dict], test_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)

    by_label: Dict[int, List[Dict]] = {}
    for sample in samples:
        by_label.setdefault(int(sample["label"]), []).append(sample)

    train_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for label_samples in by_label.values():
        patient_ids = sorted({sample["patient_id"] for sample in label_samples})
        rng.shuffle(patient_ids)
        n_test = max(1, int(round(len(patient_ids) * test_ratio))) if len(patient_ids) > 1 else 0
        test_patient_ids = set(patient_ids[:n_test])
        for sample in label_samples:
            if sample["patient_id"] in test_patient_ids:
                test_samples.append(sample)
            else:
                train_samples.append(sample)

    return train_samples, test_samples


def build_synthetic_samples(args: argparse.Namespace) -> Tuple[List[Dict], int]:
    synthetic_dataset = SyntheticTCGADataset(
        num_samples=args.synthetic_num_samples,
        num_genes=args.synthetic_num_genes,
        cohorts=COHORTS,
        seed=args.seed,
    )
    synthetic_task = SyntheticTCGARNASeqCancerTypeClassification(COHORTS)
    samples = synthetic_task(synthetic_dataset)
    return samples, len(COHORTS)


def build_real_samples(args: argparse.Namespace) -> Tuple[List[Dict], int]:
    data_dir = ensure_real_data_dir(args.data_dir)

    # TCGARNASeqDataset expects the two CSVs inside a root directory.
    dataset = TCGARNASeqDataset(root=str(data_dir))

    # Use set_task() to get a SampleDataset, then convert to plain dicts.
    # Calling task(dataset) directly does not work with PyHealth BaseDataset.
    task = TCGARNASeqCancerTypeClassification()
    sample_dataset = dataset.set_task(task)

    samples: List[Dict] = []
    for i in range(len(sample_dataset)):
        s = sample_dataset[i]
        gene_expr = s["gene_expression"]
        # Convert tensor to numpy if needed so ClassificationSampleDataset
        # can later wrap it back into a tensor cleanly.
        if hasattr(gene_expr, "numpy"):
            gene_expr = gene_expr.numpy()
        samples.append(
            {
                "patient_id": s["patient_id"],
                "gene_expression": gene_expr,
                "label": int(s["label"]),
            }
        )

    labels = sorted({int(s["label"]) for s in samples})
    num_classes = len(labels)
    return samples, num_classes


def make_loaders(
    train_samples: Sequence[Dict],
    test_samples: Sequence[Dict],
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = ClassificationSampleDataset(train_samples)
    test_dataset = ClassificationSampleDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def build_model(
    num_genes: int,
    num_classes: int,
    config: ExperimentConfig,
    device: torch.device,
) -> nn.Module:
    model = BulkRNABert(
        num_genes=num_genes,
        num_classes=num_classes,
        task="classification",
        freeze_encoder=config.freeze_encoder,
        use_ia3=config.use_ia3,
    )
    model.to(device)
    return model


def collect_num_genes(samples: Sequence[Dict]) -> int:
    if not samples:
        raise ValueError("No samples available.")
    gene_expression = samples[0]["gene_expression"]
    return int(len(gene_expression))


def make_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        gene_expression = batch["gene_expression"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(gene_expression=gene_expression, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


@torch.no_grad()
def evaluate_classification(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        gene_expression = batch["gene_expression"].to(device)
        labels = batch["label"].to(device)
        outputs = model(gene_expression=gene_expression, labels=labels)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)

        batch_size = labels.size(0)
        total_loss += float(outputs["loss"].item()) * batch_size
        total_examples += batch_size

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = compute_accuracy(all_labels, all_preds)
    macro_f1 = compute_macro_f1(all_labels, all_preds)
    weighted_f1 = compute_weighted_f1(all_labels, all_preds)
    avg_loss = total_loss / total_examples if total_examples > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def _per_class_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[int, Tuple[int, int, int, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    counts: Dict[int, Tuple[int, int, int, int]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        counts[label] = (tp, fp, fn, support)
    return counts


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    per_class = _per_class_counts(y_true, y_pred)
    f1s: List[float] = []
    for tp, fp, fn, _ in per_class.values():
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def compute_weighted_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    per_class = _per_class_counts(y_true, y_pred)
    total_support = sum(support for _, _, _, support in per_class.values())
    weighted_sum = 0.0
    for tp, fp, fn, support in per_class.values():
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        weighted_sum += f1 * support
    return float(weighted_sum / total_support) if total_support > 0 else 0.0


def save_results_csv(rows: Sequence[Dict[str, float | str]], path: str) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_and_evaluate(
    config: ExperimentConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_genes: int,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float | str]:
    print(f"\n=== Running config: {config.name} ===")
    model = build_model(
        num_genes=num_genes,
        num_classes=num_classes,
        config=config,
        device=device,
    )
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    best_metrics: Optional[Dict[str, float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_classification(model, test_loader, device)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={metrics['loss']:.4f} | acc={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | weighted_f1={metrics['weighted_f1']:.4f}"
        )

        if best_metrics is None or metrics["weighted_f1"] > best_metrics["weighted_f1"]:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_metrics is None:
        raise RuntimeError("No evaluation metrics were produced.")

    if best_state is not None:
        model.load_state_dict(best_state)

    result: Dict[str, float | str] = {
        "config": config.name,
        "accuracy": round(float(best_metrics["accuracy"]), 4),
        "macro_f1": round(float(best_metrics["macro_f1"]), 4),
        "weighted_f1": round(float(best_metrics["weighted_f1"]), 4),
    }
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    # Reduce CUDA memory fragmentation for large models on GPU.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Suppress HuggingFace interactive prompt when loading pretrained model.
    os.environ.setdefault("TRUST_REMOTE_CODE", "1")

    if args.synthetic:
        samples, num_classes = build_synthetic_samples(args)
        print(
            f"Loaded synthetic dataset with {len(samples)} samples, "
            f"{num_classes} classes, {args.synthetic_num_genes} genes."
        )
    else:
        samples, num_classes = build_real_samples(args)
        print(
            f"Loaded real dataset with {len(samples)} samples and {num_classes} classes."
        )

    num_genes = collect_num_genes(samples)
    train_samples, test_samples = stratified_split_by_patient(
        samples=samples,
        test_ratio=0.2,
        seed=args.seed,
    )

    print(
        f"Train samples: {len(train_samples)} | Test samples: {len(test_samples)} | "
        f"Device: {device}"
    )

    train_loader, test_loader = make_loaders(
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    all_configs = [
        ExperimentConfig(
            name="frozen_encoder_mlp",
            freeze_encoder=True,
            use_ia3=False,
        ),
        ExperimentConfig(
            name="ia3_finetuning",
            freeze_encoder=True,
            use_ia3=True,
        ),
        ExperimentConfig(
            name="full_finetuning",
            freeze_encoder=False,
            use_ia3=False,
        ),
    ]
    configs = (
        [c for c in all_configs if c.name == args.mode]
        if args.mode is not None
        else all_configs
    )

    logger = MetricsLogger()
    for config in configs:
        result = train_and_evaluate(
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            num_genes=num_genes,
            num_classes=num_classes,
            args=args,
            device=device,
        )
        logger.add(result)

    logger.print_table()

    if args.output_csv:
        save_results_csv(logger.rows, args.output_csv)
        print(f"\nSaved ablation summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
