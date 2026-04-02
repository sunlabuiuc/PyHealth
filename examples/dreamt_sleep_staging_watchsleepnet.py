"""Synthetic DREAMT sleep-staging example with WatchSleepNet ablations.

This script demonstrates two usage patterns:

1. A simplified window-classification pipeline.
2. A more paper-aligned sequence-style pipeline using IBI-only epoch features.

The sequence example reports metrics emphasized in the WatchSleepNet paper:
accuracy, macro F1, REM F1, Cohen's kappa, and AUROC. The training loop is
intentionally lightweight and uses synthetic DREAMT-compatible data only. It
does not implement the paper's SHHS/MESA pretraining stage.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import DREAMTDataset, get_dataloader
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingDREAMT, SleepStagingDREAMTSeq

NUM_CLASSES = 3
REM_CLASS_INDEX = 2
IGNORE_INDEX = -100


def build_synthetic_dreamt_root(root: Path, num_subjects: int = 4) -> Path:
    """Create a tiny DREAMT-style directory with wearable CSV files."""
    dreamt_root = root / "dreamt"
    (dreamt_root / "data_64Hz").mkdir(parents=True)

    participant_rows = []
    for subject_index in range(num_subjects):
        patient_id = f"S{subject_index + 1:03d}"
        participant_rows.append(
            {
                "SID": patient_id,
                "AGE": 25 + subject_index,
                "GENDER": "F" if subject_index % 2 == 0 else "M",
                "BMI": 22.0 + subject_index,
                "OAHI": 1.0,
                "AHI": 2.0,
                "Mean_SaO2": "97%",
                "Arousal Index": 10.0,
                "MEDICAL_HISTORY": "None",
                "Sleep_Disorders": "None",
            }
        )

        # Five epochs: one preparation epoch followed by Wake, NREM, REM, NREM.
        labels = (
            ["P"] * 30
            + ["W"] * 30
            + ["N2"] * 30
            + ["R"] * 30
            + ["N3"] * 30
        )
        timestamps = np.arange(len(labels), dtype=np.float32)
        frame = pd.DataFrame(
            {
                "TIMESTAMP": timestamps,
                "IBI": np.sin(timestamps * 0.05 * (subject_index + 1)) + 1.0,
                "HR": 60.0 + 3.0 * np.cos(timestamps * 0.05),
                "BVP": np.sin(timestamps * 0.03),
                "EDA": np.linspace(0.01, 0.04, len(labels)),
                "TEMP": np.full(len(labels), 33.0 + 0.1 * subject_index),
                "ACC_X": np.zeros(len(labels)),
                "ACC_Y": np.ones(len(labels)),
                "ACC_Z": np.full(len(labels), 2.0),
                "Sleep_Stage": labels,
            }
        )
        frame.to_csv(
            dreamt_root / "data_64Hz" / f"{patient_id}_whole_df.csv",
            index=False,
        )

    pd.DataFrame(participant_rows).to_csv(
        dreamt_root / "participant_info.csv",
        index=False,
    )
    return dreamt_root


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1_scores = []
    for class_index in range(num_classes):
        true_pos = np.sum((y_true == class_index) & (y_pred == class_index))
        false_pos = np.sum((y_true != class_index) & (y_pred == class_index))
        false_neg = np.sum((y_true == class_index) & (y_pred != class_index))
        precision = true_pos / max(true_pos + false_pos, 1)
        recall = true_pos / max(true_pos + false_neg, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def _class_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> float:
    true_pos = np.sum((y_true == positive_class) & (y_pred == positive_class))
    false_pos = np.sum((y_true != positive_class) & (y_pred == positive_class))
    false_neg = np.sum((y_true == positive_class) & (y_pred != positive_class))
    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[int(true_label), int(pred_label)] += 1

    total = confusion.sum()
    if total == 0:
        return 0.0
    observed = np.trace(confusion) / total
    expected = np.sum(confusion.sum(axis=0) * confusion.sum(axis=1)) / (total * total)
    if expected >= 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    num_pos = int(positive_mask.sum())
    num_neg = int(negative_mask.sum())
    if num_pos == 0 or num_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    positive_ranks = ranks[positive_mask]
    auc = (positive_ranks.sum() - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg)
    return float(auc)


def _multiclass_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> float:
    aucs = []
    for class_index in range(num_classes):
        one_vs_rest = (y_true == class_index).astype(np.int64)
        auc = _binary_auc(one_vs_rest, y_prob[:, class_index])
        if not np.isnan(auc):
            aucs.append(auc)
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    y_pred = y_prob.argmax(axis=-1)
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = _macro_f1(y_true, y_pred, num_classes)
    rem_f1 = _class_f1(y_true, y_pred, REM_CLASS_INDEX)
    kappa = _cohen_kappa(y_true, y_pred, num_classes)
    auroc = _multiclass_auroc(y_true, y_prob, num_classes)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "rem_f1": rem_f1,
        "cohen_kappa": kappa,
        "auroc": auroc,
    }


def evaluate_model(model, loader, sequence_output: bool = False) -> dict[str, float]:
    y_true_parts = []
    y_prob_parts = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(**batch)
            y_prob = _to_numpy(output["y_prob"])
            y_true = _to_numpy(output["y_true"])
            if sequence_output:
                valid_mask = y_true != IGNORE_INDEX
                if not np.any(valid_mask):
                    continue
                y_true_parts.append(y_true[valid_mask])
                y_prob_parts.append(y_prob[valid_mask])
            else:
                y_true_parts.append(y_true.reshape(-1))
                y_prob_parts.append(y_prob.reshape(-1, y_prob.shape[-1]))

    if not y_true_parts:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "rem_f1": 0.0,
            "cohen_kappa": 0.0,
            "auroc": 0.0,
        }

    y_true_all = np.concatenate(y_true_parts, axis=0)
    y_prob_all = np.concatenate(y_prob_parts, axis=0)
    return compute_metrics(y_true_all, y_prob_all)


def train_one_epoch(model, loader, optimizer) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0
    for batch in loader:
        optimizer.zero_grad()
        output = model(**batch)
        output["loss"].backward()
        optimizer.step()
        running_loss += float(output["loss"].item())
        num_batches += 1
    return running_loss / max(num_batches, 1)


def run_window_ablation(sample_dataset) -> None:
    """Run a simple window-classification ablation."""
    configs = [
        {"name": "baseline", "hidden_dim": 32, "use_tcn": True, "use_attention": True},
        {
            "name": "no_attention",
            "hidden_dim": 32,
            "use_tcn": True,
            "use_attention": False,
        },
        {"name": "no_tcn", "hidden_dim": 32, "use_tcn": False, "use_attention": True},
        {
            "name": "small_hidden",
            "hidden_dim": 16,
            "use_tcn": True,
            "use_attention": True,
        },
    ]
    loader = get_dataloader(sample_dataset, batch_size=4, shuffle=True)

    print("Window classification ablation")
    for config in configs:
        model = WatchSleepNet(
            dataset=sample_dataset,
            hidden_dim=config["hidden_dim"],
            conv_channels=config["hidden_dim"],
            num_attention_heads=4,
            use_tcn=config["use_tcn"],
            use_attention=config["use_attention"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mean_loss = train_one_epoch(model, loader, optimizer)
        metrics = evaluate_model(model, loader, sequence_output=False)
        print(
            f"{config['name']:>12} | loss={mean_loss:.4f} "
            f"| acc={metrics['accuracy']:.3f} "
            f"| macro_f1={metrics['macro_f1']:.3f}"
        )


def run_sequence_ablation(sample_dataset, feature_variant: str) -> None:
    """Run a sequence-style ablation closer to the paper."""
    configs = [
        {
            "name": "baseline_seq",
            "hidden_dim": 32,
            "use_tcn": True,
            "use_attention": True,
        },
        {
            "name": "no_attn_seq",
            "hidden_dim": 32,
            "use_tcn": True,
            "use_attention": False,
        },
        {
            "name": "no_tcn_seq",
            "hidden_dim": 32,
            "use_tcn": False,
            "use_attention": True,
        },
        {
            "name": "small_seq",
            "hidden_dim": 16,
            "use_tcn": True,
            "use_attention": True,
        },
    ]
    loader = get_dataloader(sample_dataset, batch_size=2, shuffle=True)

    print(f"\nSequence-style ablation ({feature_variant})")
    for config in configs:
        model = WatchSleepNet(
            dataset=sample_dataset,
            hidden_dim=config["hidden_dim"],
            conv_channels=config["hidden_dim"],
            num_attention_heads=4,
            use_tcn=config["use_tcn"],
            use_attention=config["use_attention"],
            sequence_output=True,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mean_loss = train_one_epoch(model, loader, optimizer)
        metrics = evaluate_model(model, loader, sequence_output=True)
        print(
            f"{config['name']:>12} | loss={mean_loss:.4f} "
            f"| acc={metrics['accuracy']:.3f} "
            f"| macro_f1={metrics['macro_f1']:.3f} "
            f"| rem_f1={metrics['rem_f1']:.3f} "
            f"| kappa={metrics['cohen_kappa']:.3f} "
            f"| auroc={metrics['auroc']:.3f}"
        )


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="pyhealth_dreamt_example_"))
    try:
        dreamt_root = build_synthetic_dreamt_root(temp_dir)
        dataset = DREAMTDataset(root=str(dreamt_root), cache_dir=temp_dir / "cache")

        window_task = SleepStagingDREAMT(
            window_size=30,
            stride=30,
            source_preference="wearable",
        )
        window_dataset = dataset.set_task(task=window_task, num_workers=1)
        print(f"Generated {len(window_dataset)} synthetic window samples")
        run_window_ablation(window_dataset)

        seq_task = SleepStagingDREAMTSeq(
            feature_columns=("IBI",),
            epoch_seconds=30.0,
            sequence_length=4,
            source_preference="wearable",
            ignore_index=IGNORE_INDEX,
        )
        seq_dataset = dataset.set_task(task=seq_task, num_workers=1)
        print(f"\nGenerated {len(seq_dataset)} IBI-only sequence samples")
        run_sequence_ablation(seq_dataset, feature_variant="IBI only")

        rich_seq_task = SleepStagingDREAMTSeq(
            feature_columns=SleepStagingDREAMT.DEFAULT_FEATURE_COLUMNS,
            epoch_seconds=30.0,
            sequence_length=4,
            source_preference="wearable",
            ignore_index=IGNORE_INDEX,
        )
        rich_seq_dataset = dataset.set_task(task=rich_seq_task, num_workers=1)
        print(
            f"Generated {len(rich_seq_dataset)} multi-signal sequence samples"
        )
        run_sequence_ablation(rich_seq_dataset, feature_variant="IBI + context")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
