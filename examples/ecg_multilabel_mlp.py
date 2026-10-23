"""
Synthetic ablation study for ECGMultiLabelCardiologyTask.

Authored by Jonathan Gong, Misael Lazaro, and Sydney Robeson
NetIDs: jgong11, misaell2, sel9

This task is inspired by Nonaka & Seita (2021)
"In-depth Benchmarking of Deep Neural Network Architectures for ECG Diagnosis"
Paper link: https://proceedings.mlr.press/v149/nonaka21a.html

This example is intended for a standalone-task style demonstration using only
synthetic data. It avoids deprecated signal dataset classes by:
1. creating synthetic PhysioNet-style ECG files (.mat + .hea),
2. running ECGMultiLabelCardiologyTask on those files,
3. converting the task outputs into a modern SampleDataset via
   create_sample_dataset(...),
4. training an existing PyHealth model on the processed samples, and
5. comparing task configurations in an ablation table.

The ablations vary task-level settings:
- label set
- epoch_sec
- shift
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from scipy.io import savemat
from sklearn.metrics import f1_score
from torch.optim import Adam

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.tasks.ecg_classification import ECGMultiLabelCardiologyTask


SEED = 24
SAMPLING_RATE = 500
N_LEADS = 12
DURATION_SEC = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AblationConfig:
    name: str
    labels: List[str]
    epoch_sec: int
    shift: int


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_time_axis(length: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, length, dtype=np.float32)


def synthesize_signal(dx_codes: Sequence[str], length: int) -> np.ndarray:
    """Generate synthetic ECG signal with label-specific patterns.

    Args:
        dx_codes: List of diagnosis labels.
        length: Number of timesteps.

    Returns:
        np.ndarray: ECG signal (leads, timesteps).

    Example:
        >>> signal = synthesize_signal(["AF"], 10000)
        >>> signal.shape
        (12, 10000)
    """
    t = make_time_axis(length)
    signal = 0.03 * np.random.randn(N_LEADS, length).astype(np.float32)

    # Shared baseline waveform
    for lead in range(N_LEADS):
        signal[lead] += 0.10 * np.sin(2 * np.pi * (lead + 1) * t)

    if "AF" in dx_codes:
        for lead in [0, 1, 2]:
            signal[lead] += 0.35 * np.sin(
                2 * np.pi * 9 * t + 0.5 * np.sin(2 * np.pi * 3 * t)
            )

    if "RBBB" in dx_codes:
        spike = np.exp(-((t - 0.72) ** 2) / 0.0009).astype(np.float32)
        for lead in [3, 4]:
            signal[lead] += 0.55 * spike

    if "LBBB" in dx_codes:
        bump = np.exp(-((t - 0.62) ** 2) / 0.0035).astype(np.float32)
        for lead in [5, 6]:
            signal[lead] += 0.45 * bump

    if "I-AVB" in dx_codes:
        for lead in [7, 8]:
            signal[lead] += 0.25 * np.sin(2 * np.pi * 2 * t)

    return signal.astype(np.float32)


def make_header(
    record_name: str,
    dx_codes: Sequence[str],
    age: int,
    sex: str,
    length: int,
) -> str:
    return "\n".join(
        [
            f"{record_name} {N_LEADS} {SAMPLING_RATE} {length}",
            f"#Age: {age}",
            f"#Sex: {sex}",
            f"#Dx: {','.join(dx_codes)}",
        ]
    )


def write_record(
    root: Path,
    patient_id: str,
    record_name: str,
    dx_codes: Sequence[str],
    age: int,
    sex: str,
) -> Dict[str, str]:
    """Create synthetic ECG record files (.mat + .hea).

    Args:
        root: Directory path.
        patient_id: Patient identifier.
        record_name: Record name.
        dx_codes: Diagnosis labels.
        age: Patient age.
        sex: Patient sex.

    Returns:
        Dict[str, str]: Visit record.

    Example:
        >>> visit = write_record(...)
    """
    length = SAMPLING_RATE * DURATION_SEC
    signal = synthesize_signal(dx_codes=dx_codes, length=length)

    mat_path = root / f"{record_name}.mat"
    hea_path = root / f"{record_name}.hea"

    savemat(mat_path, {"val": signal})
    hea_path.write_text(
        make_header(
            record_name=record_name,
            dx_codes=dx_codes,
            age=age,
            sex=sex,
            length=length,
        ),
        encoding="utf-8",
    )

    return {
        "load_from_path": str(root),
        "patient_id": patient_id,
        "signal_file": mat_path.name,
        "label_file": hea_path.name,
    }


def build_synthetic_visits(root: Path) -> List[Dict[str, str]]:
    specs = [
        ("p01", "rec01", ["AF"], 63, "Male"),
        ("p02", "rec02", ["RBBB"], 57, "Female"),
        ("p03", "rec03", ["LBBB"], 74, "Male"),
        ("p04", "rec04", ["I-AVB"], 69, "Female"),
        ("p05", "rec05", ["AF", "RBBB"], 61, "Male"),
        ("p06", "rec06", ["AF", "LBBB"], 72, "Female"),
        ("p07", "rec07", ["RBBB", "LBBB"], 58, "Male"),
        ("p08", "rec08", ["AF", "I-AVB"], 65, "Female"),
        ("p09", "rec09", ["RBBB", "I-AVB"], 60, "Male"),
        ("p10", "rec10", ["LBBB", "I-AVB"], 55, "Female"),
        ("p11", "rec11", ["AF", "RBBB", "LBBB"], 71, "Male"),
        ("p12", "rec12", ["AF", "RBBB", "I-AVB", "LBBB"], 67, "Female"),
    ]

    return [
        write_record(root, patient_id, record_name, dx_codes, age, sex)
        for patient_id, record_name, dx_codes, age, sex in specs
    ]


def run_task(config: AblationConfig, visits: List[Dict[str, str]]) -> List[Dict]:
    """Run ECG task on visit data.

    Args:
        config: Ablation configuration.
        visits: List of visit dictionaries.

    Returns:
        List of processed samples.

    Example:
        >>> samples = run_task(cfg, visits)
    """
    task = ECGMultiLabelCardiologyTask(
        labels=config.labels,
        epoch_sec=config.epoch_sec,
        shift=config.shift,
        sampling_rate=SAMPLING_RATE,
    )

    samples: List[Dict] = []
    for visit in visits:
        samples.extend(task(visit))
    return samples


def adapt_samples_for_sampledataset(
    task_samples: List[Dict],
    all_labels: Sequence[str],
) -> List[Dict]:
    """Convert task outputs into PyHealth dataset format.

    Args:
        task_samples: Raw task outputs.
        all_labels: Label list.

    Returns:
        List of dataset-compatible samples.

    Example:
        >>> adapted = adapt_samples_for_sampledataset(samples, labels)
    """
    adapted: List[Dict] = []

    for s in task_samples:
        signal = np.asarray(s["signal"], dtype=np.float32)   # (leads, timesteps)
        label_vec = np.asarray(s["label"], dtype=np.float32)

        active_labels = [
            all_labels[i]
            for i, v in enumerate(label_vec)
            if float(v) > 0.5
        ]

        adapted.append(
            {
                "patient_id": str(s["patient_id"]),
                "visit_id": str(s["visit_id"]),
                "record_id": str(s["record_id"]),
                "signal": signal.reshape(-1).astype(np.float32),  # shape: (60000,)
                "label": active_labels,
            }
        )

    return adapted


def build_dataset(samples: List[Dict], name: str):
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multilabel"},
        dataset_name=name,
        task_name="ECGMultiLabelCardiologyTask",
        in_memory=True,
    )


def split_dataset_by_patient(dataset, ratios=(0.6, 0.2, 0.2), seed: int = SEED):
    """
    Split a SampleDataset by patient while preserving fitted processors/vocab.
    """
    assert len(ratios) == 3
    assert abs(sum(ratios) - 1.0) < 1e-8

    patient_ids = list(dataset.patient_to_index.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = max(1, int(ratios[0] * n))
    n_val = max(1, int(ratios[1] * n))

    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    if len(test_ids) == 0:
        test_ids = [patient_ids[-1]]
        val_ids = patient_ids[n_train:-1]

    def gather_indices(ids):
        indices = []
        for pid in ids:
            indices.extend(dataset.patient_to_index[pid])
        return sorted(indices)

    train_idx = gather_indices(train_ids)
    val_idx = gather_indices(val_ids)
    test_idx = gather_indices(test_ids)

    train_dataset = dataset.subset(train_idx)
    val_dataset = dataset.subset(val_idx)
    test_dataset = dataset.subset(test_idx)

    return train_dataset, val_dataset, test_dataset


def build_model(dataset):
    return MLP(dataset=dataset).to(DEVICE)


def train_one_epoch(model, loader, optimizer) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = {
            k: (v.to(DEVICE) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }

        optimizer.zero_grad()
        output = model(**batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_multilabel_f1(model, loader) -> Dict[str, float]:
    model.eval()

    y_true_all = []
    y_prob_all = []

    for batch in loader:
        batch = {
            k: (v.to(DEVICE) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }
        output = model(**batch)
        y_true_all.append(output["y_true"].detach().cpu().numpy())
        y_prob_all.append(output["y_prob"].detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }


def run_ablation(config: AblationConfig) -> Dict[str, float]:
    """Run full training pipeline for one configuration.

    Args:
        config: Ablation settings.

    Returns:
        Dict of performance metrics.

    Example:
        >>> result = run_ablation(cfg)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        visits = build_synthetic_visits(root)

        task_samples = run_task(config, visits)
        if not task_samples:
            raise RuntimeError(f"No task samples produced for config={config.name}")

        adapted = adapt_samples_for_sampledataset(task_samples, config.labels)

        # Build one full dataset so processors / label vocab are fit exactly once.
        full_dataset = build_dataset(adapted, f"{config.name}_full")

        # Subset the same dataset for train/val/test so the metadata stays consistent.
        train_dataset, val_dataset, test_dataset = split_dataset_by_patient(
            full_dataset,
            ratios=(0.6, 0.2, 0.2),
            seed=SEED,
        )

        train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=8, shuffle=False)
        test_loader = get_dataloader(test_dataset, batch_size=8, shuffle=False)

        # Build the model from the full dataset so output dimensionality is fixed.
        model = build_model(full_dataset)
        optimizer = Adam(model.parameters(), lr=1e-3)

        best_val_macro = -1.0
        best_state = None

        for _ in range(6):
            train_one_epoch(model, train_loader, optimizer)
            val_metrics = evaluate_multilabel_f1(model, val_loader)

            if val_metrics["macro_f1"] > best_val_macro:
                best_val_macro = val_metrics["macro_f1"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate_multilabel_f1(model, test_loader)

        return {
            "config": config.name,
            "labels": len(config.labels),
            "epoch_sec": config.epoch_sec,
            "shift": config.shift,
            "n_task_samples": len(task_samples),
            "macro_f1": test_metrics["macro_f1"],
            "micro_f1": test_metrics["micro_f1"],
        }


def print_results(results: List[Dict[str, float]]) -> None:
    print("\n" + "=" * 96)
    print("ECGMultiLabelCardiologyTask synthetic ablation")
    print("=" * 96)
    print(
        f"{'config':28s} {'labels':>6s} {'epoch':>6s} {'shift':>6s} "
        f"{'samples':>8s} {'macro_f1':>10s} {'micro_f1':>10s}"
    )
    print("-" * 96)
    for r in results:
        print(
            f"{r['config']:28s} "
            f"{r['labels']:6d} "
            f"{r['epoch_sec']:6d} "
            f"{r['shift']:6d} "
            f"{r['n_task_samples']:8d} "
            f"{r['macro_f1']:10.4f} "
            f"{r['micro_f1']:10.4f}"
        )
    print("=" * 96)


def main() -> None:
    set_seed()

    ablations = [
        AblationConfig(
            name="labels2_epoch10_shift5",
            labels=["AF", "RBBB"],
            epoch_sec=10,
            shift=5,
        ),
        AblationConfig(
            name="labels3_epoch10_shift5",
            labels=["AF", "RBBB", "LBBB"],
            epoch_sec=10,
            shift=5,
        ),
        AblationConfig(
            name="labels4_epoch10_shift5",
            labels=["AF", "RBBB", "I-AVB", "LBBB"],
            epoch_sec=10,
            shift=5,
        ),
        AblationConfig(
            name="labels4_epoch5_shift5",
            labels=["AF", "RBBB", "I-AVB", "LBBB"],
            epoch_sec=5,
            shift=5,
        ),
        AblationConfig(
            name="labels4_epoch10_shift10",
            labels=["AF", "RBBB", "I-AVB", "LBBB"],
            epoch_sec=10,
            shift=10,
        ),
    ]

    results = [run_ablation(cfg) for cfg in ablations]
    print_results(results)


if __name__ == "__main__":
    main()
