"""
EventContrastiveModel demo + ablation on MIMIC-IV Demo (EHR only), with fallback.

This example follows the style of the provided notebook example:
1) Environment setup
2) Load dataset and set task (MIMIC-IV demo if available)
3) Convert task samples into a numeric SampleDataset
4) Split dataset
5) Initialize model
6) Test forward pass (sanity check)
7) Run ablation study (vary hyperparameters)

Fallback behavior
-----------------
If the local MIMIC4_DEMO_ROOT path does not exist or dataset loading fails, this
script falls back to a small synthetic SampleDataset built with PyHealth's
create_sample_dataset. This ensures the ablation is always runnable.

Experimental setup
------------------
- Dataset: MIMIC-IV demo v2.2 (local files), EHR-only via MIMIC4EHRDataset.
  Fallback: synthetic time-series dataset.
- Task: MortalityPredictionMIMIC4 (MIMIC path only).
- Input adapter: Task samples contain coded modalities (often processed tensors).
  We flatten and stringify these values, then build a small numeric time-series
  x:[TIME_STEPS, N_FEATURES] using a hashed count featurizer.
- Pretraining: Contrastive pretraining via EventContrastiveModel.compute_loss().
- Downstream: Freeze encoder; train a small classifier head; report accuracy.

Findings from the demo run
--------------------------
- The MIMIC-IV demo dataset can yield a small validation set, making accuracy
  coarse and sometimes similar across multiple configurations. Pretraining loss
  differences are typically more visible.

Run
---
python examples/mimiciv_eventcontrastive.py
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import MIMIC4EHRDataset, create_sample_dataset, get_dataloader
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.models.event_contrastive import EventContrastiveModel
from pyhealth.tasks import MortalityPredictionMIMIC4

# -------------------------
# 1) Environment setup
# -------------------------

SEED = 42
TIME_STEPS = 40
N_FEATURES = 8
MAX_SAMPLES = 2000

MIMIC4_DEMO_ROOT = "LOCAL PATH TO DATASET"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# 2) Helpers: numeric dataset adapter
# -------------------------


def stable_hash(token: str) -> int:
    """Deterministic hash for token strings."""
    h = 2166136261
    for ch in token:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def chunked(items: Sequence[str], n_chunks: int) -> List[List[str]]:
    """Split items into n_chunks (approximately equal size)."""
    if n_chunks <= 1:
        return [list(items)]
    chunks: List[List[str]] = [[] for _ in range(n_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % n_chunks].append(item)
    return chunks


def codes_to_timeseries(
    codes: Sequence[str],
    time_steps: int = TIME_STEPS,
    n_features: int = N_FEATURES,
) -> torch.Tensor:
    """Convert token strings into a numeric time-series tensor [T, F]."""
    x = torch.zeros(time_steps, n_features, dtype=torch.float32)
    for t, chunk in enumerate(chunked(list(codes), time_steps)):
        for token in chunk:
            x[t, stable_hash(token) % n_features] += 1.0
    return x


def get_field(sample: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Return the first present field value from sample for keys, else None."""
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def to_token_list(value: Any) -> List[str]:
    """Convert tensor/list/scalar into a list of string tokens."""
    if value is None:
        return []

    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1).tolist()
        return [str(v) for v in flat]

    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist()
        return [str(v) for v in flat]

    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]

    return [str(value)]


def sample_to_xy(sample: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    """Extract x:[T, F] and y:int from a MortalityPredictionMIMIC4 sample."""
    conditions_val = get_field(sample, ["conditions", "diagnoses"])
    procedures_val = get_field(sample, ["procedures"])
    drugs_val = get_field(sample, ["drugs", "medications"])

    codes: List[str] = []
    codes.extend(to_token_list(conditions_val))
    codes.extend(to_token_list(procedures_val))
    codes.extend(to_token_list(drugs_val))

    x = codes_to_timeseries(codes, time_steps=TIME_STEPS, n_features=N_FEATURES)

    label_val = get_field(sample, ["label", "mortality"])
    y = int(label_val) if label_val is not None else 0
    return x, y


def build_numeric_sample_dataset(task_dataset) -> Any:
    """Convert task samples into a numeric SampleDataset for EventContrastiveModel."""
    samples: List[Dict[str, Any]] = []
    for i, sample in enumerate(task_dataset):
        if i >= MAX_SAMPLES:
            break
        x, y = sample_to_xy(sample)
        samples.append(
            {
                "patient_id": sample.get("patient_id", f"patient-{i}"),
                "visit_id": sample.get("visit_id", "visit-0"),
                "x": x.numpy().tolist(),
                "label": y,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={"x": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="mimiciv_demo_eventcontrastive_numeric",
    )


def build_synthetic_dataset(
    n_samples: int = 200,
    signal_shift: float = 0.75,
) -> Any:
    """Build a fully synthetic numeric SampleDataset as a robust fallback.

    Label rule:
      label is sampled first; feature 0 is shifted by +/- signal_shift across time.
    """
    samples: List[Dict[str, Any]] = []
    for i in range(n_samples):
        label = int(torch.rand(1).item() > 0.5)
        x = torch.randn(TIME_STEPS, N_FEATURES)
        x[:, 0] += signal_shift if label == 1 else -signal_shift

        samples.append(
            {
                "patient_id": f"patient-{i}",
                "visit_id": "visit-0",
                "x": x.numpy().tolist(),
                "label": label,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={"x": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_eventcontrastive_numeric",
    )


# -------------------------
# 3) Pretrain + downstream evaluation
# -------------------------


def pool_patient_embedding(event_embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average event embeddings into one embedding per patient."""
    return torch.stack(list(event_embeddings), dim=0).mean(dim=0)


class DownstreamClassifier(nn.Module):
    """Small classifier head used for frozen-encoder evaluation."""

    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_contrastive_pretraining(
    model: EventContrastiveModel,
    train_loader,
    lr: float,
    epochs: int = 2,
) -> float:
    """Contrastive pretraining loop; returns final epoch average loss."""
    model.train()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    final_avg_loss = 0.0
    for _ in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            embeddings = model(x=x)
            loss = model.compute_loss(embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        final_avg_loss = running_loss / max(n_batches, 1)

    return final_avg_loss


def evaluate_accuracy(model: EventContrastiveModel, clf: nn.Module, loader) -> float:
    """Compute accuracy on a dataloader."""
    model.eval()
    clf.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            y = batch["label"].to(DEVICE).long().view(-1)

            emb = pool_patient_embedding(model(x=x))
            logits = clf(emb)
            pred = logits.argmax(dim=-1)

            correct += int((pred == y).sum().item())
            total += int(y.numel())

    return correct / max(total, 1)


def run_downstream_eval(
    model: EventContrastiveModel,
    train_loader,
    val_loader,
    test_loader,
    lr: float,
    dropout: float,
    epochs: int = 6,
) -> Tuple[float, float]:
    """Freeze encoder; train classifier head; return (val_acc, test_acc)."""
    model.eval()
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False

    proj_dim = int(model.projection_head[-1].out_features)
    clf = DownstreamClassifier(input_dim=proj_dim, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)

    clf.train()
    for _ in range(epochs):
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            y = batch["label"].to(DEVICE).long().view(-1)

            with torch.no_grad():
                emb = pool_patient_embedding(model(x=x))

            logits = clf(emb)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    val_acc = evaluate_accuracy(model, clf, val_loader)
    test_acc = evaluate_accuracy(model, clf, test_loader)
    return val_acc, test_acc


# -------------------------
# 4) Ablation configs + summary
# -------------------------


@dataclass(frozen=True)
class AblationConfig:
    """Ablation configuration for EventContrastiveModel + classifier head."""

    name: str
    hidden_dim: int
    projection_dim: int
    temperature: float
    pretrain_lr: float
    clf_lr: float
    clf_dropout: float


def print_summary(results: List[Tuple[AblationConfig, Dict[str, float]]]) -> None:
    """Print a compact comparison table across hyperparameter configurations."""
    header = (
        f"{'config':<16} {'hid':>5} {'proj':>5} {'temp':>6} "
        f"{'pt_lr':>9} {'clf_lr':>9} {'drop':>6} "
        f"{'pre_loss':>9} {'val':>7} {'test':>7}"
    )
    print("\n=== Ablation Summary ===")
    print(header)
    for cfg, metrics in results:
        row = (
            f"{cfg.name:<16} {cfg.hidden_dim:>5} {cfg.projection_dim:>5} "
            f"{cfg.temperature:>6.2f} {cfg.pretrain_lr:>9.1e} "
            f"{cfg.clf_lr:>9.1e} {cfg.clf_dropout:>6.2f} "
            f"{metrics['pre_loss']:>9.3f} {metrics['val_acc']:>7.3f} "
            f"{metrics['test_acc']:>7.3f}"
        )
        print(row)


# -------------------------
# Main: notebook-style steps
# -------------------------

if __name__ == "__main__":
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2) Load dataset and set task (fallback to synthetic if loading fails)
    dataset = None
    if os.path.exists(MIMIC4_DEMO_ROOT):
        tables = ["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"]
        try:
            base_dataset = MIMIC4EHRDataset(
                root=MIMIC4_DEMO_ROOT,
                tables=tables,
                dev=False,
            )
            base_dataset.stats()

            task = MortalityPredictionMIMIC4()
            task_dataset = base_dataset.set_task(task)
            print(f"Task dataset size: {len(task_dataset)} samples")

            dataset = build_numeric_sample_dataset(task_dataset)
            print(f"Numeric dataset size: {len(dataset)} samples")
        except Exception as exc:
            print(f"Failed to load MIMIC-IV demo dataset: {exc}")
            dataset = None
    else:
        print(f"MIMIC4_DEMO_ROOT not found: {MIMIC4_DEMO_ROOT}")

    if dataset is None:
        print("Falling back to synthetic demo dataset.")
        dataset = build_synthetic_dataset(n_samples=200)

    print(f"Input schema: {dataset.input_schema}")
    print(f"Output schema: {dataset.output_schema}")

    # 4) Split dataset
    train_data, val_data, test_data = split_by_patient(
        dataset,
        [0.8, 0.1, 0.1],
        seed=SEED,
    )
    print(f"Train: {len(train_data)} samples")
    print(f"Val:   {len(val_data)} samples")
    print(f"Test:  {len(test_data)} samples")

    train_loader = get_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)

    # 5) Initialize model (baseline)
    baseline_cfg = AblationConfig(
        name="baseline",
        hidden_dim=64,
        projection_dim=32,
        temperature=0.10,
        pretrain_lr=3e-3,
        clf_lr=2e-3,
        clf_dropout=0.10,
    )
    model = EventContrastiveModel(
        dataset=dataset,
        input_dim=N_FEATURES,
        hidden_dim=baseline_cfg.hidden_dim,
        projection_dim=baseline_cfg.projection_dim,
        temperature=baseline_cfg.temperature,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params} parameters")

    # 6) Forward pass sanity check
    batch = next(iter(train_loader))
    with torch.no_grad():
        embeddings = model(x=batch["x"].to(DEVICE))
        print(f"Number of events returned: {len(embeddings)}")
        print(f"One embedding shape: {tuple(embeddings[0].shape)}")
        if len(embeddings) >= 2:
            loss = model.compute_loss(embeddings)
            print(f"Contrastive loss (sanity): {loss.item():.4f}")

    # 7) Ablation study: vary key hyperparameters
    configs = [
        baseline_cfg,
        AblationConfig("tiny_hidden", 16, 32, 0.10, 3e-3, 2e-3, 0.10),
        AblationConfig("big_hidden", 128, 32, 0.10, 3e-3, 2e-3, 0.10),
        AblationConfig("low_temp", 64, 32, 0.05, 3e-3, 2e-3, 0.10),
        AblationConfig("high_temp", 64, 32, 0.30, 3e-3, 2e-3, 0.10),
        AblationConfig("larger_proj", 64, 64, 0.10, 3e-3, 2e-3, 0.10),
        AblationConfig("lower_pt_lr", 64, 32, 0.10, 1e-3, 2e-3, 0.10),
        AblationConfig("more_dropout", 64, 32, 0.10, 3e-3, 2e-3, 0.40),
    ]

    results: List[Tuple[AblationConfig, Dict[str, float]]] = []
    for cfg in configs:
        set_seed(SEED)

        model = EventContrastiveModel(
            dataset=dataset,
            input_dim=N_FEATURES,
            hidden_dim=cfg.hidden_dim,
            projection_dim=cfg.projection_dim,
            temperature=cfg.temperature,
        )

        pre_loss = run_contrastive_pretraining(
            model=model,
            train_loader=train_loader,
            lr=cfg.pretrain_lr,
            epochs=2,
        )
        val_acc, test_acc = run_downstream_eval(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=cfg.clf_lr,
            dropout=cfg.clf_dropout,
            epochs=6,
        )

        metrics = {"pre_loss": pre_loss, "val_acc": val_acc, "test_acc": test_acc}
        results.append((cfg, metrics))
        print(
            f"[{cfg.name}] pre_loss={pre_loss:.3f} val={val_acc:.3f} "
            f"test={test_acc:.3f}"
        )

    print_summary(results)