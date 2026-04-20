"""EBCL ablation script using synthetic EHR-like samples."""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets.sample_dataset import SampleEHRDataset
from pyhealth.models.ebcl import EBCL


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_synthetic_samples(
    num_samples: int = 128,
    seq_len: int = 16,
    input_dim: int = 16,
) -> List[Dict]:
    """Build synthetic pre/post-event EHR-like samples."""
    samples: List[Dict] = []
    for i in range(num_samples):
        left_x = torch.randn(seq_len, input_dim)
        right_x = left_x + 0.1 * torch.randn(seq_len, input_dim)

        sample = {
            "patient_id": f"patient-{i // 2}",
            "visit_id": f"visit-{i}",
            "left_x": left_x.tolist(),
            "right_x": right_x.tolist(),
            "label": int(i % 2),
        }
        samples.append(sample)
    return samples


def build_dataset(
    num_samples: int = 128,
    seq_len: int = 16,
    input_dim: int = 16,
) -> SampleEHRDataset:
    """Builds a synthetic dataset for the experiment."""
    samples = build_synthetic_samples(
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim,
    )
    dataset = SampleEHRDataset(
        samples=samples,
        dataset_name="synthetic_ebcl",
        task_name="event_contrastive_pretraining",
    )
    return dataset


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert SampleEHRDataset samples into tensors for EBCL."""
    left_x = torch.tensor([sample["left_x"] for sample in batch], dtype=torch.float32)
    right_x = torch.tensor([sample["right_x"] for sample in batch],dtype=torch.float32,)
    
    left_mask = torch.ones(left_x.size(0), left_x.size(1), dtype=torch.bool)
    right_mask = torch.ones(right_x.size(0), right_x.size(1), dtype=torch.bool)

    return { "left_x": left_x,"right_x": right_x,"left_mask": left_mask,"right_mask": right_mask,}


def train_one_setting(
    dataset: SampleEHRDataset,
    temperature: float,
    projection_dim: int,
    hidden_dim: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> float:
    """Trains EBCL with one set of hyperparameters."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = EBCL(
        dataset=dataset,
        input_dim=16,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        temperature=temperature,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                left_x=batch["left_x"],
                right_x=batch["right_x"],
                left_mask=batch["left_mask"],
                right_mask=batch["right_mask"],
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        print(
            f"[temp={temperature}, proj={projection_dim}] "
            f"epoch={epoch + 1}, loss={avg_loss:.4f}"
        )

    return history[-1]


def main() -> None:
    """Run EBCL ablations on an existing PyHealth dataset class."""
    set_seed(42)

    dataset = build_dataset(num_samples=128, seq_len=16, input_dim=16)

    print("Dataset class:", dataset.__class__.__name__)
    print("Dataset name:", dataset.dataset_name)
    print("Task name:", dataset.task_name)
    print("Number of samples:", len(dataset))
    print()

    settings: List[Dict[str, float]] = [
        {"temperature": 0.07, "projection_dim": 16},
        {"temperature": 0.10, "projection_dim": 16},
        {"temperature": 0.07, "projection_dim": 32},
    ]

    results: List[Tuple[Dict[str, float], float]] = []
    for cfg in settings:
        final_loss = train_one_setting(
            dataset=dataset,
            temperature=cfg["temperature"],
            projection_dim=cfg["projection_dim"],
        )
        results.append((cfg, final_loss))

    results = sorted(results, key=lambda x: x[1])

    print("\n=== Ablation Summary (lower is better) ===")
    for cfg, final_loss in results:
        print(
            f"temperature={cfg['temperature']}, "
            f"projection_dim={cfg['projection_dim']} -> "
            f"final_loss={final_loss:.4f}"
        )


if __name__ == "__main__":
    main()