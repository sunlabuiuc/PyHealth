"""LRP with StageNet on synthetic data.

Demonstrates Layer-wise Relevance Propagation (LRP) interpretability
on a StageNet model using synthetic patient data. No external datasets required.

Usage:
    python lrp_stagenet_synthetic.py
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.interpret.methods import LayerwiseRelevancePropagation
from pyhealth.models import StageNet
from pyhealth.processors import StageNetProcessor, StageNetTensorProcessor


def generate_synthetic_data(
    num_samples: int = 500,
    num_visits_range: Tuple[int, int] = (3, 10),
    num_codes_range: Tuple[int, int] = (5, 20),
    num_lab_tests: int = 5,
    vocab_size: int = 100,
    seed: int = 42,
) -> list:
    """Generate synthetic patient samples for StageNet."""
    random.seed(seed)
    np.random.seed(seed)
    samples = []

    for i in range(num_samples):
        num_visits = random.randint(*num_visits_range)
        diagnoses_list, diagnosis_times = [], []
        for v in range(num_visits):
            num_codes = random.randint(*num_codes_range)
            diagnoses_list.append(
                [f"D{random.randint(0, vocab_size - 1)}" for _ in range(num_codes)]
            )
            diagnosis_times.append(0.0 if v == 0 else random.uniform(24, 720))

        lab_values_list, lab_times = [], []
        meas_idx = 0
        for v in range(num_visits):
            for _ in range(random.randint(3, 10)):
                vec = []
                for lab_idx in range(num_lab_tests):
                    if (i == 0 and meas_idx == 0) or random.random() < 0.8:
                        vec.append(100.0 + random.gauss(0, 20))
                    else:
                        vec.append(None)
                lab_values_list.append(vec)
                lab_times.append(random.uniform(0, 24))
                meas_idx += 1

        risky = sum(
            1 for codes in diagnoses_list for c in codes if int(c[1:]) < 20
        )
        risk = num_visits * 0.1 + risky * 0.05 + random.gauss(0, 0.1)

        samples.append({
            "patient_id": f"P{i:04d}",
            "diagnoses": (diagnosis_times, diagnoses_list),
            "labs": (lab_times, lab_values_list),
            "label": 1 if risk > 0.5 else 0,
        })
    return samples


def print_top_features(attributions, sample_batch, top_k=10):
    """Print top-k features by absolute LRP relevance."""
    for key, attr_tensor in attributions.items():
        if attr_tensor is None or attr_tensor.numel() == 0:
            continue
        attr = attr_tensor[0].detach().cpu().flatten()
        k = min(top_k, attr.numel())
        _, top_idx = torch.topk(attr.abs(), k=k)

        print(f"\n  {key} (shape={attr_tensor.shape}):")
        for rank, idx in enumerate(top_idx.tolist(), 1):
            print(f"    {rank:2d}. index={idx}, relevance={attr[idx].item():+.6f}")


def main():
    print("Generating synthetic patient data...")
    samples = generate_synthetic_data(num_samples=500, seed=42)
    print(f"  {len(samples)} samples generated")

    # Create dataset
    from pyhealth.datasets.sample_dataset import InMemorySampleDataset
    from pyhealth.processors.base_processor import FeatureProcessor

    class LabelProcessor(FeatureProcessor):
        def fit(self, samples, key):
            pass
        def process(self, value):
            return torch.tensor([value], dtype=torch.float)
        def size(self):
            return 1

    dataset = InMemorySampleDataset(
        samples=samples,
        input_schema={"diagnoses": "stagenet", "labs": "stagenet_tensor"},
        output_schema={"label": "binary"},
        output_processors={"label": LabelProcessor()},
    )

    # Split
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    train_ds = dataset.subset(list(range(n_train)))
    test_ds = dataset.subset(list(range(n_train + n_val, len(dataset))))
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False)

    # Model
    model = StageNet(
        dataset=dataset, embedding_dim=128, chunk_size=128, levels=3, dropout=0.3,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor)
                else tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
                if isinstance(v, tuple) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            out = model(**batch)
            out["loss"].backward()
            optimizer.step()
            total_loss += out["loss"].item()
            preds = out["y_prob"]
            labels = batch["label"].squeeze()
            correct += ((preds > 0.5).long() == labels).sum().item()
            total += labels.size(0)
        print(f"  Epoch {epoch+1}/3: loss={total_loss/len(train_loader):.4f}, "
              f"acc={100*correct/total:.1f}%")

    # LRP
    model.eval()
    sample_batch = next(iter(test_loader))
    sample_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor)
        else tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
        if isinstance(v, tuple) else v
        for k, v in sample_batch.items()
    }

    with torch.no_grad():
        output = model(**sample_batch)
        pred_prob = torch.sigmoid(output["logit"]).item()
    print(f"\nPrediction: class={int(pred_prob > 0.5)}, prob={pred_prob:.4f}, "
          f"true={int(sample_batch['label'].item())}")

    # Epsilon rule
    print("\nLRP Epsilon-Rule (eps=0.01):")
    lrp_eps = LayerwiseRelevancePropagation(
        model, rule="epsilon", epsilon=0.01, use_embeddings=True
    )
    attr_eps = lrp_eps.attribute(**sample_batch)
    print_top_features(attr_eps, sample_batch)

    # AlphaBeta rule
    print("\nLRP AlphaBeta-Rule (alpha=1, beta=0):")
    lrp_ab = LayerwiseRelevancePropagation(
        model, rule="alphabeta", alpha=1.0, beta=0.0, use_embeddings=True
    )
    attr_ab = lrp_ab.attribute(**sample_batch)
    print_top_features(attr_ab, sample_batch)

    # Conservation check
    with torch.no_grad():
        f_x = model(**sample_batch)["logit"].squeeze().item()
    eps_sum = sum(attr_eps[k][0].sum().item() for k in attr_eps)
    ab_sum = sum(attr_ab[k][0].sum().item() for k in attr_ab)
    print(f"\nConservation: f(x)={f_x:.6f}, "
          f"eps_sum={eps_sum:.6f}, ab_sum={ab_sum:.6f}")


if __name__ == "__main__":
    main()