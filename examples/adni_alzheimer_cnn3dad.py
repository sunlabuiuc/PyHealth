import os
import sys
import time
import random

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.trainer import Trainer
from pyhealth.models.cnn3d_ad import CNN3DAD

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def make_samples():
    rng = np.random.default_rng(42)
    samples = []
    for i in range(60):
        label = i % 3
        scan = rng.standard_normal((1, 96, 96, 96)).astype("float32")
        scan[0, label*20:label*20+30, label*20:label*20+30, label*20:label*20+30] += 0.5
        scan += rng.standard_normal((1, 96, 96, 96)).astype("float32") * 0.8
        age = np.array([rng.uniform(55.0, 90.0)], dtype="float32")
        samples.append({"patient_id": f"p{i:03d}", "scan": scan, "age": age, "label": label})
    return samples

def train_and_eval(name, dataset, **model_kwargs):
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.7, 0.15, 0.15])
    train_loader = get_dataloader(train_ds, batch_size=4, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=4, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=4, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CNN3DAD(dataset=dataset, **model_kwargs)
    trainer = Trainer(model=model, device=device)

    start = time.time()
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=4,
        optimizer_params={"lr": 1e-3},
        monitor="accuracy",
        monitor_criterion="max",
    )
    duration = time.time() - start

    metrics = trainer.evaluate(test_loader)
    print(f" {name}: accuracy={metrics.get('accuracy', float('nan')):.4f} loss={metrics.get('loss', float('nan')):.4f} ({duration:.1f}s)")
    return metrics


def main():
    samples = make_samples()
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"scan": "tensor", "age": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="adni_synthetic",
    )

    class_counts = [sum(1 for s in samples if s["label"] == c) for c in range(3)]
    balanced_weights = [sum(class_counts) / (3.0 * count) for count in class_counts]

    configs = [
        ("1. Normalization type", "norm_type", ["instance", "batch"]),
        ("2. Channel widening factor", "widening_factor", [4, 8, 16]),
        ("3. Age encoding dim (0=off)", "age_encoding_dim", [0, 32, 64]),
        ("4. Number of conv blocks", "num_blocks", [2, 3, 4]),
    ]

    all_results = {}
    for title, key, values in configs:
        print(f"\n{'='*60}\n{title}\n{'='*60}")
        for v in values:
            all_results[f"{key}={v}"] = train_and_eval(f"{key}={v}", dataset, **{key: v})

    print(f"\n{'='*60}\n5. Class weights\n{'='*60}")
    all_results["class_weights=uniform"]  = train_and_eval("class_weights=uniform",  dataset, class_weights=None)
    all_results["class_weights=balanced"] = train_and_eval("class_weights=balanced", dataset, class_weights=balanced_weights)

    print(f"\n{'='*60}\nAblation Summary\n{'='*60}")
    for name, m in all_results.items():
        print(f" {name:35s} acc={m.get('accuracy', float('nan')):.4f}  loss={m.get('loss', float('nan')):.4f}")
        
if __name__ == "__main__":
    main()
    
