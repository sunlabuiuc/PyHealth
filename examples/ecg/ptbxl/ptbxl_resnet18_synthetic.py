"""CPU smoke test for the PTB-XL ResNet-18 training flow.

Synthetic data only: the AUROC scores are not meaningful ECG results.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import PTBXLResNet18

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
SIGNAL_LENGTH = 250


def make_dataset(size: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    samples = []

    for i in range(size):
        label_index = i % len(LABELS)
        labels = [LABELS[label_index]]
        if i % 2 == 0:
            labels.append(LABELS[(label_index + 1) % len(LABELS)])

        signal = torch.randn(
            12, SIGNAL_LENGTH, generator=generator
        ).tolist()

        samples.append({
            "patient_id": f"synthetic-{seed}-{i}",
            "signal": signal,
            "labels": labels,
        })

    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"labels": "multilabel"},
    )


def evaluate(model, dataset):
    model.eval()
    truths, probabilities = [], []

    with torch.no_grad():
        for batch in get_dataloader(dataset, batch_size=5, shuffle=False):
            result = model(**batch)
            truths.append(result["y_true"].cpu().numpy())
            probabilities.append(result["y_prob"].cpu().numpy())

    y_true = np.concatenate(truths)
    y_prob = np.concatenate(probabilities)
    return float(roc_auc_score(y_true, y_prob, average="macro"))


def main():
    torch.manual_seed(0)
    train_data = make_dataset(20, seed=1)
    val_data = make_dataset(10, seed=2)
    test_data = make_dataset(10, seed=3)

    model = PTBXLResNet18(
        train_data,
        base_width=4,
        layers=(1, 1, 1, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for batch in get_dataloader(train_data, batch_size=5, shuffle=True):
        optimizer.zero_grad()
        result = model(**batch)
        result["loss"].backward()
        optimizer.step()

    print("One training epoch completed.")
    print(f"Synthetic validation macro AUROC: {evaluate(model, val_data):.3f}")
    print(f"Synthetic test macro AUROC:       {evaluate(model, test_data):.3f}")
    print("Smoke test passed; synthetic scores are not research results.")


if __name__ == "__main__":
    main()
