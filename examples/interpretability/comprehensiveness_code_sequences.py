"""Comprehensiveness and sufficiency for a model on code sequences.

Removal-based metrics ablate the code ids of ``sequence`` inputs by setting
them to padding. This example trains an MLP on a small synthetic dataset and
scores Integrated Gradients attributions; it runs in seconds on a CPU and needs
no downloads.

Usage:
    python examples/interpretability/comprehensiveness_code_sequences.py
"""

import random

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.metrics.interpretability import (
    evaluate_attribution,
    threshold_sample_filter,
)
from pyhealth.models import MLP
from pyhealth.trainer import Trainer


def main():
    random.seed(0)
    torch.manual_seed(0)
    # The label is 1 when code "risk" is present, so a faithful explanation
    # should point at it.
    samples = []
    for i in range(400):
        codes = random.sample([f"c{k}" for k in range(30)], 6)
        label = i % 2
        if label:
            codes[random.randrange(6)] = "risk"
        samples.append({"patient_id": f"p{i}", "codes": codes, "label": label})
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_codes",
    )
    loader = get_dataloader(dataset, batch_size=64, shuffle=True)

    model = MLP(dataset=dataset)
    Trainer(model=model, enable_logging=False).train(
        train_dataloader=loader, epochs=5
    )
    model.eval()

    scores = evaluate_attribution(
        model,
        get_dataloader(dataset, batch_size=64),
        IntegratedGradients(model, steps=16),
        percentages=[10, 20, 50],
        sample_filter=threshold_sample_filter(0.5),
    )
    print(scores)


if __name__ == "__main__":
    main()
