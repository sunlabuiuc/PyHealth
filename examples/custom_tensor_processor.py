"""A custom tensor processor that changes the feature width, used with an MLP.

Subclass ``TensorProcessor`` to transform numeric features, here keeping a
subset of columns. Models size their input layer from the already-processed
samples (or from ``size()``), and never call ``process()`` a second time, so a
processor like this one, which is not idempotent, works with any model.

Runs in a few seconds on CPU with synthetic data; no download.

Usage:
    python examples/custom_tensor_processor.py
"""

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.processors import TensorProcessor


class SelectColumns(TensorProcessor):
    """Keeps the given columns of a numeric feature vector."""

    def __init__(self, columns):
        super().__init__()
        self.columns = list(columns)

    def process(self, value):
        return super().process(value)[..., self.columns]

    def size(self):
        return len(self.columns)


def main():
    torch.manual_seed(0)
    samples = [
        {
            "patient_id": f"p{i}",
            # age, heart rate, temperature, a column we want to drop
            "vitals": [40.0 + i, 70.0 + i % 9, 36.5 + (i % 3) / 10, 999.0],
            "label": int(i % 3 == 0),
        }
        for i in range(30)
    ]
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"vitals": SelectColumns(columns=[0, 1, 2])},
        output_schema={"label": "binary"},
        dataset_name="custom_tensor_processor",
    )
    print("processed sample:", dataset[0]["vitals"])

    model = MLP(dataset=dataset)
    print("input layer:", model.embedding_model.embedding_layers["vitals"])

    batch = next(iter(get_dataloader(dataset, batch_size=8)))
    out = model(**batch)
    print("y_prob shape:", tuple(out["y_prob"].shape), "loss:", round(float(out["loss"]), 4))


if __name__ == "__main__":
    main()
