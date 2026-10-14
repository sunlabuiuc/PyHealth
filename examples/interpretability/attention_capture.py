"""Run gradient-free Rollout and gradient-based Chefer on one synthetic model.

Run: pixi run -e test python examples/interpretability/attention_capture.py
"""

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import AttentionRollout, CheferRelevance
from pyhealth.models import Transformer


def main():
    torch.manual_seed(42)
    dataset = create_sample_dataset(
        samples=[
            {"patient_id": "p0", "visit_id": "v0", "codes": ["A", "B"], "label": 1},
            {"patient_id": "p1", "visit_id": "v1", "codes": ["B"], "label": 0},
        ],
        input_schema={"codes": "sequence"},
        output_schema={"label": "binary"},
    )
    model = Transformer(dataset=dataset, embedding_dim=8, heads=2, num_layers=2)
    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))

    with torch.no_grad():
        rollout = AttentionRollout(model).attribute(**batch)
    for attention, gradient in model.get_attention_layers()["codes"]:
        assert attention is not None and not attention.requires_grad
        assert gradient is None

    # Complete each interpretation before starting the next captured forward.
    chefer = CheferRelevance(model).attribute(**batch)
    for attention, gradient in model.get_attention_layers()["codes"]:
        assert attention is not None and gradient is not None
        assert attention.shape == gradient.shape

    print("Rollout:", rollout["codes"].tolist())
    print("Chefer:", chefer["codes"].tolist())


if __name__ == "__main__":
    main()
