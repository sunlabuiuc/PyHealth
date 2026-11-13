"""Train, calibrate and score binary LABEL prediction sets on synthetic samples.

Run from the repository root: python -m examples.conformal_label_binary
"""

import numpy as np
import torch

from pyhealth.calib.predictionset import LABEL
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import MLP
from pyhealth.trainer import Trainer


def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(160, 3))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    samples = [
        {"patient_id": str(i), "x": x.tolist(), "label": int(y)}
        for i, (x, y) in enumerate(zip(features, labels))
    ]
    dataset = create_sample_dataset(samples, {"x": "tensor"}, {"label": "binary"})
    try:
        # Each patient appears in exactly one split.
        train = dataset.subset(list(range(96)))
        calibration = dataset.subset(list(range(96, 128)))
        test = dataset.subset(list(range(128, 160)))

        model = MLP(dataset=dataset, hidden_dim=16)
        trainer = Trainer(model=model, device="cpu", enable_logging=False)
        trainer.train(get_dataloader(train, batch_size=16, shuffle=True), epochs=2)
        test_loader = get_dataloader(test, batch_size=16, shuffle=False)

        for score_type in ("threshold", "aps"):
            predictor = LABEL(model, alpha=0.2, score_type=score_type, random_state=42)
            predictor.calibrate(calibration)
            y_true, y_prob, _, extra = Trainer(
                model=predictor, device="cpu", enable_logging=False
            ).inference(test_loader, additional_outputs=["y_predset"])
            scores = binary_metrics_fn(
                y_true,
                y_prob,
                metrics=["accuracy", "set_size", "miscoverage_overall_ps"],
                y_predset=extra["y_predset"],
            )
            print(f"LABEL ({score_type}): {scores}")
            print(
                f"Probability shape: {y_prob.shape}; set shape: {extra['y_predset'].shape}"
            )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
