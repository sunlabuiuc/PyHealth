"""Synthetic missingness ablation for the hourly LOS task.

Contributor: Xiao Luo; UIUC NetID: xiaoluo6.
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
in the Intensive Care Unit, https://arxiv.org/abs/2007.09483.

Run from the repository root:
    python -m examples.eicu_hourly_los_lstm --output /tmp/tpc_ablation.json

Five invented patients; 101 trains, 103/105 test, 102/104 are excluded.
Two LSTMs train on 0%/10% removed hourly cells; both evaluate at 0%/10%/20%.
Shared initialization and test masks; processors fit on each training variant.
Default MSE on train-scaled targets, not the paper's logarithmic loss.
MAD in days; log1p-MSE in days is explicitly not the paper's log-MSE metric.
This runnable example checks experimental wiring only; clinical results and
novelty claims require A's real-data experiment and team review.
"""

import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from pyhealth.datasets import eICUDataset, create_sample_dataset, get_dataloader
from pyhealth.models import RNN_V2
from pyhealth.tasks import TPCLengthOfStayEICU
from examples.tpc_support.generate_synthetic import TABLES, generate


def run(steps: int = 3) -> list[dict]:
    """Run the six synthetic configurations and return measured metrics."""
    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    results = []
    with tempfile.TemporaryDirectory(prefix="tpc-ablation-") as temporary:
        root = Path(temporary)
        generate(root / "csv")
        data = eICUDataset(
            root=str(root / "csv"), tables=TABLES, cache_dir=root / "cache"
        )
        patients = {
            sid: data.get_patient(f"synthetic-{sid}") for sid in (101, 103, 105)
        }
        for train_rate in (0.0, 0.1):
            torch.manual_seed(42)
            task = TPCLengthOfStayEICU(train_rate, seed=42, include_decay=True)
            train = create_sample_dataset(
                task(patients[101]), task.input_schema, task.output_schema
            )
            model = RNN_V2(
                train,
                embedding_dim={
                    "dynamic": [{"values": 8}],
                    "static": [{"values": 4}],
                    "decay": [{"values": 8}],
                },
                hidden_dim=8,
                rnn_type="LSTM",
                mask="concat",
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            batch = next(iter(get_dataloader(train, batch_size=1, shuffle=False)))
            for _ in range(steps):
                optimizer.zero_grad()
                loss = model(**batch)["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
            model.eval()
            for test_rate in (0.0, 0.1, 0.2):
                evaluation_task = TPCLengthOfStayEICU(
                    test_rate, seed=42, include_decay=True
                )
                samples = [evaluation_task(patients[sid])[0] for sid in (103, 105)]
                evaluation = create_sample_dataset(
                    samples,
                    task.input_schema,
                    task.output_schema,
                    input_processors=train.input_processors,
                    output_processors=train.output_processors,
                )
                with torch.no_grad():
                    out = model(
                        **next(
                            iter(
                                get_dataloader(evaluation, batch_size=2, shuffle=False)
                            )
                        )
                    )
                # activation() reverses label normalization; y_prob is in hours.
                pred = out["y_prob"].numpy().reshape(-1) / 24
                truth = (
                    np.concatenate(
                        [np.asarray(s["los"]["values"]).ravel() for s in samples]
                    )
                    / 24
                )
                if not np.isfinite(pred).all() or pred.shape != truth.shape:
                    raise RuntimeError("Invalid predictions")
                results.append(
                    {
                        "train_missing_rate": train_rate,
                        "test_missing_rate": test_rate,
                        "mad_days": float(np.abs(pred - truth).mean()),
                        "log1p_mse_days": float(
                            (
                                (np.log1p(np.maximum(pred, 0)) - np.log1p(truth)) ** 2
                            ).mean()
                        ),
                        "negative_predictions_clipped_for_log1p": int((pred < 0).sum()),
                        "prediction_count": len(truth),
                        "steps": steps,
                        "seed": 42,
                        "data": "synthetic; not evidence of clinical performance",
                    }
                )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = run(args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
