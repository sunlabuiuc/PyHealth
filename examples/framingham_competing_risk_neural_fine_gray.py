"""Neural Fine-Gray full-pipeline example and CDF parameterization ablation.

The default run is fully offline and uses synthetic competing-risk data. Pass
``--data-root`` to use the public Framingham cohort through
``FraminghamDataset``. The script trains the paper parameterization and the
SumoNet-style no-time-factor ablation, then reports cause-1 concordance and
IPCW Brier scores at the 25th, 50th, and 75th percentiles of training event
times.

Example:
    python examples/framingham_competing_risk_neural_fine_gray.py \
        --epochs 20 --samples 600

Framingham:
    python examples/framingham_competing_risk_neural_fine_gray.py \
        --data-root /path/to/framingham --download --epochs 100
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass

import numpy as np
import torch

from pyhealth.datasets import (
    FraminghamDataset,
    PatientSplit,
    SampleDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.metrics import (
    competing_risk_brier_score,
    competing_risk_concordance_index,
)
from pyhealth.models import NeuralFineGray
from pyhealth.tasks import FraminghamCompetingRiskPrediction


@dataclass
class DataBundle:
    """Training, validation, and test sample datasets."""

    train: SampleDataset
    valid: SampleDataset
    test: SampleDataset


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_dataset(
    features: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    indices: np.ndarray,
    name: str,
) -> SampleDataset:
    return create_sample_dataset(
        samples=[
            {
                "patient_id": f"synthetic-{index}",
                "features": features[index].tolist(),
                "duration": [float(duration[index])],
                "event": [int(event[index])],
            }
            for index in indices
        ],
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name=name,
    )


def synthetic_data(samples: int, seed: int) -> DataBundle:
    """Generate independent latent times for two competing causes."""
    if samples < 60:
        raise ValueError("samples must be at least 60")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(samples, 8)).astype(np.float32)
    rate_one = np.exp(-0.3 + 0.65 * features[:, 0] - 0.25 * features[:, 1])
    rate_two = np.exp(-0.5 - 0.45 * features[:, 0] + 0.35 * features[:, 2])
    time_one = rng.exponential(1.0 / rate_one)
    time_two = rng.exponential(1.0 / rate_two)
    censor_time = rng.exponential(3.0, size=samples)
    stacked = np.column_stack([censor_time, time_one, time_two])
    event = np.argmin(stacked, axis=1).astype(int)
    duration = stacked[np.arange(samples), event] + 0.01

    order = rng.permutation(samples)
    train_end, valid_end = int(0.7 * samples), int(0.85 * samples)
    train_idx = order[:train_end]
    valid_idx = order[train_end:valid_end]
    test_idx = order[valid_end:]
    mean = features[train_idx].mean(axis=0)
    scale = features[train_idx].std(axis=0)
    features = (features - mean) / np.where(scale > 0, scale, 1.0)
    return DataBundle(
        _sample_dataset(features, duration, event, train_idx, "nfg_train"),
        _sample_dataset(features, duration, event, valid_idx, "nfg_valid"),
        _sample_dataset(features, duration, event, test_idx, "nfg_test"),
    )


def framingham_data(root: str, download: bool, seed: int) -> DataBundle:
    """Load Framingham and fit feature scaling on training participants."""
    dataset = FraminghamDataset(root=root, download=download)
    train, valid, test = dataset.set_task(
        FraminghamCompetingRiskPrediction(),
        split=PatientSplit(ratios=(0.7, 0.15, 0.15), seed=seed),
    )
    return DataBundle(train, valid, test)


def _loss(model: NeuralFineGray, dataset: SampleDataset, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in get_dataloader(dataset, batch_size=batch_size, shuffle=False):
        value = float(model(**batch)["loss"].detach())
        size = int(batch["duration"].shape[0])
        total += value * size
        count += size
    return total / count


def train_model(
    bundle: DataBundle,
    cdf_mode: str,
    epochs: int,
    batch_size: int,
    seed: int,
) -> NeuralFineGray:
    """Train with validation-loss checkpointing."""
    seed_everything(seed)
    train_duration, _ = _collect_labels(bundle.train)
    time_scale = float(train_duration.max())
    model = NeuralFineGray(
        bundle.train,
        num_risks=2,
        hidden_dims=(32, 32),
        monotonic_hidden_dims=(32, 32),
        dropout=0.25,
        cdf_mode=cdf_mode,
        time_scale=time_scale,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        loader = get_dataloader(
            bundle.train,
            batch_size=batch_size,
            shuffle=True,
        )
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
        valid_loss = _loss(model, bundle.valid, batch_size)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"{cdf_mode:7s} epoch={epoch + 1:03d} "
                f"validation_nll={valid_loss:.4f}"
            )
    model.load_state_dict(best_state)
    model.eval()
    return model


def _collect_labels(dataset: SampleDataset) -> tuple[np.ndarray, np.ndarray]:
    duration, event = [], []
    for batch in get_dataloader(dataset, batch_size=256, shuffle=False):
        duration.extend(batch["duration"].reshape(-1).numpy())
        event.extend(batch["event"].reshape(-1).numpy())
    return np.asarray(duration), np.asarray(event, dtype=int)


def evaluate(
    model: NeuralFineGray,
    train: SampleDataset,
    test: SampleDataset,
) -> dict[str, float]:
    """Evaluate the CVD/first-cause CIF at three event-time quantiles."""
    train_duration, train_event = _collect_labels(train)
    event_times = train_duration[train_event > 0]
    horizons = np.quantile(event_times, [0.25, 0.5, 0.75])
    test_duration, test_event = _collect_labels(test)
    predictions = []
    for batch in get_dataloader(test, batch_size=256, shuffle=False):
        features = batch["features"]
        if isinstance(features, tuple):
            features = features[0]
        predictions.append(
            model.predict_cif(features, torch.as_tensor(horizons))[:, 0].numpy()
        )
    cif = np.concatenate(predictions, axis=0)
    result: dict[str, float] = {"test_nll": _loss(model, test, 256)}
    for index, (label, horizon) in enumerate(zip(("q25", "q50", "q75"), horizons)):
        result[f"{label}_time"] = float(horizon)
        result[f"{label}_c_index"] = competing_risk_concordance_index(
            test_duration,
            test_event,
            cif[:, index],
            float(horizon),
            cause=1,
        )
        result[f"{label}_brier"] = competing_risk_brier_score(
            test_duration,
            test_event,
            cif[:, index],
            float(horizon),
            cause=1,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", help="Framingham directory; synthetic if omitted"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    bundle = (
        framingham_data(args.data_root, args.download, args.seed)
        if args.data_root
        else synthetic_data(args.samples, args.seed)
    )
    results = {}
    for cdf_mode in ("nfg", "sumonet"):
        model = train_model(
            bundle,
            cdf_mode,
            args.epochs,
            args.batch_size,
            args.seed,
        )
        results[cdf_mode] = evaluate(model, bundle.train, bundle.test)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
