"""Synthetic mortality-style example for :class:`~pyhealth.models.TransEHR`.

This script shows how to instantiate and train the simplified TransEHR model
on small synthetic ICU-style samples. It also includes a small ablation that
compares the full dual-stream model against a version with the event stream
disabled.

This script does not download MIMIC-III or reproduce the full benchmark from
the original paper. It is intended as a lightweight runnable example for the
PyHealth contribution.

Run::

    python examples/mimic3_mortality_transehr.py
    python examples/mimic3_mortality_transehr.py --steps 30 --seed 0
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TransEHR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_demo_dataset(n_samples: int = 32):
    """Synthetic ICU-style samples: irregular labs/vitals, diagnosis codes, age.

    Returns:
        Tuple of ``(dataset, dataloader)`` for ``InMemorySampleDataset`` from
        :func:`~pyhealth.datasets.create_sample_dataset`.
    """
    rng = np.random.default_rng(0)
    t0 = datetime(2020, 1, 1, 8, 0)
    codes_pool = [
        "DX_SEPSIS",
        "DX_PNEUMONIA",
        "DX_CHF",
        "DX_AKI",
        "PROC_VENT",
        "MED_VASO",
    ]
    samples = []
    for i in range(n_samples):
        n_steps = int(rng.integers(2, 6))
        # TemporalTimeseriesProcessor expects raw input as
        # (list[datetime], ndarray[T, F]).
        hours = rng.uniform(0.0, 24.0, size=n_steps)
        order = np.argsort(hours)
        times = [t0 + timedelta(hours=float(hours[j])) for j in order]
        values = rng.normal(0.0, 1.0, size=(n_steps, 4)).astype(np.float32)[order]
        n_evt = int(rng.integers(1, 5))
        events = list(rng.choice(codes_pool, size=n_evt, replace=True))
        age = float(rng.uniform(40.0, 90.0))
        static = [age, float(rng.choice([0.0, 1.0]))]
        label = int(rng.choice([0, 1]))
        samples.append(
            {
                "patient_id": f"syn-{i}",
                "visit_id": f"icu-{i}",
                "multivariate": (times, values),
                "events": events,
                "static": static,
                "mortality": label,
            }
        )

    ds = create_sample_dataset(
        samples=samples,
        input_schema={
            "multivariate": "temporal_timeseries",
            "events": "sequence",
            "static": "tensor",
        },
        output_schema={"mortality": "binary"},
        dataset_name="mimic3_mortality_transehr_demo",
    )
    loader = get_dataloader(ds, batch_size=8, shuffle=True)
    return ds, loader


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor leaves to ``device``; leave lists / dicts for model unpack logic."""
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                sk: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for sk, t in v.items()
            }
        else:
            out[k] = v
    return out


def train_short_run(
    *,
    dataset,
    loader,
    use_event_stream: bool,
    hidden_dim: int,
    steps: int,
    device: torch.device,
    lr: float,
) -> float:
    """Return mean loss over the last up-to-5 steps."""
    model = TransEHR(
        dataset=dataset,
        feature_keys={
            "multivariate": "multivariate",
            "events": "events",
            "static": "static",
        },
        label_key="mortality",
        mode="binary",
        embedding_dim=64,
        hidden_dim=hidden_dim,
        num_heads=4,
        dropout=0.1,
        num_encoder_layers=1,
        max_event_len=64,
        max_ts_len=128,
        use_event_stream=use_event_stream,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    it = iter(loader)
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = _batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        out = model(**batch)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].detach().cpu()))
    tail = losses[-5:] if losses else [0.0]
    return sum(tail) / len(tail)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--steps",
        type=int,
        default=40,
        help="optimizer steps per ablation arm",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu", action="store_true", help="force CPU")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    ds, loader = build_demo_dataset()

    print(
        "Synthetic MIMIC-style mortality demo + structural/hparam ablation"
    )
    print(f"device={device}, steps_per_arm={args.steps}, lr={args.lr}\n")

    loss_with = train_short_run(
        dataset=ds,
        loader=loader,
        use_event_stream=True,
        hidden_dim=32,
        steps=args.steps,
        device=device,
        lr=args.lr,
    )
    set_seed(args.seed)
    loss_with_h64 = train_short_run(
        dataset=ds,
        loader=loader,
        use_event_stream=True,
        hidden_dim=64,
        steps=args.steps,
        device=device,
        lr=args.lr,
    )
    set_seed(args.seed)
    loss_without = train_short_run(
        dataset=ds,
        loader=loader,
        use_event_stream=False,
        hidden_dim=64,
        steps=args.steps,
        device=device,
        lr=args.lr,
    )

    print(f"use_event_stream=True, hidden_dim=32  -> mean last-5 loss: {loss_with:.4f}")
    print(
        f"use_event_stream=True, hidden_dim=64  -> mean last-5 loss: "
        f"{loss_with_h64:.4f}"
    )
    print(
        f"use_event_stream=False, hidden_dim=64 -> mean last-5 loss: "
        f"{loss_without:.4f}"
    )
    print(
        "\nInterpretation: on this toy setup, numbers are not benchmarks; the "
        "point is to show both structural and tiny hyperparameter ablations. "
        "Compare AUC / calibration on real MIMIC after wiring the dataset."
    )


if __name__ == "__main__":
    main()
