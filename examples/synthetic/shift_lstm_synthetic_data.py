"""Synthetic data generation for shiftLSTM ablation experiments.

This script follows Section 4.1 of:
"Relaxed Parameter Sharing: Effectively Modeling Time-Varying Relationships
in Clinical Time-Series" (Oh et al., 2019).

It also borrows the high-level structure of the authors' released synthetic
data generator that maintains:
  - sparse input sequences X in R^{N x T x d}
  - time-varying temporal weights over the previous l timesteps
  - time-varying feature weights over d dimensions
  - smooth drift controlled by delta

Compared with the original prototype-style script, this version:
  - is CPU-friendly
  - is reproducible through a random seed
  - can export NumPy arrays
  - can convert synthetic arrays into PyHealth SampleDataset-ready samples
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class SyntheticConfig:
    """Configuration for the synthetic generator.

    Args:
        N: Number of samples.
        T: Sequence length.
        d: Number of input features.
        l: Lookback window used to generate targets. The paper uses l=10.
        delta: Drift magnitude controlling how much the temporal/feature
            distributions can change between adjacent timesteps.
        sparsity: Probability that an input entry is non-zero.
        value_low: Lower bound for non-zero values.
        value_high: Upper bound for non-zero values.
        seed: Random seed for reproducibility.
    """

    N: int = 1000
    T: int = 30
    d: int = 3
    l: int = 10
    delta: float = 0.2
    sparsity: float = 0.1
    value_low: float = 0.0
    value_high: float = 100.0
    seed: int = 42

    def validate(self) -> None:
        if self.T <= self.l:
            raise ValueError("T must be greater than l for target generation.")
        if not 0.0 < self.sparsity <= 1.0:
            raise ValueError("sparsity must be in (0, 1].")
        if self.delta < 0:
            raise ValueError("delta must be non-negative.")
        if self.value_high <= self.value_low:
            raise ValueError("value_high must be greater than value_low.")


def normalize_to_distribution(values: np.ndarray) -> np.ndarray:
    """Projects a vector to a probability distribution.

    The authors' script first min-max normalizes, then renormalizes to sum to 1.
    We retain that behavior but add guards for constant vectors.
    """

    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        return np.full_like(values, fill_value=1.0 / len(values), dtype=float)

    scaled = (values - vmin) / (vmax - vmin)
    total = float(np.sum(scaled))
    if np.isclose(total, 0.0):
        return np.full_like(values, fill_value=1.0 / len(values), dtype=float)
    return scaled / total


def generate_sparse_inputs(config: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Generates sparse inputs X of shape (N, T, d).

    Following the paper, each entry is active with probability 0.1 by default,
    and active values are sampled uniformly on [0, 100].
    """

    active = rng.binomial(1, config.sparsity, size=(config.N, config.T, config.d))
    values = rng.uniform(
        low=config.value_low,
        high=config.value_high,
        size=(config.N, config.T, config.d),
    )
    return active * values


def generate_weight_trajectories(
    config: SyntheticConfig,
    rng: np.random.Generator,
    k_dist: Optional[list[np.ndarray]] = None,
    d_dist: Optional[list[np.ndarray]] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generates time-varying temporal and feature distributions.

    At timestep t >= l:
      - k_dist[t] is a distribution over the previous l timesteps
      - d_dist[t] is a distribution over the d feature dimensions

    Distances evolve smoothly with additive perturbations bounded by delta.
    """

    if k_dist is not None and d_dist is not None:
        return k_dist, d_dist

    temporal_weights: list[np.ndarray] = []
    feature_weights: list[np.ndarray] = []

    for t in range(config.T):
        if t < config.l:
            temporal_weights.append(np.ones(config.l, dtype=float) / config.l)
            feature_weights.append(np.ones(config.d, dtype=float) / config.d)
            continue

        if t == config.l:
            temporal_weights.append(
                normalize_to_distribution(rng.uniform(size=config.l))
            )
            feature_weights.append(
                normalize_to_distribution(rng.uniform(size=config.d))
            )
            continue

        delta_t = rng.uniform(-config.delta, config.delta, size=config.l)
        delta_d = rng.uniform(-config.delta, config.delta, size=config.d)
        temporal_weights.append(
            normalize_to_distribution(temporal_weights[t - 1] + delta_t)
        )
        feature_weights.append(
            normalize_to_distribution(feature_weights[t - 1] + delta_d)
        )

    return temporal_weights, feature_weights


def generate_targets(
    x: np.ndarray,
    config: SyntheticConfig,
    k_dist: list[np.ndarray],
    d_dist: list[np.ndarray],
) -> np.ndarray:
    """Generates regression targets Y of shape (N, T, 1).

    For timestep t >= l, the target is formed by:
      1. combining feature dimensions at each previous timestep using d_dist[t]
      2. combining the resulting l-length history using k_dist[t]
    """

    y = np.ones((config.N, config.T, 1), dtype=float)
    for t in range(config.l, config.T):
        history = x[:, t - config.l : t, :]  # (N, l, d)
        # Use explicit weighted sums instead of np.matmul to avoid spurious
        # BLAS-level overflow warnings on some local NumPy builds.
        feature_agg = np.einsum("nld,d->nl", history, d_dist[t])  # (N, l)
        y[:, t, 0] = np.einsum("nl,l->n", feature_agg, k_dist[t])  # (N,)
    return y


def generate_synthetic_arrays(
    config: SyntheticConfig,
    k_dist: Optional[list[np.ndarray]] = None,
    d_dist: Optional[list[np.ndarray]] = None,
) -> dict[str, Any]:
    """Generates a full synthetic dataset bundle.

    Returns:
        A dictionary containing X, Y, k_dist, d_dist, and config.
    """

    config.validate()
    rng = np.random.default_rng(config.seed)

    x = generate_sparse_inputs(config, rng)
    k_dist, d_dist = generate_weight_trajectories(config, rng, k_dist, d_dist)
    y = generate_targets(x, config, k_dist, d_dist)

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "k_dist": np.asarray(k_dist, dtype=np.float32),
        "d_dist": np.asarray(d_dist, dtype=np.float32),
        "config": asdict(config),
    }


def save_synthetic_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Saves the generated arrays and metadata to a compressed NPZ file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=bundle["x"],
        y=bundle["y"],
        k_dist=bundle["k_dist"],
        d_dist=bundle["d_dist"],
        config_json=json.dumps(bundle["config"]),
    )
    return output_path


def to_pyhealth_samples(
    x: np.ndarray,
    y: np.ndarray,
    task: str = "binary_final_step",
    start_time: Optional[datetime] = None,
    threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Converts arrays to raw Python samples for ``create_sample_dataset``.

    The default output is a sequence-level binary label derived from the final
    timestep target, which is convenient for shiftLSTM ablation studies.

    Each sample contains one timeseries feature:
      - "signal": (timestamps, values)
      - "label": scalar
    """

    if start_time is None:
        start_time = datetime(2020, 1, 1, 0, 0, 0)

    final_targets = y[:, -1, 0]
    if threshold is None:
        threshold = float(np.median(final_targets))

    timestamps = None
    samples: list[dict[str, Any]] = []
    for idx in range(x.shape[0]):
        if timestamps is None:
            timestamps = [start_time + timedelta(hours=t) for t in range(x.shape[1])]

        if task == "binary_final_step":
            label = int(final_targets[idx] > threshold)
        elif task == "regression_final_step":
            label = float(final_targets[idx])
        else:
            raise ValueError(f"Unsupported task: {task}")

        samples.append(
            {
                "patient_id": f"synthetic-patient-{idx}",
                "visit_id": f"synthetic-visit-{idx}",
                "signal": (timestamps, x[idx]),
                "label": label,
            }
        )
    return samples


def create_repeated_bundles(
    config: SyntheticConfig,
    synth_num: int,
    output_dir: str | Path,
    run_prefix: str = "synthetic_shift",
) -> list[Path]:
    """Creates multiple synthetic datasets with different random seeds."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for run in range(synth_num):
        run_config = SyntheticConfig(**{**asdict(config), "seed": config.seed + run})
        bundle = generate_synthetic_arrays(run_config)
        out_path = output_dir / f"{run_prefix}_model{run}.npz"
        save_synthetic_bundle(bundle, out_path)
        saved_paths.append(out_path)
    return saved_paths


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for shiftLSTM ablation experiments."
    )
    parser.add_argument("--N", type=int, default=1000, help="Number of samples.")
    parser.add_argument("--T", type=int, default=30, help="Sequence length.")
    parser.add_argument("--d", type=int, default=3, help="Number of features.")
    parser.add_argument(
        "--l", type=int, default=10, help="Lookback window used for target generation."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.2,
        help="Maximum drift per step for temporal/feature distributions.",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.1,
        help="Probability that an input entry is non-zero.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--synth-num",
        type=int,
        default=1,
        help="How many independent synthetic datasets to generate.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="examples/synthetic/generated",
        help="Directory for generated NPZ files.",
    )
    parser.add_argument(
        "--runname",
        type=str,
        default="synthetic_shift",
        help="Filename prefix for generated datasets.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = SyntheticConfig(
        N=args.N,
        T=args.T,
        d=args.d,
        l=args.l,
        delta=args.delta,
        sparsity=args.sparsity,
        seed=args.seed,
    )
    saved = create_repeated_bundles(
        config=config,
        synth_num=args.synth_num,
        output_dir=args.savedir,
        run_prefix=args.runname,
    )
    print(f"Generated {len(saved)} synthetic dataset(s):")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
