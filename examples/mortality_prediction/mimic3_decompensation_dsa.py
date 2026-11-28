"""Runnable DSA decompensation example with a lightweight synthetic ablation.

This example demonstrates the PyHealth implementation contributed for
"Dynamic Survival Analysis for Early Event Prediction" (Yeche et al., 2024).
It uses synthetic samples by default so that the complete example can run
without access to protected MIMIC-III data.

The example compares three configurations at the 24-hour prediction horizon:

1. Truncated DSA likelihood.
2. Truncated DSA with hazard-prevalence bias initialization.
3. Truncated DSA with bias initialization and survTLS (lengthscale=8).

The synthetic experiment is intended only as a runnable API demonstration;
its metrics should not be interpreted as reproduction results.

Compute-constrained MIMIC-III reproduction
------------------------------------------
Using a fixed 100,000/20,000/50,000 train/validation/test timestep subset,
seed 42, a 2-layer GRU with hidden dimension 128, and the same training
budget across variants, the observed test timestep AuPRCs were:

    Vanilla DSA                         0.0196
    + hazard bias initialization       0.2062
    + truncated likelihood             0.2145
    + survTLS                          0.2177
    EEP baseline                       0.2159

These experiments use a reduced compute budget relative to the original
paper and are therefore intended to reproduce the qualitative ablation
trend rather than the paper's absolute reported values.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DSA
from pyhealth.tasks import DecompensationMIMIC3
from pyhealth.trainer import Trainer


SEED = 42
HORIZON = 24


def _record(steps: int, width: int, rng: np.random.Generator) -> dict:
    """Creates one synthetic dynamic sequence."""
    return {
        "time": [float(i + 1) for i in range(steps)],
        "values": rng.normal(size=(steps, width)).tolist(),
    }


def build_synthetic_samples(
    task: DecompensationMIMIC3,
    n_samples: int = 48,
) -> list[dict]:
    """Builds deterministic synthetic DSA samples using the task schema."""
    rng = np.random.default_rng(SEED)
    samples = []

    for i in range(n_samples):
        steps = 5 + (i % 6)
        positive = i % 3 == 0

        hazard_label = [0.0] * HORIZON
        hazard_mask = [1.0] * HORIZON

        if positive:
            event_bin = 6 + (i % 13)
            event_index = event_bin - 1

            if task.survtls_lengthscale is None:
                hazard_label[event_index] = 1.0
            else:
                hazard_label = task._smooth_survival_label(event_bin)

            for k in range(event_bin, HORIZON):
                hazard_mask[k] = 0.0

        samples.append(
            {
                "patient_id": f"synthetic-{i:03d}",
                "dynamic_continuous": _record(
                    steps,
                    task.N_CONTINUOUS,
                    rng,
                ),
                "dynamic_categorical": _record(
                    steps,
                    task.N_CATEGORICAL,
                    rng,
                ),
                "dynamic_mask": {
                    "time": [float(j + 1) for j in range(steps)],
                    "values": np.ones(
                        (steps, task.N_MASKS),
                        dtype=float,
                    ).tolist(),
                },
                "hazard_label": hazard_label,
                "hazard_mask": hazard_mask,
                "y_true": int(positive),
            }
        )

    return samples


def make_dataset(samples: list[dict], name: str):
    """Creates a PyHealth sample dataset from synthetic DSA samples."""
    task = DecompensationMIMIC3()
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=name,
    )


def hazard_prevalence(samples: list[dict]) -> np.ndarray:
    """Estimates per-bin hazard prevalence for bias initialization."""
    labels = np.asarray(
        [sample["hazard_label"] for sample in samples],
        dtype=np.float32,
    )
    masks = np.asarray(
        [sample["hazard_mask"] for sample in samples],
        dtype=np.float32,
    )

    numerator = labels.sum(axis=0)
    denominator = masks.sum(axis=0)

    return np.divide(
        numerator,
        denominator,
        out=np.zeros(HORIZON, dtype=np.float32),
        where=denominator > 0,
    )


@torch.no_grad()
def evaluate_auprc(
    model: DSA,
    dataloader,
) -> float:
    """Evaluates fixed-horizon binary AuPRC."""
    model.eval()
    y_true = []
    y_prob = []

    for batch in dataloader:
        output = model(**batch)
        y_true.append(
            output["y_true"].cpu().numpy().reshape(-1)
        )
        y_prob.append(
            output["y_prob"].cpu().numpy().reshape(-1)
        )

    return float(
        average_precision_score(
            np.concatenate(y_true),
            np.concatenate(y_prob),
        )
    )


def run_variant(
    name: str,
    use_bias: bool,
    use_survtls: bool,
) -> float:
    """Trains one lightweight synthetic DSA ablation configuration."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    task = DecompensationMIMIC3(
        horizon_hours=HORIZON,
        min_observation_hours=5,
        survtls_lengthscale=8.0 if use_survtls else None,
    )
    samples = build_synthetic_samples(task)

    train_samples = samples[:32]
    val_samples = samples[32:40]
    test_samples = samples[40:]

    train_dataset = make_dataset(
        train_samples,
        f"{name}_train",
    )
    val_dataset = make_dataset(
        val_samples,
        f"{name}_val",
    )
    test_dataset = make_dataset(
        test_samples,
        f"{name}_test",
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=8,
        shuffle=False,
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    prevalence = (
        hazard_prevalence(
            build_synthetic_samples(
                DecompensationMIMIC3(
                    horizon_hours=HORIZON,
                    min_observation_hours=5,
                ),
                n_samples=32,
            )
        )
        if use_bias
        else None
    )

    model = DSA(
        dataset=train_dataset,
        hidden_dim=16,
        max_horizon=HORIZON,
        prediction_horizon=HORIZON - 1,
        num_layers=1,
        dropout=0.0,
        hazard_prevalence=prevalence,
    )

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
    )

    auprc = evaluate_auprc(model, test_loader)
    print(f"{name:24s} synthetic AuPRC = {auprc:.4f}")
    return auprc


if __name__ == "__main__":
    run_variant(
        "truncated",
        use_bias=False,
        use_survtls=False,
    )
    run_variant(
        "truncated+bias",
        use_bias=True,
        use_survtls=False,
    )
    run_variant(
        "truncated+bias+survTLS",
        use_bias=True,
        use_survtls=True,
    )
