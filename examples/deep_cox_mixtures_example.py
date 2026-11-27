"""Runnable synthetic example for Deep Cox Mixtures.

Demonstrates approximate EM training and survival prediction
on synthetic data with heterogeneous survival distributions.

Run:
    python examples/deep_cox_mixtures_example.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from pyhealth.models import DeepCoxMixtures


def generate_synthetic_data(
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate heterogeneous synthetic survival data.

    Args:
        n_samples: Number of synthetic patients.
        seed: Random seed.

    Returns:
        Features, observed durations, event indicators,
        and true latent subgroup assignments.
    """
    rng = np.random.default_rng(seed)

    features = rng.normal(size=(n_samples, 4)).astype(np.float32)

    # Generate feature-dependent latent subgroup probabilities.
    logits = np.column_stack(
        [
            features[:, 0],
            features[:, 1],
            -features[:, 0] - features[:, 1],
        ]
    )

    logits -= logits.max(axis=1, keepdims=True)

    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    groups = np.array([rng.choice(3, p=p) for p in probabilities])

    # Component-specific baseline hazard rates.
    baseline_rates = np.array([0.003, 0.008, 0.015])

    # Component-specific feature effects.
    coefficients = np.array(
        [
            [0.8, -0.4, 0.2, 0.1],
            [-0.5, 0.9, -0.3, 0.2],
            [0.2, -0.6, 0.8, -0.4],
        ]
    )

    log_risk = np.sum(
        features * coefficients[groups],
        axis=1,
    )

    rates = baseline_rates[groups] * np.exp(log_risk)

    event_times = rng.exponential(scale=1.0 / rates)

    censoring_times = rng.exponential(
        scale=400.0,
        size=n_samples,
    )

    durations = np.minimum(
        event_times,
        censoring_times,
    )

    events = (event_times <= censoring_times).astype(np.float32)

    durations = np.maximum(durations, 1e-3).astype(np.float32)

    return features, durations, events, groups


def main() -> None:
    """Train and evaluate DCM on synthetic survival data."""
    torch.manual_seed(42)

    X, T, E, groups = generate_synthetic_data()

    print("\nSynthetic Survival Dataset")
    print("-" * 50)
    print("Patients:", len(X))
    print("Features:", X.shape[1])
    print("Observed events:", int(E.sum()))
    print("Censored patients:", int((E == 0).sum()))
    print("True subgroup sizes:", np.bincount(groups))

    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=42,
        stratify=E,
    )

    X_train = torch.tensor(X[train_idx])
    T_train = torch.tensor(T[train_idx])
    E_train = torch.tensor(E[train_idx])

    X_test = torch.tensor(X[test_idx])
    T_test = T[test_idx]
    E_test = E[test_idx]

    model = DeepCoxMixtures(
        dataset=None,
        input_dim=4,
        num_components=3,
        hidden_dims=(32, 16),
    )

    print("\nTraining Deep Cox Mixtures")

    history = model.fit_em(
        features=X_train,
        duration=T_train,
        event=E_train,
        epochs=100,
        learning_rate=0.001,
        seed=42,
    )

    horizons = np.linspace(0, 365, 100)

    predictions = (
        model.predict_survival(
            X_test,
            horizons,
        )
        .cpu()
        .numpy()
    )

    print("\nPrediction shape:", predictions.shape)

    print(
        "Valid probabilities:",
        bool(np.all(predictions >= 0) and np.all(predictions <= 1)),
    )

    print(
        "Monotonic survival:",
        bool(np.all(np.diff(predictions, axis=1) <= 1e-6)),
    )

    # Evaluate risk discrimination at a fixed horizon.
    evaluation_time = 180.0

    survival_180 = (
        model.predict_survival(
            X_test,
            [evaluation_time],
        )
        .cpu()
        .numpy()
        .reshape(-1)
    )

    risk_180 = 1.0 - survival_180

    c_index = concordance_index_censored(
        E_test.astype(bool),
        T_test,
        risk_180,
    )[0]

    print(f"Concordance using 180-day risk: {c_index:.4f}")

    # Save plots and trained model.
    output_dir = Path("results/dcm_synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        np.arange(1, len(history) + 1),
        history,
        linewidth=1.5,
    )

    axes[0].set_title("DCM Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Approximate EM Loss")
    axes[0].grid(alpha=0.3)

    for i in range(5):
        axes[1].plot(
            horizons,
            predictions[i],
            label=f"Patient {i + 1}",
        )

    axes[1].set_title("Predicted Survival Curves")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Survival Probability")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "synthetic_results.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    torch.save(
        model.state_dict(),
        output_dir / "dcm_model.pt",
    )

    print("\nResults saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
