import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.models.neural_fg import NeuralFG
from neuralfg.train import TrainConfig, run_training, stratified_split
from neuralfg.metrics import evaluate_neuralfg

from neuralfg.data import (
    load_data,
    prepare_pyhealth_samples,
    create_dataset,
)

RUN_FEATURE_SHUFFLING = False
RUN_SMALL_SUBSET_TEST = False  # Set True to recheck basic forward/backward integration.
RUN_TEST_METRICS = False  # Set True only for the final chosen model.

def test_small_subset(df):
    """Test NeuralFG integration using 100 synthetic samples."""

    samples = prepare_pyhealth_samples(
        df,
        max_samples=100,
    )

    dataset = create_dataset(samples)

    loader = get_dataloader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    model = NeuralFG(
        dataset=dataset,
        num_risks=2,
        hidden_dim=16,
        embedding_dim=16,
        dropout=0.0,
    )

    batch = next(iter(loader))
    result = model(**batch)

    print("\nSmall-subset test:")
    print("Loss:", result["loss"].item())
    print("y_prob shape:", result["y_prob"].shape)
    print("y_true shape:", result["y_true"].shape)
    print("observed_cif shape:", result["observed_cif"].shape)

    result["loss"].backward()
    print("Backward pass successful.")


if __name__ == "__main__":
    torch.manual_seed(42)

    df = load_data()

    # Examine event-time distributions
    print("\nObserved time summary:")
    print(df["time"].describe())

    print("\nUncensored event-time quantiles:")
    event_times = df.loc[df["label"] > 0, "time"]

    print(
        event_times.quantile(
            [0.25, 0.50, 0.75]
        )
    )

    print("\nEvent-time summary by risk:")
    print(
        df.groupby("label")["time"].describe()
    )

    # Optional integration smoke test
    if RUN_SMALL_SUBSET_TEST:
        test_small_subset(df)

    # NeuralFG training configuration
    config = TrainConfig(
        max_samples=None,
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        hidden_dim=50,
        embedding_dim=50,
        dropout=0.0,
        patience=20,
        seed=42,
    )

    # Check the configured early-stopping patience
    print("Configured patience:", config.patience)

    # Start training
    model, history, test_loss = run_training(
        df,
        config,
    )

    # ---------------------------------------------------------
    # Check NeuralFG predicted cumulative incidence functions
    # ---------------------------------------------------------

    # Select 100 patients for a diagnostic check
    feature_columns = [
        f"feature{i}" for i in range(1, 13)
    ]

    sample_df = df.sample(
        n=100,
        random_state=config.seed,
    )

    features = torch.tensor(
        sample_df[feature_columns].to_numpy(
            dtype="float32"
        )
    )

    # Evaluation times selected from the training subset
    times = model.time_grid.detach().cpu()

    # Shape: [100 patients, 3 times, 2 risks]
    cif = model.predict_cif(
        features=features,
        times=times,
    ).detach().cpu()

    print("\nCIF diagnostic results:")
    print("Evaluation times:", times.tolist())
    print("CIF shape:", cif.shape)

    print("\nMean CIF by time and risk:")
    print(cif.mean(dim=0))

    print("\nCIF standard deviation across patients:")
    print(cif.std(dim=0))

    print(
        "\nMaximum sum of CIFs:",
        cif.sum(dim=-1).max().item(),
    )

    print(
        "CIFs nondecreasing across evaluation times:",
        bool(
            (
                    cif[:, 1:, :] >= cif[:, :-1, :] - 1e-6
            ).all()
        ),
    )

    print("\nTraining history:")
    print(history)

    print("\nFinal test NLL:")
    print(test_loss)

    # ---------------------------------------------------------
    # Save training history and loss curve
    # ---------------------------------------------------------
    experiment_name = (
        f"n{config.max_samples if config.max_samples is not None else 'full'}"
        f"_epochs{config.epochs}"
        f"_hidden{config.hidden_dim}"
        f"_embed{config.embedding_dim}"
        f"_elayers{config.embedding_layers}"
        f"_mlayers{config.monotonic_layers}"
        f"_dropout{config.dropout}"
        f"_lr{config.learning_rate}"
        f"_wd{config.weight_decay}"
        f"_bs{config.batch_size}"
        f"_patience{config.patience}"
        f"_seed{config.seed}"
    )

    # Save all NeuralFG experiments in a dedicated folder
    RESULTS_DIR = (
            Path(__file__).resolve().parent
            / "neuralfg"
            / "results"
            / "neuralfg"  # New subfolder
            / experiment_name
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    # Save model and exact hyperparameters before any evaluation.
    model_path = RESULTS_DIR / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    (RESULTS_DIR / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    print(f"Best model saved to: {model_path}")

    # ---------------------------------------------------------
    # Reconstruct the deterministic splits from train.py
    # ---------------------------------------------------------

    # Reconstruct the exact data selection used in train.py.
    # This is important when max_samples is not None.
    if config.max_samples is None:
        experiment_df = df
    else:
        experiment_df = df.sample(
            n=min(config.max_samples, len(df)),
            random_state=config.seed,
        ).copy()

    # Use the same split settings and seed as training.
    train_df, val_df, test_df = stratified_split(
        experiment_df,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    print("\nEvaluation dataset sizes:")
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    # ---------------------------------------------------------
    # Validation metrics for selecting model configurations
    # ---------------------------------------------------------
    validation_metrics = evaluate_neuralfg(
        model=model, train_df=train_df, test_df=val_df, batch_size=512,
    )
    print("\nNeuralFG validation metrics:")
    print(validation_metrics.to_string(index=False))
    validation_metrics.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)

    # ---------------------------------------------------------
    # Optional feature-shuffling diagnostic on the validation split
    # ---------------------------------------------------------
    if RUN_FEATURE_SHUFFLING:
        feature_cols = [f"feature{i}" for i in range(1, 13)]
        rng = np.random.default_rng(config.seed)
        shuffled_val_df = val_df.copy()
        shuffled_val_df.loc[:, feature_cols] = (
            val_df[feature_cols].to_numpy()[rng.permutation(len(val_df))]
        )
        shuffled_metrics = evaluate_neuralfg(
            model=model, train_df=train_df, test_df=shuffled_val_df,
            batch_size=512,
        )
        print("\nOriginal validation C-index:")
        print(validation_metrics[["risk", "time", "c_index"]])
        print("\nShuffled-feature validation C-index:")
        print(shuffled_metrics[["risk", "time", "c_index"]])
        shuffled_metrics.to_csv(
            RESULTS_DIR / "shuffled_validation_metrics.csv", index=False
        )

    # NOTE: run_training currently computes test NLL internally regardless of
    # this flag. For strict isolation during hyperparameter search, also make
    # its test-NLL step optional in train.py (see review notes).
    if RUN_TEST_METRICS:
        test_metrics = evaluate_neuralfg(
            model=model, train_df=train_df, test_df=test_df, batch_size=512,
        )
        print("\nNeuralFG held-out test metrics:")
        print(test_metrics.to_string(index=False))
        test_metrics.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)

    # ---------------------------------------------------------
    # Plot predicted CIF curves for individual patients
    # ---------------------------------------------------------

    # Select 3 patients from the existing diagnostic sample
    patient_features = features[:3]

    # Use a finer time grid for visualization
    plot_times = torch.linspace(0, 60, 121)

    # Shape: [3 patients, 121 time points, 2 risks]
    patient_cif = model.predict_cif(
        features=patient_features,
        times=plot_times,
    ).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for risk_idx in range(2):
        for patient_idx in range(3):
            axes[risk_idx].plot(
                plot_times.numpy(),
                patient_cif[patient_idx, :, risk_idx],
                label=f"Patient {patient_idx + 1}",
            )

        axes[risk_idx].set_title(f"Risk {risk_idx + 1} CIF")
        axes[risk_idx].set_xlabel("Time")
        axes[risk_idx].set_ylabel("Cumulative incidence")
        axes[risk_idx].set_ylim(0, 1)
        axes[risk_idx].legend()
        axes[risk_idx].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "patient_cif_curves.png",
        dpi=300,
    )

    plt.close()

    print("Patient CIF curves saved.")

    # Save training history
    history.to_csv(
        RESULTS_DIR / "training_history.csv",
        index=False,
    )

    # Plot training and validation losses
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        label="Training NLL",
    )

    plt.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        label="Validation NLL",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("NeuralFG Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(
        RESULTS_DIR / "training_loss.png",
        dpi=300,
    )

    plt.close()

    print(f"\nTraining results saved to: {RESULTS_DIR}")
