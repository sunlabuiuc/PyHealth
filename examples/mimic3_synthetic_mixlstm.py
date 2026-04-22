"""
MixLSTM Hyperparameter Search Experiment
Synthetic time-series regression task with PyHealth.

All intermediate results (distributions, predictions, search metrics)
are kept in memory and passed directly to the visualization functions
instead of being written to / read from disk.
"""

import os
import random
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.models import MixLSTM


# ──────────────────────────────────────────────────────────────
# In-memory result containers
# ──────────────────────────────────────────────────────────────

@dataclass
class AblationResult:
    """Everything produced by a single ablation run."""
    learning_rate: float
    optimizer_name: str
    results_df: pd.DataFrame
    k_dist: list[np.ndarray]
    d_dist: list[np.ndarray]
    best_predictions: dict | None = None   # {"pred": ..., "y_true": ..., "k": ..., "hidden_size": ...}
    best_model_state: dict | None = None

    @property
    def label(self) -> str:
        """Human-readable label for plots and logs."""
        return f"{self.optimizer_name} lr={self.learning_rate}"


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

SEED = 42
NUM_SAMPLES = 1000
T = 30                       # sequence length
INPUT_DIM = 3
PREV_USED_TIMESTAMPS = 10    # l
CHANGE_BETWEEN_TASKS = 0.05  # delta

BATCH_SIZE = 100
K_LIST = [2]
HIDDEN_SIZE_LIST = [100, 150, 300, 500, 700, 900, 1100]
NUM_RUNS = 1  # 20
MAX_EPOCHS = 2  # 30

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Visualization
MAX_MSE = 100
ABLATION_LRS = [0.0001, 0.0005, 0.001, 0.005, 0.01]


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    return device


# ──────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────

def convert_distb(a: np.ndarray) -> np.ndarray:
    a_min = min(a)
    a_max = max(a)
    a = (a - a_min) / (a_max - a_min)
    a_sum = sum(a)
    a = a / a_sum
    return a


def generate_distributions(
    T: int,
    prev_used_timestamps: int,
    input_dim: int,
    change_between_tasks: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    k_dist = []
    d_dist = []
    for i in range(T):
        if i < prev_used_timestamps:
            k_dist.append(np.ones(prev_used_timestamps))
            d_dist.append(np.ones(input_dim))
        elif i == prev_used_timestamps:
            k_dist.append(convert_distb(np.random.uniform(size=(prev_used_timestamps,))))
            d_dist.append(convert_distb(np.random.uniform(size=(input_dim,))))
        else:
            delta_t = np.random.uniform(
                -change_between_tasks, change_between_tasks, size=(prev_used_timestamps,)
            )
            delta_d = np.random.uniform(
                -change_between_tasks, change_between_tasks, size=(input_dim,)
            )
            k_dist.append(convert_distb(k_dist[i - 1] + delta_t))
            d_dist.append(convert_distb(d_dist[i - 1] + delta_d))
    return k_dist, d_dist


def generate_xy(
    num_samples: int,
    T: int,
    input_dim: int,
    prev_used_timestamps: int,
    k_dist: list[np.ndarray],
    d_dist: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_size = num_samples * T * input_dim
    x = np.zeros(x_size)
    sparse_count = int(x_size / 10)
    x[np.random.choice(x_size, size=sparse_count, replace=False)] = (
        np.random.uniform(size=sparse_count) * 100
    )
    x = np.resize(x, (num_samples, T, input_dim))

    y = np.ones((num_samples, T, 1))
    for i in range(T):
        if i >= prev_used_timestamps:
            y[:, i, 0] = np.matmul(
                np.matmul(x[:, i - prev_used_timestamps : i, :], d_dist[i]),
                k_dist[i],
            )
    return x, y


# ──────────────────────────────────────────────────────────────
# PyHealth dataset helpers
# ──────────────────────────────────────────────────────────────

def make_dataset(x: np.ndarray, y: np.ndarray, split_name: str):
    samples = [
        {
            "patient_id": f"{split_name}-patient-{i}",
            "visit_id": "visit-0",
            "series": x[i].tolist(),
            "y": y[i].squeeze(-1).tolist(),
        }
        for i in range(len(x))
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"series": "tensor"},
        output_schema={"y": "tensor"},
        dataset_name=f"mixlstm_{split_name}",
    )


def build_dataloaders(
    k_dist, d_dist, num_samples, T, input_dim, prev_used_timestamps, batch_size
):
    x_train, y_train = generate_xy(num_samples, T, input_dim, prev_used_timestamps, k_dist, d_dist)
    x_val, y_val     = generate_xy(num_samples, T, input_dim, prev_used_timestamps, k_dist, d_dist)
    x_test, y_test   = generate_xy(num_samples, T, input_dim, prev_used_timestamps, k_dist, d_dist)

    train_data = make_dataset(x_train, y_train, "train")
    val_data   = make_dataset(x_val, y_val, "val")
    test_data  = make_dataset(x_test, y_test, "test")

    train_loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = get_dataloader(val_data,   batch_size=batch_size, shuffle=True)
    test_loader  = get_dataloader(test_data,  batch_size=batch_size, shuffle=True)

    return train_data, train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────────────────────

def collect_predictions(model, test_loader, device):
    """Run inference and return predictions + ground truth as numpy arrays."""
    model.eval()
    l = model.prev_used_timestamps
    preds, y_trues = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {
                k_: v.to(device) if isinstance(v, torch.Tensor) else v
                for k_, v in batch.items()
            }
            output = model(**batch)
            preds.append(output["y_prob"][:, l:, :].cpu().numpy())
            y_trues.append(output["y_true"][:, l:, :].cpu().numpy())

    return {
        "pred": np.concatenate(preds, axis=0).flatten(),
        "y_true": np.concatenate(y_trues, axis=0).flatten(),
    }


def run_hyperparameter_search(
    train_data,
    train_loader,
    val_loader,
    test_loader,
    device,
    prev_used_timestamps,
    k_list,
    hidden_size_list,
    num_runs,
    max_epochs,
    learning_rate,
    optimizer_class=optim.Adam,
):
    """Execute the hyperparameter search. Returns (results_df, best_predictions, best_model_state)."""
    results = []
    best_val_loss_overall = np.inf
    best_predictions = None
    best_model_state = None

    for run in range(num_runs):
        k = random.choice(k_list)
        hidden_size = random.choice(hidden_size_list)

        print(f"\n{'=' * 60}")
        print(f"Run {run + 1}/{num_runs} | k (num_experts): {k} | hidden_size: {hidden_size}")
        print("=" * 60)

        model = MixLSTM(
            dataset=train_data,
            num_experts=k,
            hidden_size=hidden_size,
            prev_used_timestamps=prev_used_timestamps,
        )
        model = model.to(device)

        trainer = Trainer(model=model, device=device)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer_class=optimizer_class,
            optimizer_params={"lr": learning_rate},
            epochs=max_epochs,
            monitor="loss",
            monitor_criterion="min",
        )

        print(f"\nEvaluating Best Model for Run {run + 1}...")
        val_metrics = trainer.evaluate(val_loader)
        test_metrics = trainer.evaluate(test_loader)

        val_loss = val_metrics.get("loss", None)
        test_loss = test_metrics.get("loss", None)

        if val_loss < best_val_loss_overall:
            best_val_loss_overall = val_loss
            print(f"  New best val loss: {val_loss:.6f}")
            predictions = collect_predictions(model, test_loader, device)
            predictions["k"] = k
            predictions["hidden_size"] = hidden_size
            predictions["run"] = run
            best_predictions = predictions
            best_model_state = {k_: v.cpu().clone() for k_, v in model.state_dict().items()}

        results.append({
            "Run": run + 1,
            "k (experts)": k,
            "Hidden Size": hidden_size,
            "Val Loss": val_loss,
            "Test Loss": test_loss,
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "epoch": max_epochs,
        })

    return pd.DataFrame(results), best_predictions, best_model_state


# ──────────────────────────────────────────────────────────────
# Ablation study — learning rate sweep
# ──────────────────────────────────────────────────────────────

def run_single_ablation(
    learning_rate: float,
    optimizer_class=optim.Adam,
    optimizer_name: str = "Adam",
) -> AblationResult:
    """Run the full search for one (optimizer, lr) combination and return all results in memory."""
    set_seed(SEED)
    device = get_device()
    logging.getLogger("pyhealth.trainer").setLevel(logging.WARNING)

    if device.type == "cuda":
        torch.set_default_device(device)

    k_dist, d_dist = generate_distributions(
        T, PREV_USED_TIMESTAMPS, INPUT_DIM, CHANGE_BETWEEN_TASKS
    )

    train_data, train_loader, val_loader, test_loader = build_dataloaders(
        k_dist, d_dist, NUM_SAMPLES, T, INPUT_DIM, PREV_USED_TIMESTAMPS, BATCH_SIZE
    )

    print(f"\n{'#' * 60}")
    print(f"  ABLATION — optimizer = {optimizer_name}, learning_rate = {learning_rate}")
    print(f"{'#' * 60}")

    results_df, best_predictions, best_model_state = run_hyperparameter_search(
        train_data=train_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        prev_used_timestamps=PREV_USED_TIMESTAMPS,
        k_list=K_LIST,
        hidden_size_list=HIDDEN_SIZE_LIST,
        num_runs=NUM_RUNS,
        max_epochs=MAX_EPOCHS,
        learning_rate=learning_rate,
        optimizer_class=optimizer_class,
    )

    best = results_df.sort_values(by="Test Loss").reset_index(drop=True)
    print(f"\nTop 5 results for {optimizer_name} lr={learning_rate}:")
    print(best.head(5))

    return AblationResult(
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        results_df=results_df,
        k_dist=k_dist,
        d_dist=d_dist,
        best_predictions=best_predictions,
        best_model_state=best_model_state,
    )


def run_all_ablations() -> list[AblationResult]:
    """Run learning-rate ablations with Adam (original behaviour)."""
    ablation_results = [
        run_single_ablation(lr, optim.Adam, "Adam") for lr in ABLATION_LRS
    ]
    _print_summary("Learning Rate Sweep (Adam)", ablation_results)
    return ablation_results


# ──────────────────────────────────────────────────────────────
# Ablation study — optimizer comparison  (Adam vs SGD)
# ──────────────────────────────────────────────────────────────

ABLATION_OPTIMIZER_LR = 0.001  # fixed LR used for the optimizer comparison

def ablations_optimizing_adam() -> AblationResult:
    """Ablation: Adam optimizer at lr=0.001."""
    return run_single_ablation(ABLATION_OPTIMIZER_LR, optim.Adam, "Adam")


def ablations_optimizing_sgd() -> AblationResult:
    """Ablation: SGD optimizer at lr=0.001."""
    return run_single_ablation(ABLATION_OPTIMIZER_LR, optim.SGD, "SGD")


def run_optimizer_ablations() -> list[AblationResult]:
    """Run Adam vs SGD at a fixed learning rate and print a comparison."""
    results = [
        ablations_optimizing_adam(),
        ablations_optimizing_sgd(),
    ]
    _print_summary("Optimizer Comparison (Adam vs SGD)", results)
    return results


def _print_summary(title: str, ablation_results: list[AblationResult]):
    """Pretty-print a summary table for a list of ablation results."""
    summary_rows = []
    for result in ablation_results:
        best_row = result.results_df.sort_values(by="Test Loss").iloc[0]
        summary_rows.append({
            "Optimizer": result.optimizer_name,
            "Learning Rate": result.learning_rate,
            "Best Val Loss": best_row["Val Loss"],
            "Best Test Loss": best_row["Test Loss"],
            "k (experts)": best_row["k (experts)"],
            "Hidden Size": best_row["Hidden Size"],
            "num_params": best_row["num_params"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print(summary_df.to_string(index=False))


# ──────────────────────────────────────────────────────────────
# Visualization  (all functions take in-memory data)
# ──────────────────────────────────────────────────────────────

def visualize_hyperparameter_search(ablation_results: list[AblationResult], prefix: str = ""):
    """Plot MSE loss vs. hidden size for every learning rate."""
    print("--- 1. Analyzing Hyperparameter Search (Ablation) ---")

    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("tab10", len(ablation_results))

    for i, result in enumerate(ablation_results):
        tag = result.label
        df = result.results_df.copy()
        df = df[(df["Val Loss"] <= MAX_MSE) & (df["Test Loss"] <= MAX_MSE)]

        best = df.sort_values(by="Val Loss").head(1)
        print(
            f"  {tag}  Best Val Loss: {best['Val Loss'].values[0]:.6f}  "
            f"(Hidden Size={best['Hidden Size'].values[0]})"
        )

        color = palette[i]

        sns.scatterplot(
            data=df, x="Hidden Size", y="Val Loss",
            label=f"Val ({tag})", color=color,
            marker="o", alpha=0.4, s=40,
        )
        sns.scatterplot(
            data=df, x="Hidden Size", y="Test Loss",
            label=f"Test ({tag})", color=color,
            marker="x", alpha=0.4, s=40,
        )

        val_mean = df.groupby("Hidden Size")["Val Loss"].mean().sort_index()
        test_mean = df.groupby("Hidden Size")["Test Loss"].mean().sort_index()
        plt.plot(val_mean.index, val_mean.values, color=color, linewidth=2, linestyle="-")
        plt.plot(test_mean.index, test_mean.values, color=color, linewidth=2, linestyle="--")

    plt.title("Ablation: MSE Loss vs. Hidden Size by Learning Rate")
    plt.xlabel("Hidden Size")
    plt.ylabel("MSE Loss")
    plt.legend(title="Loss Type / LR", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, MAX_MSE)
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, f"{prefix}ablation_loss_vs_hidden_size.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def visualize_predictions(ablation_results: list[AblationResult], num_samples: int = 3, prefix: str = ""):
    """Plot predicted vs. true values for a few test samples per learning rate."""
    print("\n--- 2. Analyzing Predictions (Ablation) ---")

    # Only include results that have saved predictions
    valid_results = [r for r in ablation_results if r.best_predictions is not None]
    if not valid_results:
        print("  No predictions available to plot.")
        return

    fig, axes = plt.subplots(len(valid_results), num_samples, figsize=(15, 4 * len(valid_results)))
    if len(valid_results) == 1:
        axes = [axes]

    for row, result in enumerate(valid_results):
        tag = result.label
        y_true_flat = result.best_predictions["y_true"]
        pred_flat = result.best_predictions["pred"]

        l = result.best_predictions.get("k", PREV_USED_TIMESTAMPS)
        eval_steps = T - l
        num_test_samples = len(y_true_flat) // eval_steps
        limit = num_test_samples * eval_steps

        y_true = np.reshape(y_true_flat[:limit], (num_test_samples, eval_steps))
        pred = np.reshape(pred_flat[:limit], (num_test_samples, eval_steps))

        sample_indices = np.random.choice(
            num_test_samples, min(num_samples, num_test_samples), replace=False
        )

        for col, sample_idx in enumerate(sample_indices):
            ax = axes[row][col]
            ax.plot(y_true[sample_idx], label="True", color="blue", marker="o", markersize=4)
            ax.plot(pred[sample_idx], label="Predicted", color="red", linestyle="--", marker="x", markersize=4)
            ax.set_title(f"{tag} | Sample #{sample_idx}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"{prefix}ablation_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def visualize_synthetic_shift(ablation_results: list[AblationResult], prefix: str = ""):
    """Plot the k_dist heatmap for each learning rate's task distributions."""
    print("\n--- 3. Analyzing Synthetic Data Shift (Ablation) ---")

    fig, axes = plt.subplots(1, len(ablation_results), figsize=(6 * len(ablation_results), 5))
    if len(ablation_results) == 1:
        axes = [axes]

    for i, result in enumerate(ablation_results):
        k_dist_matrix = np.stack(result.k_dist)
        sns.heatmap(k_dist_matrix.T, cmap="viridis", ax=axes[i], cbar_kws={"label": "Weight"})
        axes[i].set_title(f"k_dist Shift ({result.label})")
        axes[i].set_xlabel("Time Step (T)")
        axes[i].set_ylabel("Lookback Step (l)")

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f"{prefix}ablation_synthetic_shift.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run_all_visualizations(ablation_results: list[AblationResult], prefix: str = ""):
    """Generate all three ablation plots from in-memory results."""
    visualize_hyperparameter_search(ablation_results, prefix=prefix)
    visualize_synthetic_shift(ablation_results, prefix=prefix)
    visualize_predictions(ablation_results, prefix=prefix)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    # Learning-rate sweep (Adam only, multiple LRs)
    lr_results = run_all_ablations()
    run_all_visualizations(lr_results, prefix="lr_sweep_")

    # Optimizer comparison (Adam vs SGD at fixed LR)
    optimizer_results = run_optimizer_ablations()
    run_all_visualizations(optimizer_results, prefix="optim_comp_")


if __name__ == "__main__":
    main()