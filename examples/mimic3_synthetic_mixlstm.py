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


# ======================================================================
# MixLSTM Hyperparameter Search Experiment
# Synthetic time-series regression task with PyHealth
# ======================================================================
#
# EXPERIMENTAL SETUP
# ------------------
# Dataset: Synthetic non-stationary time-series regression. 1,000 sequences
#   per split (train/val/test), length T=30, 3 input features. Inputs are
#   90% sparse. Targets from step l=10 onward are weighted combinations of
#   prior inputs, where the weights drift by delta=0.05 per step to simulate
#   distribution shift.
#
# Model: MixLSTM (PyHealth) with k=2 experts and lookback window l=10.
#   Hidden size sampled from {100, 150, 300, 500, 700, 900, 1100}.
#   20 random-search runs per config, 30 epochs each, batch size 100.
#
# ABLATION STUDIES
# ----------------
# 1) Learning rate sweep: Adam at lr in {0.0001, 0.0005, 0.001, 0.005, 0.01}
# 2) Optimizer comparison: Adam vs SGD at lr=0.001
# 3) Every other parameter kept as default
#
# FINDINGS
# ----------------
# 1. OPTIMIZER COMPARISON
# ----------------------------------------------------------------------------
# Conclusion: Adam consistently outperformed SGD across training runs.
# 
# | Optimizer | Lowest Val Loss (MSE) | Lowest Test Loss (MSE) |
# |-----------|-----------------------|------------------------|
# | Adam      | 0.430089              | 0.467544               |
# | SGD       | 16.388920             | 16.411073              |
#
# 2. LEARNING RATE VS. HIDDEN SIZE COMPARISON
# --------------------------------------------------------------------------------------------------------------------------
# Format: (Validation Loss MSE - Test Loss MSE)
# 
#           | Hidden Size
# LR        | 100             150             300             500             700             900             1100
# ----------|---------------------------------------------------------------------------------------------------------------
# 0.0001    | (-)             (14.14 - 14.54) (10.76 - 11.05) (7.59 - 7.73)   (5.53 - 5.44)   (4.60 - 4.76)   (4.31 - 4.39)
# 0.0005    | (10.78 - 11.06) (9.30 - 9.76)   (5.31 - 5.87)   (4.02 - 4.39)   (2.81 - 3.09)   (1.61 - 1.89)   (1.33 - 1.52)
# 0.001     | (6.37 - 6.51)   (4.49 - 4.62)   (2.60 - 2.67)   (1.26 - 1.31)   (0.87 - 0.91)   (0.69 - 0.77)   (0.43 - 0.46)
# 0.005     | (2.20 - 2.28)   (1.41 - 1.53)   (0.68 - 0.77)   (0.48 - 0.62)   (0.89 - 1.01)   (-)             (0.68 - 0.74)
# 0.01      | (1.79 - 1.88)   (1.42 - 1.47)   (1.10 - 1.14)   (1.03 - 1.10)   (1.54 - 1.58)   (0.91 - 0.98)   (2.24 - 2.41)
# ==========================================================================================================================
#   
# Conclution:
#  LR = 0.0001 was the worst performer overall across all hidden sizes
#  LR = 0.0005 was also the second word performer overall across almost all hidden states
#  LR = 0.001 this was the learning rate that the paper used. LR value 0.01 and 0.005 were better in the lower hidden sizes 
#       eg 100, 150, 300, 500. For the reast LR 0.001 was the best choice overall 
#  LR = 0.05 this rate was the best overall for the lower hidden sizes from 100 to 500 but then had a spike 
#       at 700 but then managed to go down. Ideal for lower hidden rates
#  LR = 0.01 this rate was quite spradic and unstable and it went up and down multiple times and is not recommended
#
# Overall Conclution of the entire study:
#  Adam optimization gives the best results
#  For learning rate 0.001 is great for hidden sizes above 500 and LR = 0.005 is the best for hidden size below 500
#  
# How to run Study
# pip install seaborn
# run the python file
# you will see 6 .png files diplaying the results as graphs
#





# ──────────────────────────────────────────────────────────────
# In-memory result containers
# ──────────────────────────────────────────────────────────────

@dataclass
class AblationResult:
    """Container for every artefact produced by a single ablation run.
 
    Attributes:
        learning_rate: The learning rate used for this ablation.
        optimizer_name: Human-readable optimizer name (e.g. ``"Adam"``).
        results_df: DataFrame with one row per random-search run.
            Columns include ``Run``, ``k (experts)``, ``Hidden Size``,
            ``Val Loss``, ``Test Loss``, ``num_params``, and ``epoch``.
        k_dist: List of *T* numpy arrays representing the temporal
            weight distribution at each time step.
        d_dist: List of *T* numpy arrays representing the feature
            weight distribution at each time step.
        best_predictions: Dictionary with keys ``"pred"``,
            ``"y_true"``, ``"k"``, ``"hidden_size"``, and ``"run"``
            for the model that achieved the lowest validation loss.
            ``None`` if no valid model was produced.
        best_model_state: ``state_dict`` (on CPU) of the best model.
            ``None`` if no valid model was produced.
    """
    learning_rate: float
    optimizer_name: str
    results_df: pd.DataFrame
    k_dist: list[np.ndarray]
    d_dist: list[np.ndarray]
    best_predictions: dict | None = None 
    best_model_state: dict | None = None

    @property
    def label(self) -> str:
        """Return a human-readable label for plots and logs.
 
        Returns:
            A string of the form ``"<optimizer> lr=<lr>"``.
        """
        
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
NUM_RUNS = 20  # 20
MAX_EPOCHS = 30  # 30

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Visualization
MAX_MSE = 100
ABLATION_LRS = [0.0001, 0.0005, 0.001, 0.005, 0.01]


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.
 
    Args:
        seed: Integer seed value applied to every RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detect and return the best available compute device.
 
    Returns:
        ``torch.device("cuda")`` when a CUDA GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    return device


# ──────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────

def convert_distb(a: np.ndarray) -> np.ndarray:
    """Min-max normalize an array and rescale it to sum to one.
 
    The array is first shifted and scaled to the [0, 1] range via
    min-max normalization, then divided by its sum so that it
    forms a valid discrete probability distribution.
 
    Args:
        a: 1-D numpy array of raw (un-normalized) weights.
 
    Returns:
        A 1-D numpy array of the same shape whose elements are
        non-negative and sum to 1.
    """
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
    """Generate time-varying weight distributions for synthetic targets.
 
    Creates ``k_dist`` (temporal) and ``d_dist`` (feature) weight
    vectors that drift by a small delta at each step beyond the
    lookback window, simulating non-stationary distribution shift.
 
    For time steps before *prev_used_timestamps*, both distributions
    are uniform placeholders.  At step *prev_used_timestamps* the
    distributions are initialized randomly, and at each subsequent
    step a uniform perturbation in ``[-change_between_tasks,
    +change_between_tasks]`` is added before re-normalization.
 
    Args:
        T: Total sequence length.
        prev_used_timestamps: Lookback window size (*l*).
            Distributions before this index are uniform placeholders.
        input_dim: Number of input features per time step.
        change_between_tasks: Maximum per-step drift (*delta*)
            applied uniformly at random to each weight element.
 
    Returns:
        A tuple ``(k_dist, d_dist)`` where:
 
        * ``k_dist`` is a list of *T* arrays, each of shape
          ``(prev_used_timestamps,)``.
        * ``d_dist`` is a list of *T* arrays, each of shape
          ``(input_dim,)``.
    """
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
    
    """Generate sparse input sequences and their regression targets.
 
    Inputs are 90 % sparse (zeros) with the remaining 10 % drawn
    uniformly from ``[0, 100)``.  For time steps ``t >= l`` the target
    is ``x[t-l:t, :] @ d_dist[t] @ k_dist[t]``; earlier targets are
    ones (placeholders).
 
    Args:
        num_samples: Number of independent sequences to generate.
        T: Sequence length (number of time steps).
        input_dim: Dimensionality of input features.
        prev_used_timestamps: Lookback window size (*l*).
        k_dist: Temporal weight distributions as returned by
            :func:`generate_distributions`.
        d_dist: Feature weight distributions as returned by
            :func:`generate_distributions`.
 
    Returns:
        A tuple ``(x, y)`` where:
 
        * ``x`` has shape ``(num_samples, T, input_dim)``.
        * ``y`` has shape ``(num_samples, T, 1)``.
    """
    
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

def make_dataset(x: np.ndarray, y: np.ndarray, split_name: str) -> "SampleDataset":
    """Wrap numpy arrays into a PyHealth ``SampleDataset``.
 
    Each sequence is registered as a separate patient with a single
    visit containing the full time-series.
 
    Args:
        x: Input tensor of shape ``(N, T, D)``.
        y: Target tensor of shape ``(N, T, 1)``.
        split_name: Identifier for the split (e.g. ``"train"``,
            ``"val"``, ``"test"``).  Used in patient IDs and as the
            PyHealth dataset name suffix.
 
    Returns:
        A PyHealth ``SampleDataset`` ready to be passed to
        ``get_dataloader``.
    """

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
) -> tuple["SampleDataset", torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    """Generate train / val / test splits and wrap them in DataLoaders.
 
    Three independent datasets are synthesized from the same
    underlying distributions so that the only source of variance is
    the random sparse masking and the ordering of non-zero entries.
 
    Args:
        k_dist: Temporal weight distributions (see
            :func:`generate_distributions`).
        d_dist: Feature weight distributions (see
            :func:`generate_distributions`).
        num_samples: Number of sequences per split.
        T: Sequence length.
        input_dim: Number of input features.
        prev_used_timestamps: Lookback window size (*l*).
        batch_size: Mini-batch size for every DataLoader.
 
    Returns:
        A tuple ``(train_dataset, train_loader, val_loader,
        test_loader)``.  The raw ``train_dataset`` is also returned
        because ``MixLSTM.__init__`` requires it to infer schema
        metadata.
    """
    x_train, y_train = generate_xy(
        num_samples, T, input_dim, prev_used_timestamps, k_dist, d_dist
        )
    x_val, y_val     = generate_xy(
        num_samples, T, input_dim, prev_used_timestamps, k_dist, d_dist
        )
    x_test, y_test   = generate_xy(
        num_samples, T, input_dim, prev_used_timestamps, k_dist, d_dist
        )

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

def collect_predictions(model, test_loader, device) -> dict[str, np.ndarray]:
    """Run inference on *test_loader* and collect predictions.
 
    The model is set to eval mode and gradients are disabled.  Only
    time steps from index *l* onward (the non-placeholder region)
    are retained.
 
    Args:
        model: A trained ``MixLSTM`` model instance.
        test_loader: DataLoader yielding test batches.
        device: Device the model resides on.
 
    Returns:
        A dictionary with two keys:
 
        * ``"pred"`` — flattened 1-D numpy array of predicted values.
        * ``"y_true"`` — flattened 1-D numpy array of ground-truth
          values, aligned element-wise with ``"pred"``.
    """
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
) -> tuple[pd.DataFrame, dict | None, dict | None]:
    
    """Execute a random hyperparameter search over MixLSTM configs.
 
    Each run samples ``k`` (number of experts) and ``hidden_size``
    uniformly from the provided lists, trains for *max_epochs*, and
    records validation / test loss.  The model with the lowest
    validation loss is retained.
 
    Args:
        train_data: PyHealth ``SampleDataset`` used to initialize
            ``MixLSTM`` (needed for schema inference).
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        test_loader: DataLoader for the test split.
        device: Compute device (CPU or CUDA).
        prev_used_timestamps: Lookback window size (*l*) passed to
            ``MixLSTM``.
        k_list: Candidate values for the number of mixture experts.
        hidden_size_list: Candidate values for the LSTM hidden
            dimension.
        num_runs: Total number of random configurations to evaluate.
        max_epochs: Training epochs per run.
        learning_rate: Learning rate passed to the optimizer.
        optimizer_class: PyTorch optimizer class (e.g.
            ``torch.optim.Adam``).
 
    Returns:
        A tuple ``(results_df, best_predictions, best_model_state)``
        where:
 
        * ``results_df`` — DataFrame with columns ``Run``,
          ``k (experts)``, ``Hidden Size``, ``Val Loss``,
          ``Test Loss``, ``num_params``, and ``epoch``.
        * ``best_predictions`` — dictionary as returned by
          :func:`collect_predictions`, augmented with ``"k"``,
          ``"hidden_size"``, and ``"run"`` keys.  ``None`` when no
          valid model was found.
        * ``best_model_state`` — CPU ``state_dict`` of the
          best-performing model.  ``None`` when no valid model was
          found.
    """

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
    
    """Run the full hyperparameter search for one (optimizer, lr) pair.
 
    This is the main entry point for a single ablation cell.  It
    seeds RNGs, generates data, builds data loaders, trains all
    random-search runs, and packages the results into an
    :class:`AblationResult`.
 
    Args:
        learning_rate: Learning rate forwarded to the optimizer.
        optimizer_class: PyTorch optimizer class to use (e.g.
            ``torch.optim.Adam``, ``torch.optim.SGD``).
        optimizer_name: Human-readable name stored in the result
            object and used in plot labels.
 
    Returns:
        An :class:`AblationResult` containing the results DataFrame,
        weight distributions, best predictions, and best model state.
    """
    
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
    """Run the learning-rate sweep ablation using the Adam optimizer.
 
    Iterates over every learning rate in :data:`ABLATION_LRS`, runs
    the full hyperparameter search for each, and prints a summary
    table.
 
    Returns:
        A list of :class:`AblationResult` objects, one per learning
        rate, in the same order as :data:`ABLATION_LRS`.
    """
    
    
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

    """Run the Adam ablation at the fixed comparison learning rate.
 
    Returns:
        An :class:`AblationResult` for Adam at
        lr = :data:`ABLATION_OPTIMIZER_LR`.
    """

    
    return run_single_ablation(ABLATION_OPTIMIZER_LR, optim.Adam, "Adam")


def ablations_optimizing_sgd() -> AblationResult:

    """Run the SGD ablation at the fixed comparison learning rate.
 
    Returns:
        An :class:`AblationResult` for SGD at
        lr = :data:`ABLATION_OPTIMIZER_LR`.
    """

    return run_single_ablation(ABLATION_OPTIMIZER_LR, optim.SGD, "SGD")


def run_optimizer_ablations() -> list[AblationResult]:
    """Compare Adam and SGD at a fixed learning rate.
 
    Both optimizers are trained with
    lr = :data:`ABLATION_OPTIMIZER_LR` and the results are printed
    side by side.
 
    Returns:
        A two-element list ``[adam_result, sgd_result]``.
    """
    
    results = [
        ablations_optimizing_adam(),
        ablations_optimizing_sgd(),
    ]
    _print_summary("Optimizer Comparison (Adam vs SGD)", results)
    return results


def _print_summary(title: str, ablation_results: list[AblationResult])-> None:

    """Pretty-print a summary table for a list of ablation results.
 
    For each :class:`AblationResult` the row with the lowest test
    loss is selected and its key metrics are displayed.
 
    Args:
        title: Header string printed above the table.
        ablation_results: Results to summarize.
    """
    
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

def visualize_hyperparameter_search(ablation_results: list[AblationResult], prefix: str = "") -> None:

    """Plot MSE loss vs. hidden size for every learning rate.
 
    Individual run results are shown as translucent scatter points
    and per-hidden-size means are overlaid as solid (validation) and
    dashed (test) lines.
 
    Args:
        ablation_results: One :class:`AblationResult` per learning
            rate / optimizer configuration.
        prefix: String prepended to the output filename (e.g.
            ``"lr_sweep_"``).
    """
    
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


def visualize_predictions(ablation_results: list[AblationResult], num_samples: int = 3, prefix: str = "") -> None:

    """Plot predicted vs. true values for sample test sequences.
 
    One row of subplots is created per ablation result, with
    *num_samples* columns each showing a randomly chosen test
    sequence.
 
    Args:
        ablation_results: Ablation results whose
            ``best_predictions`` field will be visualized.  Entries
            with ``best_predictions is None`` are silently skipped.
        num_samples: Number of randomly selected test sequences to
            plot per ablation.
        prefix: String prepended to the output filename.
    """
    
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
            ax.plot(
                pred[sample_idx], label="Predicted", 
                color="red", linestyle="--", marker="x", markersize=4
                )
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


def visualize_synthetic_shift(ablation_results: list[AblationResult], prefix: str = "") -> None:
    """Plot the ``k_dist`` heatmap for each ablation's task distributions.
 
    Each subplot shows how the temporal weight distribution evolves
    across the *T* time steps (x-axis) over the *l* lookback
    positions (y-axis).
 
    Args:
        ablation_results: Ablation results whose ``k_dist`` fields
            will be visualized.
        prefix: String prepended to the output filename.
    """

    
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


def run_all_visualizations(ablation_results: list[AblationResult], prefix: str = "") -> None:
    """Generate all three ablation plot types from in-memory results.
 
    Delegates to :func:`visualize_hyperparameter_search`,
    :func:`visualize_synthetic_shift`, and
    :func:`visualize_predictions`.
 
    Args:
        ablation_results: The list of :class:`AblationResult` objects
            to visualize.
        prefix: Filename prefix forwarded to each plotting function.
    """
    
    visualize_hyperparameter_search(ablation_results, prefix=prefix)
    visualize_synthetic_shift(ablation_results, prefix=prefix)
    visualize_predictions(ablation_results, prefix=prefix)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    # Learning-rate sweep (Adam only, multiple LRs)
    lr_results = run_all_ablations()
    run_all_visualizations(lr_results, prefix="lr_sweep_")

    # Optimizer comparison (Adam vs SGD at fixed LR)
    optimizer_results = run_optimizer_ablations()
    run_all_visualizations(optimizer_results, prefix="optim_comp_")


if __name__ == "__main__":
    main()