"""
Ablation study for DeepCoxMixtures on SUPPORT2 (2-month survival horizon).
 
Three ablations: K in {1,3,6}, hidden_dim in {32,64,128}, dropout in
{0.0,0.2,0.5}. Each config reports MSE, MAE, and C-index on a held-out
test set. K=1 is the single Cox baseline.
 
Data: download support2.csv from https://archive.ics.uci.edu/dataset/880/support2
and place at <DATA_ROOT>/support2.csv. The sno column is added automatically
if missing. Set USE_SYNTHETIC=1 to skip the download and just verify the
pipeline runs end-to-end (C-index values won't be meaningful at that scale).
 
Setup: 9104 patients, 70/15/15 split, Adam lr=1e-3, 20 epochs.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from pyhealth.datasets import Support2Dataset, get_dataloader, split_by_patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.models.deep_cox_mixtures import DeepCoxMixtures
from pyhealth.tasks import SurvivalPreprocessSupport2
from pyhealth.trainer import Trainer
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# Configuration
# Set USE_SYNTHETIC=1 to run without real data for smoke-test only.
# Set DATA_ROOT to your SUPPORT2 CSV directory for the full ablation.
USE_SYNTHETIC = os.environ.get("USE_SYNTHETIC", "0") == "1"
DATA_ROOT = "./data/support2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
SEED = 42

# Reduced settings for synthetic smoke-test
SYNTHETIC_EPOCHS = 2
SYNTHETIC_BATCH_SIZE = 4


# Synthetic data fallback

SYNTHETIC_SAMPLES = [
    {"patient_id": f"p{i}", "visit_id": "v0",
     "demographics": [f"age_{50+i}.0", f"sex_{'male' if i%2==0 else 'female'}"],
     "disease_codes": [f"dzgroup_Cancer"],
     "vitals": [f"meanbp_{80+i}.0"],
     "labs": [f"alb_{3.5}.0"],
     "scores": [f"sps_{30+i}.0"],
     "comorbidities": ["diabetes_0"],
     "survival_probability": round(0.9 - i * 0.1, 1)}
    for i in range(8)
]

SYNTHETIC_INPUT_SCHEMA = {
    "demographics": "sequence",
    "disease_codes": "sequence",
    "vitals": "sequence",
    "labs": "sequence",
    "scores": "sequence",
    "comorbidities": "sequence",
}
SYNTHETIC_OUTPUT_SCHEMA = {"survival_probability": "regression"}


def build_synthetic_dataset():
    """Build a small synthetic SampleDataset matching the SUPPORT2 schema."""
    return create_sample_dataset(
        samples=SYNTHETIC_SAMPLES,
        input_schema=SYNTHETIC_INPUT_SCHEMA,
        output_schema=SYNTHETIC_OUTPUT_SCHEMA,
        dataset_name="support2_synthetic",
    )


def prepare_csv(data_root: str) -> None:
    """Ensure the CSV has the 'sno' patient-ID column expected by Support2Dataset."""
    csv_path = Path(data_root) / "support2.csv"
    df = pd.read_csv(csv_path)
    if "sno" not in df.columns:
        df.insert(0, "sno", range(1, len(df) + 1))
        df.to_csv(csv_path, index=False)
        print(f"  Added 'sno' column to {csv_path}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def concordance_index(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Compute Harrell's concordance index (C-index). Higher predicted survival probability should 
    correspond to longer survival (lower risk). Counts concordant pairs where a patient with 
    higher true survival also receives a higher predicted survival."""

    n = len(y_true)
    concordant = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            comparable += 1
            if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                concordant += 1
    return concordant / comparable if comparable > 0 else 0.5


def evaluate_model(
    model: DeepCoxMixtures,
    loader: DataLoader,
) -> Dict[str, float]:
    """Run inference on a dataloader and return MSE, MAE, and C-index."""
    model.eval()
    all_y_true: List[np.ndarray] = []
    all_y_prob: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            all_y_true.append(out["y_true"].cpu().numpy().flatten())
            all_y_prob.append(out["y_prob"].cpu().numpy().flatten())

    y_true = np.concatenate(all_y_true)
    y_prob = np.concatenate(all_y_prob)

    mse = float(np.mean((y_true - y_prob) ** 2))
    mae = float(np.mean(np.abs(y_true - y_prob)))
    c_idx = concordance_index(y_true, y_prob)

    return {"mse": mse, "mae": mae, "c_index": c_idx}


def run_config(
    sample_dataset,
    train_dataset,
    val_dataset,
    test_dataset,
    label: str,
    epochs: int = EPOCHS,
    **model_kwargs,
) -> Dict[str, float]:
    """Train and evaluate one model configuration."""
    set_seed(SEED)

    train_loader = get_dataloader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = get_dataloader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = DeepCoxMixtures(dataset=sample_dataset, **model_kwargs)

    trainer = Trainer(
        model=model,
        device=DEVICE,
        metrics=["mse", "mae"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="mse",
        monitor_criterion="min",
        optimizer_params={"lr": LR},
    )

    metrics = evaluate_model(model, test_loader)
    print(
        f"  {label:<45} | MSE={metrics['mse']:.4f} "
        f"MAE={metrics['mae']:.4f} C-index={metrics['c_index']:.4f}"
    )
    return metrics


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<45} | MSE     MAE     C-index")
    print(f"  {'-' * 67}")



if __name__ == "__main__":

    if USE_SYNTHETIC:
        # ------------------------------------------------------------------
        # Synthetic smoke-test — no real data required
        # Verifies the model trains and produces valid metrics.
        # C-index values are not meaningful at this scale.
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("  Running in SYNTHETIC mode (smoke-test only)")
        print("  Set USE_SYNTHETIC=0 for full SUPPORT2 ablation.")
        print("=" * 70)

        sample_dataset = build_synthetic_dataset()
        # Use entire synthetic dataset for all splits in smoke-test
        train_ds = val_ds = test_ds = sample_dataset
        epochs = SYNTHETIC_EPOCHS

    else:
        # Full ablation on real SUPPORT2 data
        print_section("Loading SUPPORT2 Dataset")

        #Prepare CSV, will add 'sno' column if missing to avoid error
        prepare_csv(DATA_ROOT)

        base_dataset = Support2Dataset(
            root=DATA_ROOT,
            tables=["support2"],
        )

        # Apply survival preprocessing task on a 2-month horizion
        sample_dataset = base_dataset.set_task(
            task=SurvivalPreprocessSupport2(time_horizon="2m")
        )
        print(f"  Total samples  : {len(sample_dataset)}")
        print(f"  Input schema   : {sample_dataset.input_schema}")
        print(f"  Output schema  : {sample_dataset.output_schema}")

        # Split 70/15/15 by patient
        train_ds, val_ds, test_ds = split_by_patient(
            sample_dataset, [0.70, 0.15, 0.15]
        )
        print(
            f"  Train / Val / Test: "
            f"{len(train_ds)} / {len(val_ds)} / {len(test_ds)}"
        )
        epochs = EPOCHS

    # Ablation 1 — Number of mixture components K
    # K=1 is the neural Cox baseline (no mixture). Claim is K>1 improves survival prediction over K=1.
    print_section("Ablation 1: Number of Mixture Components (K)")
    ablation1_results = {}
    for k in [1, 3, 6]:
        label = f"K={k} ({'neural Cox baseline' if k == 1 else 'DCM'})"
        ablation1_results[k] = run_config(
            sample_dataset, train_ds, val_ds, test_ds,
            label=label,
            epochs=epochs,
            num_mixtures=k,
            hidden_dim=64,
            dropout=0.0,
        )

    # Ablation 2 — Hidden dimension
    # Tests whether a wider encoder improves predictive performance
    print_section("Ablation 2: Hidden Dimension (K=3 fixed)")
    ablation2_results = {}
    for hdim in [32, 64, 128]:
        label = f"hidden_dim={hdim}"
        ablation2_results[hdim] = run_config(
            sample_dataset, train_ds, val_ds, test_ds,
            label=label,
            epochs=epochs,
            num_mixtures=3,
            hidden_dim=hdim,
            dropout=0.0,
        )

    # Ablation 3 — Dropout regularisation
    # Tests whether dropout helps generalisation on this dataset
    print_section("Ablation 3: Dropout (K=3, hidden_dim=64 fixed)")
    ablation3_results = {}
    for drop in [0.0, 0.2, 0.5]:
        label = f"dropout={drop}"
        ablation3_results[drop] = run_config(
            sample_dataset, train_ds, val_ds, test_ds,
            label=label,
            epochs=epochs,
            num_mixtures=3,
            hidden_dim=64,
            dropout=drop,
        )

    # Summary
    print_section("Summary: Best C-index per Ablation")
    best_k = max(ablation1_results, key=lambda k: ablation1_results[k]["c_index"])
    best_h = max(ablation2_results, key=lambda k: ablation2_results[k]["c_index"])
    best_d = max(ablation3_results, key=lambda k: ablation3_results[k]["c_index"])

    print(f"  Best K           : {best_k}  "
          f"(C-index={ablation1_results[best_k]['c_index']:.4f})")
    print(f"  Best hidden_dim  : {best_h}  "
          f"(C-index={ablation2_results[best_h]['c_index']:.4f})")
    print(f"  Best dropout     : {best_d}  "
          f"(C-index={ablation3_results[best_d]['c_index']:.4f})")
    print(f"\n  Baseline (K=1) C-index : {ablation1_results[1]['c_index']:.4f}")
    print(
        f"  Best DCM (K>1)  C-index : "
        f"{max(v['c_index'] for k, v in ablation1_results.items() if k > 1):.4f}"
    )
    print("\nDone.")