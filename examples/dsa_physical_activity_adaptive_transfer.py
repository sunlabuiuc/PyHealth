"""
Ablation Study: Adaptive Transfer Learning for Physical Activity Monitoring
===========================================================================
Paper: Zhang et al. "Daily Physical Activity Monitoring: Adaptive Learning
       from Multi-source Motion Sensor Data." CHIL 2024.

This script runs THREE novel ablation studies not present in the original paper:

  Ablation 1 - Backbone Comparison:
      Tests LSTM vs ResNet backbone classifiers to see which benefits
      more from IPD-guided transfer learning.

  Ablation 2 - Target Sensor Variation:
      Varies which body-part sensor is treated as the "daily wearable"
      (target domain). Tests whether limb sensors suffer more than
      torso sensors when used as single-source input.

  Ablation 3 - Distance Metric for IPD:
      Compares Euclidean vs DTW distance when computing IPD scores,
      measuring how metric choice affects final classification accuracy.

NOTE: All ablations use synthetic/demo data for fast reproducibility.
      For full results, set USE_REAL_DATA=True and provide the DSA path.

Usage:
    python examples/dsa_physical_activity_adaptive_transfer.py

Requirements:
    pip install pyhealth torch numpy
"""

import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyhealth.models.adaptive_transfer_model import (
    AdaptiveTransferModel,
    LSTMBackbone,
    ResNetBackbone,
)

# ── Configuration ──────────────────────────────────────────────────────────────
USE_REAL_DATA  = False   # set True + fill path below for full results
DSA_DATA_PATH  = "/content/drive/MyDrive/CS598_DLH/Final_Project/Datasets/DSA/data"
DSA_PICKLE     = "/content/drive/MyDrive/CS598_DLH/Final_Project/dsa_dataset_dev.pkl"

EPOCHS_SOURCE  = 5       # increase to 50 for paper-level results
EPOCHS_TARGET  = 10      # increase to 100 for paper-level results
N_REPEATS      = 3       # increase to 15 to match paper
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Synthetic Data Helper ──────────────────────────────────────────────────────

def make_synthetic_datasets(
    n_samples: int = 200,
    n_channels: int = 9,
    timesteps: int = 125,
    n_classes: int = 19,
    n_source_domains: int = 4,
):
    """Create small synthetic datasets that mimic the DSA structure.

    Used for fast testing and reproducibility without the real dataset.
    Synthetic data has separable class structure via additive class signal.

    Args:
        n_samples: Samples per domain.
        n_channels: Sensor channels per sample.
        timesteps: Timesteps per sample.
        n_classes: Number of activity classes.
        n_source_domains: Number of source sensor domains.

    Returns:
        Tuple of (source_datasets dict, target_train, target_test).
    """
    def _make_ds(n, shift=0.0):
        X = torch.randn(n, n_channels, timesteps)
        y = torch.randint(0, n_classes, (n,))
        # Add class-dependent signal so model can learn
        for c in range(n_classes):
            mask = (y == c)
            X[mask] += (c * 0.1 + shift)
        return TensorDataset(X.float(), y.long())

    source_datasets = {
        "s{}".format(i+1): _make_ds(n_samples, shift=i * 0.3)
        for i in range(n_source_domains)
    }
    # Split target into train/test
    target_all   = _make_ds(n_samples)
    split        = int(n_samples * 0.8)
    target_train = TensorDataset(
        target_all.tensors[0][:split],
        target_all.tensors[1][:split],
    )
    target_test = TensorDataset(
        target_all.tensors[0][split:],
        target_all.tensors[1][split:],
    )
    return source_datasets, target_train, target_test


def load_real_datasets(target_sensor: str = "s2"):
    """Load real DSA datasets from pickle or raw files.

    Args:
        target_sensor: Sensor ID to use as target domain.

    Returns:
        Tuple of (source_datasets, target_train, target_test).
    """
    import pickle
    from pyhealth.tasks.physical_activity_task import PhysicalActivityTask

    if os.path.exists(DSA_PICKLE):
        with open(DSA_PICKLE, "rb") as f:
            dsa = pickle.load(f)
        # Override target sensor
        dsa.target_sensor = target_sensor
    else:
        from pyhealth.datasets.dsa_dataset import DSADataset
        dsa = DSADataset(root=DSA_DATA_PATH, target_sensor=target_sensor)

    task = PhysicalActivityTask(dsa)
    return (
        task.get_source_datasets(split="train"),
        task.get_target_dataset(split="train"),
        task.get_target_dataset(split="test"),
    )


def run_single_experiment(
    source_datasets,
    target_train,
    target_test,
    backbone: str = "lstm",
    distance: str = "euclidean",
    epochs_source: int = EPOCHS_SOURCE,
    epochs_target: int = EPOCHS_TARGET,
) -> float:
    """Run one full pipeline and return test RCC.

    Args:
        source_datasets: Dict of sensor_id -> dataset.
        target_train: Target domain training dataset.
        target_test: Target domain test dataset.
        backbone: "lstm" or "resnet".
        distance: "euclidean" or "dtw".
        epochs_source: Epochs per source domain.
        epochs_target: Fine-tuning epochs.

    Returns:
        float: Test RCC (accuracy).
    """
    model = AdaptiveTransferModel(
        backbone=backbone,
        distance=distance,
        epochs_per_source=epochs_source,
        epochs_target=epochs_target,
    )
    model.fit(source_datasets, target_train)
    return model.evaluate(target_test)


def print_results_table(title: str, results: dict) -> None:
    """Print a formatted results table.

    Args:
        title: Table title string.
        results: Dict of condition_name -> list of RCC floats.
    """
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)
    print("{:<25} {:>10} {:>10}".format("Condition", "Mean RCC", "Std RCC"))
    print("-" * 55)
    for name, scores in results.items():
        mean = np.mean(scores)
        std  = np.std(scores)
        print("{:<25} {:>10.4f} {:>10.4f}".format(name, mean, std))
    print("=" * 55)


# ── Ablation 1: Backbone Comparison ───────────────────────────────────────────

def ablation_backbone():
    """
    Ablation 1: LSTM vs ResNet Backbone
    ====================================
    Hypothesis: ResNet may generalize better across source domains due to
    its hierarchical feature extraction, while LSTM may better capture
    temporal dynamics. We test which backbone benefits more from
    IPD-guided pre-training.

    This ablation is NOVEL — the paper treats backbones as independent
    experiments, but does not directly compare their response to IPD
    guidance in a controlled setting.
    """
    print("\n" + "#" * 55)
    print("ABLATION 1: Backbone Comparison (LSTM vs ResNet)")
    print("#" * 55)
    print("Hypothesis: Which backbone benefits more from IPD guidance?")

    results = {"LSTM": [], "ResNet": []}

    for repeat in range(N_REPEATS):
        src, trn, tst = (
            load_real_datasets() if USE_REAL_DATA
            else make_synthetic_datasets()
        )
        for backbone, key in [("lstm", "LSTM"), ("resnet", "ResNet")]:
            rcc = run_single_experiment(src, trn, tst, backbone=backbone)
            results[key].append(rcc)
            print("  Repeat {}/{} | {} RCC={:.4f}".format(
                repeat+1, N_REPEATS, key, rcc
            ))

    print_results_table("Ablation 1 Results: Backbone Comparison", results)
    return results


# ── Ablation 2: Target Sensor Variation ───────────────────────────────────────

def ablation_target_sensor():
    """
    Ablation 2: Target Sensor Variation
    =====================================
    Hypothesis: Sensors on extremities (legs) are further from other body
    parts and thus have higher IPD to all source domains, making transfer
    harder. We expect leg sensors to show lower RCC than arm/torso sensors
    when used as the single-source target.

    This ablation is NOVEL — the paper always uses one fixed target sensor
    and does not study how target sensor choice affects performance.
    """
    print("\n" + "#" * 55)
    print("ABLATION 2: Target Sensor Variation")
    print("#" * 55)
    print("Hypothesis: Limb sensors suffer more than torso as single-source target.")

    sensor_names = {
        "s1": "Torso",
        "s2": "Right Arm",
        "s3": "Left Arm",
        "s4": "Right Leg",
        "s5": "Left Leg",
    }
    results = {v: [] for v in sensor_names.values()}

    for sid, sname in sensor_names.items():
        for repeat in range(N_REPEATS):
            if USE_REAL_DATA:
                src, trn, tst = load_real_datasets(target_sensor=sid)
            else:
                # Simulate different domain gaps with different shifts
                shift = {"s1": 0.0, "s2": 0.1, "s3": 0.1,
                         "s4": 0.5, "s5": 0.5}[sid]
                src, trn, tst = make_synthetic_datasets()
                # Add extra shift to target to simulate larger domain gap
                trn_X = trn.tensors[0] + shift
                tst_X = tst.tensors[0] + shift
                trn = torch.utils.data.TensorDataset(trn_X, trn.tensors[1])
                tst = torch.utils.data.TensorDataset(tst_X, tst.tensors[1])

            rcc = run_single_experiment(src, trn, tst)
            results[sname].append(rcc)
            print("  Repeat {}/{} | Target={} RCC={:.4f}".format(
                repeat+1, N_REPEATS, sname, rcc
            ))

    print_results_table("Ablation 2 Results: Target Sensor Variation", results)
    return results


# ── Ablation 3: IPD Distance Metric ───────────────────────────────────────────

def ablation_distance_metric():
    """
    Ablation 3: IPD Distance Metric (Euclidean vs DTW)
    ====================================================
    Hypothesis: DTW captures temporal alignment better than Euclidean
    distance, producing more accurate IPD scores and thus better learning
    rate guidance. However, Euclidean may be sufficient if activities have
    consistent temporal patterns.

    This extends the paper's Table 2 by isolating the effect of the
    distance metric specifically on IPD computation quality, rather than
    overall classification accuracy with different backbones.
    """
    print("\n" + "#" * 55)
    print("ABLATION 3: IPD Distance Metric (Euclidean vs DTW)")
    print("#" * 55)
    print("Hypothesis: DTW produces more accurate IPD guidance than Euclidean.")

    results = {"Euclidean": [], "DTW": []}

    for repeat in range(N_REPEATS):
        src, trn, tst = (
            load_real_datasets() if USE_REAL_DATA
            else make_synthetic_datasets()
        )
        for dist, key in [("euclidean", "Euclidean"), ("dtw", "DTW")]:
            rcc = run_single_experiment(src, trn, tst, distance=dist)
            results[key].append(rcc)
            print("  Repeat {}/{} | Distance={} RCC={:.4f}".format(
                repeat+1, N_REPEATS, key, rcc
            ))

    print_results_table("Ablation 3 Results: IPD Distance Metric", results)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("DSA Physical Activity - Ablation Study")
    print("Data mode: {}".format("REAL" if USE_REAL_DATA else "SYNTHETIC"))
    print("Repeats: {} | Epochs/source: {} | Epochs/target: {}".format(
        N_REPEATS, EPOCHS_SOURCE, EPOCHS_TARGET
    ))
    print("=" * 55)

    r1 = ablation_backbone()
    r2 = ablation_target_sensor()
    r3 = ablation_distance_metric()

    print("\n" + "=" * 55)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 55)
    print("Key Findings:")
    print("  Ablation 1 (Backbone):")
    for k, v in r1.items():
        print("    {} mean RCC = {:.4f}".format(k, np.mean(v)))
    print("  Ablation 2 (Target Sensor):")
    for k, v in r2.items():
        print("    {} mean RCC = {:.4f}".format(k, np.mean(v)))
    print("  Ablation 3 (Distance Metric):")
    for k, v in r3.items():
        print("    {} mean RCC = {:.4f}".format(k, np.mean(v)))
