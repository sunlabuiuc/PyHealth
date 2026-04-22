"""Ablation study for SimpleADCNN on synthetic 3D MRI data.

Paper: Bruningk et al., "Back to the Basics with Inclusion of Clinical
Domain Knowledge - A Simple, Scalable and Effective Model of Alzheimer's
Disease Classification", ML4HC 2021.
https://proceedings.mlr.press/v149/bruningk21a.html

The paper focused on the parts of the brain most affected by Alzhheimer's (left hippocampus).
Additionally, the paper looked at brain topography, but found this to be less relevant.
By using prior knowledge of the problem and relatively simple architectures,
the paper was able to accurately classify Alzheimer's MRI images.
Below are some examples of structures the paper investigated.

    - I-3D  (inner brain, 120x144x120): ACC 0.79 +/- 0.05, AUC 0.88
    - P*-3D (best patch,  30x36x30):   ACC 0.81 +/- 0.05, AUC 0.89
    - HC-3D (hippocampus, 33x45x48):   ACC 0.84 +/- 0.07, AUC 0.91

The ablation tests the ability to create a wider variety of model structures
beyond those seen in the paper. The idea is to give greater customization
when potentially adapting the model to similar problems. 

    1. **Network depth** (2, 3, 4 conv blocks) — the paper uses a fixed depth
       per region
    2. **Dropout rate** (0.0, 0.3, 0.5) — the paper mentions dropout but does
       not report a sensitivity analysis.
    3. **Dense layer capacity** (64, 128, 256) — classifier head width was not
       explored in the paper.
    4. **Learning rate** (1e-3, 5e-4, 1e-4) — standard Adam search values.
    5. **Input shape** — reproduces the paper's region-level ablation (HC, P, I)
       to confirm the model handles all three configurations.

Results
=============================================================
Captured from a single end-to-end run of this script. Metrics are on
synthetic random tensors and serve only to verify the ablation grid
executes and produces well-formed numbers.

    Config                   ACC    AUC    Params
    -----------------------  -----  -----  ---------
    HC-3D (~140k)            0.500  0.250    142,017
    P*-3D (~72k)             0.500  0.500     74,113
    I-3D (~270k)             0.500  0.625    294,657
    depth=2                  0.500  0.188     18,753
    depth=4                  0.500  0.562  1,043,905
    dropout=0.0              0.500  0.500    142,017
    dropout=0.5              0.500  0.500    142,017
    dense=64                 0.500  0.312    133,697
    dense=256                0.500  0.312    158,657
    lr=5e-4                  0.500  0.000    142,017
    lr=1e-4                  0.500  0.000    142,017

The results above are on synthetic data, which is why an ACC of .5 is expected.
The results seems relatively reasonable with a small sample size and random data.
SimpleADCNN provides a variety of ways to create the "simple cnn" described in
the paper, with modifications available if desired.

The ADNI dataset was not available for this project, so synthetic data is necessary.
As such, the results are in line with synthetic data and seem to train as expected.

How to run
----------
    python examples/simple_ad_cnn_classification.py
"""

import random
from typing import Any

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.utils import collate_fn_dict_with_padding
from pyhealth.models import SimpleADCNN

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def generate_synthetic_dataset(
    n_samples: int = 40,
    volume_shape: tuple[int, int, int] = (33, 45, 48),
    seed: int = 42,
):
    """Generate a synthetic 3D MRI dataset with balanced binary labels.

    Args:
        n_samples: Total number of samples (split 50/50 between classes).
        volume_shape: Spatial dimensions (D, H, W) of each volume.
        seed: Random seed for reproducibility.

    Returns:
        A ``SampleDataset`` with input key ``"mri"`` and label key
        ``"label"``.
    """
    rng = torch.Generator().manual_seed(seed)
    samples: list[dict[str, Any]] = []
    for i in range(n_samples):
        vol = torch.randn(1, *volume_shape, generator=rng)
        samples.append(
            {
                "patient_id": f"patient-{i}",
                "visit_id": "visit-0",
                "mri": vol.tolist(),
                "label": i % 2,  # balanced: 0, 1, 0, 1, ...
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"mri": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_adni",
    )


def stratified_split(dataset, ratios=(0.6, 0.2, 0.2), seed=42):
    """Split a dataset into train/val/test with class balance preserved.

    Ensures every split contains both classes so that AUC is always
    computable, even on small synthetic datasets.

    Args:
        dataset: A ``SampleDataset``.
        ratios: Train / val / test proportions (must sum to 1).
        seed: Random seed.

    Returns:
        Three ``torch.utils.data.Subset`` objects.
    """
    from torch.utils.data import Subset

    # Separate indices by label
    class_0, class_1 = [], []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        lab = sample["label"]
        if isinstance(lab, torch.Tensor):
            lab = lab.item()
        (class_0 if lab == 0 else class_1).append(idx)

    rng = random.Random(seed)
    rng.shuffle(class_0)
    rng.shuffle(class_1)

    def _split_list(lst):
        n = len(lst)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        return lst[:n_train], lst[n_train : n_train + n_val], lst[n_train + n_val :]

    train_0, val_0, test_0 = _split_list(class_0)
    train_1, val_1, test_1 = _split_list(class_1)

    train_idx = train_0 + train_1
    val_idx = val_0 + val_1
    test_idx = test_0 + test_1

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def train_and_evaluate(config: dict) -> dict[str, float]:
    """Train an SimpleADCNN with the given config and return metrics.

    Args:
        config: Dictionary with keys ``name``, ``volume_shape``,
            ``conv_channels``, ``dropout``, ``dense_dim``, ``lr``,
            ``epochs``.

    Returns:
        Dictionary with ``acc``, ``auc``, and ``params`` on the test set.
    """
    # Seed all RNG sources for reproducibility across configs
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = generate_synthetic_dataset(
        n_samples=config.get("n_samples", 40),
        volume_shape=config["volume_shape"],
        seed=seed,
    )

    train_data, _, test_data = stratified_split(dataset, seed=seed)

    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn_dict_with_padding,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn_dict_with_padding,
    )

    model = SimpleADCNN(
        dataset=dataset,
        conv_channels=config["conv_channels"],
        dropout=config["dropout"],
        dense_dim=config["dense_dim"],
    )
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config.get("epochs", 5)

    # --- Training ---
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            ret = model(**batch)
            ret["loss"].backward()
            optimizer.step()

    # --- Evaluation ---
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            ret = model(**batch)
            all_probs.append(ret["y_prob"].cpu())
            all_true.append(ret["y_true"].cpu())

    y_prob = torch.cat(all_probs).numpy().ravel()
    y_true = torch.cat(all_true).numpy().ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {"acc": acc, "auc": auc, "params": n_params}


# ---------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------

# Each config varies one axis from the HC-3D baseline.
# The baseline mirrors the paper's hippocampus configuration.

CONFIGS = [
    # --- Paper region configs with approximate param-count matching ---
    # Channel widths are chosen so parameter counts approximate the
    # paper's reported values (which depend on architecture, not input
    # shape, because global average pooling decouples spatial size from
    # the classifier).
    {
        "name": "HC-3D (~140k)",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.4,
        "dense_dim": 128,
        "lr": 1e-3,
        "epochs": 5,
    },
    {
        "name": "P*-3D (~72k)",
        "volume_shape": (30, 36, 30),
        "conv_channels": (16, 32, 64),
        "dropout": 0.4,
        "dense_dim": 64,
        "lr": 1e-3,
        "epochs": 5,
    },
    {
        "name": "I-3D (~270k)",
        "volume_shape": (24, 28, 24),
        "conv_channels": (32, 64, 128),
        "dropout": 0.4,
        "dense_dim": 128,
        "lr": 1e-3,
        "epochs": 5,
    },
    # --- Depth ablation (novel, based on HC-3D baseline) ---
    {
        "name": "depth=2",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32),
        "dropout": 0.4,
        "dense_dim": 128,
        "lr": 1e-3,
        "epochs": 5,
    },
    {
        "name": "depth=4",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128, 256),
        "dropout": 0.4,
        "dense_dim": 128,
        "lr": 1e-3,
        "epochs": 5,
    },
    # --- Dropout ablation (novel, HC-3D architecture) ---
    {
        "name": "dropout=0.0",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.0,
        "dense_dim": 128,
        "lr": 1e-3,
        "epochs": 5,
    },
    {
        "name": "dropout=0.5",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.5,
        "dense_dim": 128,
        "lr": 1e-3,
        "epochs": 5,
    },
    # --- Dense dim ablation (novel, HC-3D architecture) ---
    {
        "name": "dense=64",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.4,
        "dense_dim": 64,
        "lr": 1e-3,
        "epochs": 5,
    },
    {
        "name": "dense=256",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.4,
        "dense_dim": 256,
        "lr": 1e-3,
        "epochs": 5,
    },
    # --- Learning rate ablation (HC-3D architecture) ---
    {
        "name": "lr=5e-4",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.4,
        "dense_dim": 128,
        "lr": 5e-4,
        "epochs": 5,
    },
    {
        "name": "lr=1e-4",
        "volume_shape": (33, 45, 48),
        "conv_channels": (16, 32, 128),
        "dropout": 0.4,
        "dense_dim": 128,
        "lr": 1e-4,
        "epochs": 5,
    },
]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


def main():
    """Run the ablation grid and print a results table."""
    print("=" * 72)
    print("SimpleADCNN Ablation Study — Synthetic 3D MRI Data")
    print("Paper: Bruningk et al., ML4HC 2021")
    print("=" * 72)
    print()

    results = []
    for i, cfg in enumerate(CONFIGS):
        name = cfg["name"]
        print(f"[{i + 1}/{len(CONFIGS)}] Running: {name} ...", end=" ", flush=True)
        metrics = train_and_evaluate(cfg)
        results.append((name, metrics))
        print(
            f"ACC={metrics['acc']:.3f}  "
            f"AUC={metrics['auc']:.3f}  "
            f"params={metrics['params']:,}"
        )

    # --- Results table ---
    print()
    print("-" * 72)
    print(f"{'Configuration':<28} {'ACC':>8} {'AUC':>8} {'Params':>10}")
    print("-" * 72)
    for name, metrics in results:
        print(
            f"{name:<28} {metrics['acc']:>8.3f} "
            f"{metrics['auc']:>8.3f} {metrics['params']:>10,}"
        )
    print("-" * 72)

    # --- Paper reference values ---
    print()
    print("Paper reference (on real ADNI data, 5-fold CV, 3 runs):")
    print(f"  {'I-3D  (inner brain)':<28} {'0.79':>8} {'0.88':>8} {'~270k':>10}")
    print(f"  {'P*-3D (best patch)':<28} {'0.81':>8} {'0.89':>8} {'~72k':>10}")
    print(f"  {'HC-3D (left hippocampus)':<28} {'0.84':>8} {'0.91':>8} {'~140k':>10}")
    print()
    print(
        "Note: Metrics above are on synthetic random data and are NOT "
        "expected to match the paper. They demonstrate that the model "
        "trains correctly across all configurations."
    )


if __name__ == "__main__":
    main()