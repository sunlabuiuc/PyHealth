"""Sleep staging with wav2sleep on NSRR / SHHS data.

This script demonstrates how to use the :class:`~pyhealth.models.Wav2Sleep`
model for multi-modal sleep stage classification.  It covers:

1. Loading real SHHS recordings via
   :func:`~pyhealth.tasks.sleep_staging_shhs_wav2sleep_fn` (ECG + ABD + THX).
2. Falling back to a fully synthetic dataset when SHHS data is unavailable,
   so the script always runs end-to-end.
3. Training Wav2Sleep and evaluating cross-modal generalisation (full-modality,
   ECG-only, respiratory-only subsets) using the same trained model.
4. An ablation study comparing ``feature_dim`` hyper-parameters.

**Getting SHHS data**
SHHS is freely available after registration at https://sleepdata.org/.
Set ``SHHS_ROOT`` below to the ``polysomnography/`` directory that contains
``edfs/shhs1/`` and ``annotations-events-profusion/shhs1/``.

**Paper reference**
Jonathan F. Carter and Lionel Tarassenko.
"wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification
from Physiological Signals." arXiv:2411.04644, 2024.
https://arxiv.org/abs/2411.04644
"""

import os

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import Wav2Sleep, load_shhs_samples
from pyhealth.trainer import Trainer

# ---------------------------------------------------------------------------
# Set this to your local SHHS polysomnography root to use real data.
# Leave as None to run with synthetic data instead.
# ---------------------------------------------------------------------------
SHHS_ROOT = None  # e.g. "/data/shhs/polysomnography"

# ---------------------------------------------------------------------------
# Sleep-stage label definitions (AASM 4-class, merging N1+N2 as "Light")
# ---------------------------------------------------------------------------
SLEEP_STAGES = {0: "Wake", 1: "Light (N1+N2)", 2: "N3 (Deep)", 3: "REM"}
NUM_CLASSES = len(SLEEP_STAGES)

# ---------------------------------------------------------------------------
# Signal parameters matching the wav2sleep paper
# ---------------------------------------------------------------------------
ECG_SAMPLES_PER_EPOCH = 256   # ~34 Hz × 30 s  (reduced for demo; paper: 1024)
RESP_SAMPLES_PER_EPOCH = 128  # ~8 Hz × 30 s   (reduced for demo; paper: 256)


# ---------------------------------------------------------------------------
# Helper: build a synthetic dataset with ECG + ABD + THX signals
# ---------------------------------------------------------------------------

def _build_synthetic_dataset(
    n_patients: int = 5,
    epochs_per_patient: int = 40,
    dataset_name: str = "synthetic_nsrr",
    seed: int = 0,
):
    """Create a synthetic SampleDataset that mirrors wav2sleep's signal schema.

    Each sample corresponds to one 30-second PSG epoch with:
    - ``ecg``  : shape ``(1, ECG_SAMPLES_PER_EPOCH)``  — electrocardiogram
    - ``abd``  : shape ``(1, RESP_SAMPLES_PER_EPOCH)`` — abdominal belt
    - ``thx``  : shape ``(1, RESP_SAMPLES_PER_EPOCH)`` — thoracic belt
    - ``label``: integer sleep stage (0=Wake, 1=Light, 2=N3, 3=REM)

    Args:
        n_patients: Number of synthetic patients.
        epochs_per_patient: Number of 30-second epochs per patient.
        dataset_name: Dataset name passed to ``create_sample_dataset``.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`~pyhealth.datasets.SampleDataset` ready for model training.
    """
    rng = np.random.RandomState(seed)
    samples = []

    for pid in range(n_patients):
        for epoch_idx in range(epochs_per_patient):
            label = rng.randint(0, NUM_CLASSES)
            samples.append(
                {
                    "patient_id": f"patient-{pid:03d}",
                    "visit_id": f"epoch-{epoch_idx:04d}",
                    # ECG: 1-channel, ECG_SAMPLES_PER_EPOCH samples
                    "ecg": rng.randn(1, ECG_SAMPLES_PER_EPOCH).astype(np.float32),
                    # Abdominal respiratory effort
                    "abd": rng.randn(1, RESP_SAMPLES_PER_EPOCH).astype(np.float32),
                    # Thoracic respiratory effort
                    "thx": rng.randn(1, RESP_SAMPLES_PER_EPOCH).astype(np.float32),
                    "label": label,
                }
            )

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "ecg": "tensor",   # raw ECG waveform
            "abd": "tensor",   # abdominal respiratory signal
            "thx": "tensor",   # thoracic respiratory signal
        },
        output_schema={"label": "multiclass"},
        dataset_name=dataset_name,
    )
    return dataset


# ---------------------------------------------------------------------------
# Modality-subset evaluation helper
# ---------------------------------------------------------------------------

def evaluate_modality_subset(model, loader, modality_keys):
    """Compute mean cross-entropy loss over *loader* for a modality subset.

    Modalities not in *modality_keys* are removed from each batch before the
    forward pass, demonstrating wav2sleep's ability to degrade gracefully when
    signals are missing.

    Args:
        model: Trained :class:`~pyhealth.models.Wav2Sleep` instance.
        loader: A PyHealth DataLoader over the evaluation split.
        modality_keys: Iterable of modality names to keep (e.g. ``["ecg"]``).

    Returns:
        Mean loss (float) and mean accuracy (float) over all batches.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            # Keep only the requested modalities
            filtered = {
                k: v for k, v in batch.items() if k in modality_keys or k == "label"
            }
            out = model(**filtered)
            total_loss += out["loss"].item() * out["y_true"].shape[0]
            preds = out["y_prob"].argmax(dim=-1)
            total_correct += (preds == out["y_true"]).sum().item()
            total_samples += out["y_true"].shape[0]

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Main: full pipeline + ablation study
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("wav2sleep – Multi-Modal Sleep Staging Demo")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Build dataset (real SHHS or synthetic fallback)
    # -----------------------------------------------------------------------
    if SHHS_ROOT and os.path.isdir(SHHS_ROOT):
        print(f"\n[1] Loading SHHS data from {SHHS_ROOT} …")
        # wav2sleep 4-class label map: merge N1+N2 as "Light"
        label_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3}
        samples = load_shhs_samples(
            shhs_root=SHHS_ROOT,
            epoch_seconds=30,
            ecg_samples_per_epoch=ECG_SAMPLES_PER_EPOCH,
            resp_samples_per_epoch=RESP_SAMPLES_PER_EPOCH,
            max_recordings=20,   # set to None to load all recordings
            label_map=label_map,
        )
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "ecg": "tensor",
                "abd": "tensor",
                "thx": "tensor",
            },
            output_schema={"label": "multiclass"},
            dataset_name="shhs_wav2sleep",
        )
    else:
        print("\n[1] SHHS_ROOT not set — using synthetic data …")
        dataset = _build_synthetic_dataset(
            n_patients=5, epochs_per_patient=40, seed=42
        )

    print(f"    Total samples : {len(dataset)}")
    print(f"    Input modalities: {list(dataset.input_processors.keys())}")
    print(f"    Label          : multiclass ({NUM_CLASSES} classes)")

    # Simple 80/20 train-val split (no per-patient guarantee for this demo)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_total))

    train_loader = get_dataloader(
        dataset, batch_size=16, shuffle=True, indices=train_indices
    )
    val_loader = get_dataloader(
        dataset, batch_size=16, shuffle=False, indices=val_indices
    )

    # -----------------------------------------------------------------------
    # 2. Train the unified (all-modality) wav2sleep model
    # -----------------------------------------------------------------------
    print("\n[2] Training wav2sleep (ECG + ABD + THX) …")
    model = Wav2Sleep(
        dataset=dataset,
        feature_dim=128,
        n_transformer_layers=2,
        n_attention_heads=8,
        transformer_ff_dim=512,
        dropout=0.1,
    )
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=5,
        optimizer_params={"lr": 1e-3},
        weight_decay=1e-2,
        monitor="accuracy",
    )

    # -----------------------------------------------------------------------
    # 3. Cross-modal evaluation: same model, different modality subsets
    # -----------------------------------------------------------------------
    print("\n[3] Cross-modal evaluation on validation set …")
    subsets = {
        "All (ECG + ABD + THX)": ["ecg", "abd", "thx"],
        "ECG + THX only"        : ["ecg", "thx"],
        "ECG only"              : ["ecg"],
        "ABD + THX only"        : ["abd", "thx"],
    }
    for name, keys in subsets.items():
        loss, acc = evaluate_modality_subset(model, val_loader, keys)
        print(f"    {name:<28s}  loss={loss:.4f}  acc={acc:.3f}")

    # -----------------------------------------------------------------------
    # 4. Ablation: feature_dim hyper-parameter
    # -----------------------------------------------------------------------
    print("\n[4] Ablation – feature_dim hyper-parameter …")
    print(
        f"    {'feature_dim':<14s}  {'#params':>10s}  {'val_loss':>10s}"
        f"  {'val_acc':>8s}"
    )

    for fdim, n_heads in [(32, 4), (64, 4), (128, 8), (256, 8)]:
        abl_model = Wav2Sleep(
            dataset=dataset,
            feature_dim=fdim,
            n_transformer_layers=2,
            n_attention_heads=n_heads,
            dropout=0.1,
        )
        abl_trainer = Trainer(model=abl_model)
        abl_trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=3,
            optimizer_params={"lr": 1e-3},
            weight_decay=1e-2,
        )
        loss, acc = evaluate_modality_subset(
            abl_model, val_loader, ["ecg", "abd", "thx"]
        )
        n_params = sum(p.numel() for p in abl_model.parameters())
        print(f"    {fdim:<14d}  {n_params:>10,}  {loss:>10.4f}  {acc:>8.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
