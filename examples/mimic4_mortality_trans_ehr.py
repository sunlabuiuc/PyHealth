"""TransEHR Ablation Study: In-Hospital Mortality Prediction on MIMIC-IV.

This script demonstrates and ablates the :class:`~pyhealth.models.TransEHR`
model on the in-hospital mortality prediction task using either:

* **Real MIMIC-IV data** (if you have PhysioNet credentials and the dataset
  downloaded), or
* **Synthetic demo data** (default) that runs without any data download.

Paper:
    Xu et al. "TransEHR: Self-Supervised Transformer for Clinical Time
    Series Data", PMLR 2023.
    https://proceedings.mlr.press/v209/xu23a.html

Usage (demo / synthetic data, no MIMIC needed)::

    python mimic4_mortality_trans_ehr.py

Usage (real MIMIC-IV data)::

    python mimic4_mortality_trans_ehr.py --mimic_dir /path/to/mimic-iv-2.2

Ablation Study:
    We vary three hyperparameters and compare AUROC on a held-out test split:

    1. **num_layers**   — 1 vs 2 vs 4 transformer layers
    2. **embedding_dim** — 64 vs 128 vs 256
    3. **num_heads**    — 2 vs 4 vs 8

    All other hyperparameters are held at their defaults.  Results are
    printed as a summary table at the end.

    TransEHR key novelty vs. the existing PyHealth Transformer:
    * Accepts ``nested_sequence`` inputs preserving visit-level temporal
      structure (patient → visits → codes) instead of a flat code list.
    * Each visit is aggregated via mean-pooling before the transformer, so
      attention operates over visits, not individual codes.
    * Sinusoidal positional encoding over visit order captures the
      longitudinal nature of clinical trajectories.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch

from pyhealth.models.trans_ehr import TransEHR
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics.binary import binary_metrics_fn
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# Synthetic data generator (runs without any real dataset)
# ---------------------------------------------------------------------------

def make_synthetic_samples(
    n_patients: int = 200,
    max_visits: int = 5,
    max_codes: int = 6,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic EHR samples for demo / testing.

    Each sample has:
    * ``conditions``: nested list of ICD-style codes per visit.
    * ``procedures``: nested list of CPT-style codes per visit.
    * ``label``: binary mortality label (roughly 20 % positive rate).

    Args:
        n_patients: Number of synthetic patients to generate.
        max_visits: Maximum number of visits per patient.
        max_codes: Maximum number of codes per visit.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dictionaries compatible with
        :func:`~pyhealth.datasets.create_sample_dataset`.
    """
    import random

    random.seed(seed)
    condition_vocab = [f"ICD{i:04d}" for i in range(50)]
    procedure_vocab = [f"CPT{i:04d}" for i in range(30)]

    samples = []
    for i in range(n_patients):
        n_visits = random.randint(1, max_visits)
        conditions = [
            random.sample(condition_vocab, random.randint(1, max_codes))
            for _ in range(n_visits)
        ]
        procedures = [
            random.sample(procedure_vocab, random.randint(1, max_codes))
            for _ in range(n_visits)
        ]
        # Mortality label: 20 % base rate with a weak signal
        label = 1 if (random.random() < 0.2 + 0.15 * (n_visits > 3)) else 0
        samples.append(
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": conditions,
                "procedures": procedures,
                "label": label,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_demo_dataset(
    n_train: int = 150,
    n_val: int = 25,
    n_test: int = 25,
) -> Tuple:
    """Return (train_ds, val_ds, test_ds) with synthetic data."""
    all_samples = make_synthetic_samples(n_patients=n_train + n_val + n_test)
    input_schema = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
    }
    output_schema = {"label": "binary"}

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train : n_train + n_val]
    test_samples = all_samples[n_train + n_val :]

    train_ds = create_sample_dataset(
        train_samples, input_schema, output_schema, dataset_name="mimic4_mortality_train"
    )
    val_ds = create_sample_dataset(
        val_samples, input_schema, output_schema, dataset_name="mimic4_mortality_val"
    )
    test_ds = create_sample_dataset(
        test_samples, input_schema, output_schema, dataset_name="mimic4_mortality_test"
    )
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(model: TransEHR, dataset, batch_size: int = 32) -> float:
    """Compute AUROC on a dataset split.

    Args:
        model: A trained TransEHR instance.
        dataset: A PyHealth SampleDataset split to evaluate.
        batch_size: Evaluation batch size.

    Returns:
        AUROC score as a float.
    """
    model.eval()
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            all_probs.append(out["y_prob"].cpu())
            all_labels.append(out["y_true"].cpu())

    y_prob = torch.cat(all_probs).numpy().squeeze()
    y_true = torch.cat(all_labels).numpy().squeeze()

    metrics = binary_metrics_fn(y_true, y_prob, metrics=["roc_auc"])
    return metrics["roc_auc"]


def train_and_eval(
    train_ds,
    val_ds,
    test_ds,
    embedding_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    feedforward_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, float]:
    """Train a TransEHR model and return val/test AUROC.

    Args:
        train_ds: Training split.
        val_ds: Validation split.
        test_ds: Test split.
        embedding_dim: Token embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        dropout: Dropout probability.
        feedforward_dim: Feed-forward inner dimension.
        lr: Adam learning rate.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        Dictionary with ``"val_auroc"`` and ``"test_auroc"`` keys.
    """
    model = TransEHR(
        dataset=train_ds,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        feedforward_dim=feedforward_dim,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)

    # --- Training loop ---
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(**{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                           for k, v in batch.items()})
            out["loss"].backward()
            optimizer.step()
            total_loss += out["loss"].item()
        avg_loss = total_loss / len(loader)
        val_auroc = evaluate(model, val_ds)
        print(f"  epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_roc_auc={val_auroc:.4f}")

    test_auroc = evaluate(model, test_ds)
    return {"val_auroc": val_auroc, "test_auroc": test_auroc}


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(
    train_ds,
    val_ds,
    test_ds,
    device: str = "cpu",
    epochs: int = 5,
) -> None:
    """Run the three ablation experiments and print a comparison table.

    Ablation 1: Number of transformer layers  (1, 2, 4)
    Ablation 2: Embedding dimension           (64, 128, 256)
    Ablation 3: Number of attention heads     (2, 4, 8)

    Args:
        train_ds: Training split.
        val_ds: Validation split.
        test_ds: Test split.
        device: Torch device string.
        epochs: Training epochs per configuration.
    """
    results = []

    # ----------------------------------------------------------------
    # Ablation 1 — number of transformer layers
    #
    # Hypothesis: more layers allow the model to learn more abstract
    # representations of visit sequences, but too many layers may
    # overfit on small datasets.  The TransEHR paper uses 2 layers in
    # its primary configuration.  We test 1, 2, and 4 layers while
    # holding embedding_dim=128 and num_heads=4 fixed.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ablation 1: Number of transformer layers")
    print("=" * 60)
    for n_layers in [1, 2, 4]:
        print(f"\n  num_layers={n_layers}")
        t0 = time.time()
        scores = train_and_eval(
            train_ds, val_ds, test_ds,
            embedding_dim=128, num_heads=4, num_layers=n_layers,
            epochs=epochs, device=device,
        )
        elapsed = time.time() - t0
        results.append({
            "ablation": "num_layers",
            "value": n_layers,
            **scores,
            "time_s": elapsed,
        })

    # ----------------------------------------------------------------
    # Ablation 2 — embedding dimension
    #
    # Hypothesis: larger embeddings capture richer code semantics, but
    # require more data to train without overfitting.  We test 64, 128,
    # and 256 while holding num_layers=2 fixed.  num_heads is set to 4
    # for all sizes since 4 divides each evenly.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ablation 2: Embedding dimension")
    print("=" * 60)
    for emb_dim in [64, 128, 256]:
        # num_heads=4 must divide embedding_dim
        n_heads = 4 if emb_dim >= 64 else 2
        print(f"\n  embedding_dim={emb_dim}")
        t0 = time.time()
        scores = train_and_eval(
            train_ds, val_ds, test_ds,
            embedding_dim=emb_dim, num_heads=n_heads, num_layers=2,
            epochs=epochs, device=device,
        )
        elapsed = time.time() - t0
        results.append({
            "ablation": "embedding_dim",
            "value": emb_dim,
            **scores,
            "time_s": elapsed,
        })

    # ----------------------------------------------------------------
    # Ablation 3 — number of attention heads
    #
    # Hypothesis: more attention heads allow the model to attend to
    # different aspects of the visit sequence simultaneously (e.g.,
    # recent vs. distant visits, different code types).  We test 2, 4,
    # and 8 heads with embedding_dim=128 fixed, since 128 is divisible
    # by all three values.
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Ablation 3: Number of attention heads")
    print("=" * 60)
    for n_heads in [2, 4, 8]:
        # embedding_dim must be divisible by num_heads; use 128
        print(f"\n  num_heads={n_heads}")
        t0 = time.time()
        scores = train_and_eval(
            train_ds, val_ds, test_ds,
            embedding_dim=128, num_heads=n_heads, num_layers=2,
            epochs=epochs, device=device,
        )
        elapsed = time.time() - t0
        results.append({
            "ablation": "num_heads",
            "value": n_heads,
            **scores,
            "time_s": elapsed,
        })

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n\n" + "=" * 65)
    print("ABLATION STUDY SUMMARY — TransEHR (in-hospital mortality)")
    print("=" * 65)
    print(f"{'Ablation':<20} {'Value':>8} {'Val ROC-AUC':>13} {'Test ROC-AUC':>13}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['ablation']:<20} {r['value']:>8} "
            f"{r['val_auroc']:>13.4f} {r['test_auroc']:>13.4f}"
        )
    print("=" * 65)
    print("\nNote: Results on synthetic demo data. Real MIMIC-IV results will differ.")
    print("To run with MIMIC-IV: python mimic4_mortality_trans_ehr.py --mimic_dir <path>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TransEHR ablation study on MIMIC-IV mortality prediction"
    )
    parser.add_argument(
        "--mimic_dir",
        type=str,
        default=None,
        help="Path to MIMIC-IV dataset root (required for real data). "
             "If not provided, synthetic demo data is used.",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Training epochs per configuration."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Epochs per config: {args.epochs}")

    if args.mimic_dir is not None:
        # ----------------------------------------------------------------
        # Real MIMIC-IV data path (requires PhysioNet credentials + download)
        # ----------------------------------------------------------------
        print(f"\nLoading MIMIC-IV from: {args.mimic_dir}")
        try:
            from pyhealth.datasets import MIMIC4Dataset
            from pyhealth.tasks import mortality_prediction_mimic4_fn

            raw = MIMIC4Dataset(
                root=args.mimic_dir,
                tables=["diagnoses_icd", "procedures_icd"],
                code_mapping={"ICD10CM": "CCSCM", "ICD10PCS": "CCSPROC"},
            )
            task_ds = raw.set_task(mortality_prediction_mimic4_fn)
            # Build nested_sequence schema (one sample = one patient's last N visits)
            # This wrapper converts the task dataset to nested_sequence format
            # for TransEHR.  Adapt based on actual MIMIC4 preprocessing output.
            print("MIMIC-IV loaded. Using first 3000 patients for demo.")
            samples = [task_ds[i] for i in range(min(3000, len(task_ds)))]
            import random
            random.shuffle(samples)
            n = len(samples)
            n_train, n_val = int(0.7 * n), int(0.15 * n)
            # NOTE: task_ds samples may need adaptation to nested_sequence format.
            # See PyHealth MIMIC4 documentation for field names.
        except Exception as e:
            print(f"MIMIC-IV loading failed: {e}")
            print("Falling back to synthetic demo data.\n")
            args.mimic_dir = None

    if args.mimic_dir is None:
        print("\nUsing synthetic demo data (200 patients).")
        print("Generating samples...")
        train_ds, val_ds, test_ds = load_demo_dataset(
            n_train=150, n_val=25, n_test=25
        )
        print(
            f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} samples"
        )

    run_ablation(train_ds, val_ds, test_ds, device=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
