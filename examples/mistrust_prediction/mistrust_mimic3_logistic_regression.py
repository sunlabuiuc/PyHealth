"""
Medical Mistrust Prediction on MIMIC-III
=========================================
End-to-end example reproducing the interpersonal-feature mistrust classifiers
from Boag et al. 2018 "Racial Disparities and Mistrust in End-of-Life Care"
using the PyHealth LogisticRegression model with L1 regularisation.

Two tasks are demonstrated:
  1. Noncompliance prediction  — label from "noncompliant" in NOTEEVENTS
  2. Autopsy-consent prediction — label from autopsy consent/decline in NOTEEVENTS

Both use the same interpersonal CHARTEVENTS feature representation, mirroring
the original trust.ipynb pipeline.

Paper:  https://arxiv.org/abs/1808.03827
GitHub: https://github.com/wboag/eol-mistrust

Requirements
------------
  - MIMIC-III v1.4 access via PhysioNet
  - pyhealth installed (pip install pyhealth)

Usage
-----
    # With real MIMIC-III data:
    python mistrust_mimic3_logistic_regression.py \\
        --mimic3_root /path/to/physionet.org/files/mimiciii/1.4

    # Smoke-test with synthetic MIMIC-III (no data access needed):
    python mistrust_mimic3_logistic_regression.py --synthetic
"""

import argparse
import tempfile

from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.tasks import (
    MistrustNoncomplianceMIMIC3,
    MistrustAutopsyMIMIC3,
    build_interpersonal_itemids,
)
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# L1 lambda equivalence to sklearn C=0.1:
#   l1_lambda = 1 / (C * n_train)  ≈  10 / n_train
# We use a fixed value here; tune based on actual training set size.
# ---------------------------------------------------------------------------
L1_LAMBDA_NONCOMPLIANCE = 2.62e-4   # 10 / 38_157  (paper's 70% of 54,510)
L1_LAMBDA_AUTOPSY       = 1.43e-2   # 10 /    697  (paper's 70% of 1,009)
EMBEDDING_DIM           = 128
BATCH_SIZE              = 256
EPOCHS                  = 50


def run_task(task_name: str, sample_dataset, l1_lambda: float) -> None:
    """Split, train, and evaluate one mistrust task."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  samples: {len(sample_dataset)}")
    print(f"  l1_lambda = {l1_lambda:.2e}  (equiv. sklearn C = {1/l1_lambda:.1f} / n_train)")
    print(f"{'='*60}")

    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.7, 0.15, 0.15])

    train_loader = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train / Val / Test : {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    model = LogisticRegression(
        dataset=sample_dataset,
        embedding_dim=EMBEDDING_DIM,
        l1_lambda=l1_lambda,
    )
    print(f"  Model parameters   : {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=EPOCHS,
        monitor="roc_auc",
    )

    metrics = trainer.evaluate(test_loader)
    print(f"\n  Test metrics ({task_name}):")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")


def main(mimic3_root: str, synthetic: bool) -> None:
    # ------------------------------------------------------------------
    # STEP 1: Load MIMIC-III dataset
    # ------------------------------------------------------------------
    if synthetic:
        print("Loading synthetic MIMIC-III (no PhysioNet access needed) ...")
        root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
        cache_dir = tempfile.mkdtemp()
        dev = True
    else:
        root = mimic3_root
        cache_dir = None
        dev = False

    base_dataset = MIMIC3Dataset(
        root=root,
        tables=["CHARTEVENTS", "NOTEEVENTS"],
        cache_dir=cache_dir,
        dev=dev,
    )
    base_dataset.stats()

    # ------------------------------------------------------------------
    # STEP 2: Build interpersonal itemid → label mapping from D_ITEMS
    # ------------------------------------------------------------------
    if synthetic:
        # Synthetic dataset has no D_ITEMS; use an empty dict — features
        # will be absent and most samples will be empty (smoke-test only).
        print("\nWARNING: Synthetic mode — interpersonal features will be empty.")
        print("         This is a pipeline smoke-test only, not a valid experiment.")
        itemid_to_label = {}
    else:
        d_items_path = f"{mimic3_root}/D_ITEMS.csv.gz"
        print(f"\nBuilding interpersonal itemid map from {d_items_path} ...")
        itemid_to_label = build_interpersonal_itemids(d_items_path)
        print(f"  Matched {len(itemid_to_label)} interpersonal ITEMIDs")

    # ------------------------------------------------------------------
    # STEP 3: Noncompliance task
    # ------------------------------------------------------------------
    nc_task = MistrustNoncomplianceMIMIC3(
        itemid_to_label=itemid_to_label,
        min_features=1,
    )
    nc_dataset = base_dataset.set_task(nc_task)

    if len(nc_dataset) == 0:
        print("\nNoncompliance task: no samples generated (expected in synthetic mode)")
    else:
        run_task("NoncompliantMistrust", nc_dataset, l1_lambda=L1_LAMBDA_NONCOMPLIANCE)

    # ------------------------------------------------------------------
    # STEP 4: Autopsy-consent task
    # ------------------------------------------------------------------
    au_task = MistrustAutopsyMIMIC3(
        itemid_to_label=itemid_to_label,
        min_features=1,
    )
    au_dataset = base_dataset.set_task(au_task)

    if len(au_dataset) == 0:
        print("\nAutopsy task: no samples generated (expected in synthetic mode)")
    else:
        run_task("AutopsyConsentMistrust", au_dataset, l1_lambda=L1_LAMBDA_AUTOPSY)

    # ------------------------------------------------------------------
    # STEP 5: Paper-equivalent evaluation notes
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Paper-equivalent evaluation notes")
    print("="*60)
    print("""
  Boag et al. 2018 used sklearn LogisticRegression(C=0.1, penalty='l1')
  trained on 54,510 patients (all with interpersonal chartevents).
  Equivalent PyHealth setup:

      model = LogisticRegression(
          dataset=sample_dataset,
          embedding_dim=128,
          l1_lambda=10 / len(train_dataset),   # = 1/(C * n_train), C=0.1
      )

  Expected test AUC-ROC (paper Table 4 / PROGRESS.md):
    Noncompliance : 0.667
    Autopsy       : 0.531

  Higher AUC than sklearn is possible because PyHealth uses learned
  embeddings (128-dim) rather than 1-hot DictVectorizer features,
  giving the model richer representations of the feature vocabulary.
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mistrust prediction with PyHealth LogisticRegression + L1"
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        default=None,
        help="Path to MIMIC-III v1.4 directory (required unless --synthetic)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic MIMIC-III for pipeline smoke-test (no PhysioNet access needed)",
    )
    args = parser.parse_args()

    if not args.synthetic and args.mimic3_root is None:
        parser.error("Provide --mimic3_root or pass --synthetic for smoke-test mode")

    main(mimic3_root=args.mimic3_root, synthetic=args.synthetic)
