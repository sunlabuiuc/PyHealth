"""
Name: Ranjithkumar Rajendran
NetID: rr54
Paper: KEEP (CHIL 2025) — Elhussein et al.

Ablation 2 — Architecture Comparison.
Compares Transformer vs RETAIN on the new
ERReadmissionMIMIC4 task.
"""
from pyhealth.datasets import (
    MIMIC4EHRDataset,
    split_by_patient,
    get_dataloader,
)
from pyhealth.tasks import ERReadmissionMIMIC4
from pyhealth.models import Transformer, RETAIN
from pyhealth.trainer import Trainer
import math


def _fmt(v):
    """Format a metric, showing 'n/a' for NaN."""
    return "n/a" if math.isnan(v) else f"{v:.4f}"


def _print_metrics(name, m):
    """Print ROC-AUC and PR-AUC for a model."""
    print(f"{name}  ROC-AUC: {_fmt(m['roc_auc'])}")
    print(f"{name}  PR-AUC : {_fmt(m['pr_auc'])}")


def main():
    """Run the Architecture-Comparison ablation."""
    print("Loading Dataset ...")
    # Point this to your MIMIC-IV root directory.
    dataset = MIMIC4EHRDataset(
        root="/path/to/mimic-iv-2.2",
        tables=["diagnoses_icd"],
        dev=True,
    )

    print("\nApplying ER-Specific Readmission Task ...")
    ds_er = dataset.set_task(ERReadmissionMIMIC4())
    print(f"ER samples: {len(ds_er)}")

    # --- Initialise both architectures -------------
    print("\n[Ablation] Architecture 1: RETAIN")
    model_ret = RETAIN(dataset=ds_er)
    print(" -> RETAIN OK")

    print("\n[Ablation] Architecture 2: Transformer")
    model_tfm = Transformer(dataset=ds_er)
    print(" -> Transformer OK")

    # --- Split + Dataloaders -----------------------
    print("\n--- Splitting data ---")
    tr, va, te = split_by_patient(
        ds_er, [0.8, 0.1, 0.1]
    )

    if len(va) == 0:
        print(
            "Val set is empty (tiny synthetic data).\n"
            "Pipeline verified — skipping Trainer."
        )
        return

    dl = get_dataloader
    tr_l = dl(tr, batch_size=64, shuffle=True)
    va_l = dl(va, batch_size=64, shuffle=False)
    te_l = dl(te, batch_size=64, shuffle=False)

    # --- Train RETAIN ------------------------------
    print("\n--- Training: RETAIN ---")
    t_ret = Trainer(model=model_ret)
    t_ret.train(
        train_dataloader=tr_l,
        val_dataloader=va_l,
        epochs=10,
        monitor="pr_auc",
    )
    m_ret = t_ret.evaluate(te_l)
    _print_metrics("RETAIN", m_ret)

    # --- Train Transformer -------------------------
    print("\n--- Training: Transformer ---")
    t_tfm = Trainer(model=model_tfm)
    t_tfm.train(
        train_dataloader=tr_l,
        val_dataloader=va_l,
        epochs=10,
        monitor="pr_auc",
    )
    m_tfm = t_tfm.evaluate(te_l)
    _print_metrics("Transformer", m_tfm)

    # --- Compare -----------------------------------
    r = m_ret["pr_auc"]
    t = m_tfm["pr_auc"]
    if math.isnan(r) or math.isnan(t):
        print(
            "\nAblation note: PR-AUC undefined "
            "on this tiny split (expected)."
        )
    else:
        d = t - r
        print(
            f"\nAblation result: Transformer "
            f"PR-AUC is {d * 100:.2f}% "
            f"{'higher' if d > 0 else 'lower'}"
            f" than RETAIN."
        )


if __name__ == "__main__":
    main()
