"""
Name: Ranjithkumar Rajendran
NetID: rr54
Paper: KEEP (CHIL 2025) — Elhussein et al.

Ablation 1 — Task Comparison.
Compares standard inpatient readmission
(ReadmissionPredictionMIMIC4) vs ER-specific
readmission (ERReadmissionMIMIC4) using RETAIN.
"""
from pyhealth.datasets import (
    MIMIC4EHRDataset,
    split_by_patient,
    get_dataloader,
)
from pyhealth.tasks import (
    ERReadmissionMIMIC4,
    ReadmissionPredictionMIMIC4,
)
from pyhealth.models import RETAIN
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
    """Run the Task-Comparison ablation."""
    print("Loading Dataset ...")
    # Point this to your MIMIC-IV root directory.
    # e.g. "/content/drive/MyDrive/mimic-iv/2.2"
    dataset = MIMIC4EHRDataset(
        root="/path/to/mimic-iv-2.2",
        tables=[
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
        dev=True,
    )

    # --- Task 1: Standard Inpatient Readmission ----
    print("\n[Ablation] Task 1: Standard Readmission")
    ds_std = dataset.set_task(
        ReadmissionPredictionMIMIC4()
    )

    # --- Task 2: ER-Specific Readmission -----------
    print("\n[Ablation] Task 2: ER Readmission")
    ds_er = dataset.set_task(ERReadmissionMIMIC4())

    print(f"\nStandard samples : {len(ds_std)}")
    print(f"ER-Specific samples: {len(ds_er)}")

    # --- Initialise models -------------------------
    print("\nInitializing RETAIN on both cohorts ...")
    model_std = RETAIN(dataset=ds_std)
    print(" -> Standard task: OK")
    model_er = RETAIN(dataset=ds_er)
    print(" -> ER task      : OK")

    # --- Split + Dataloaders -----------------------
    print("\n--- Splitting data ---")
    tr_s, va_s, te_s = split_by_patient(
        ds_std, [0.8, 0.1, 0.1]
    )
    tr_e, va_e, te_e = split_by_patient(
        ds_er, [0.8, 0.1, 0.1]
    )

    if len(va_s) == 0 or len(va_e) == 0:
        print(
            "Val set is empty (tiny synthetic data).\n"
            "Pipeline verified — skipping Trainer."
        )
        return

    dl = get_dataloader  # alias for brevity
    tr_l_s = dl(tr_s, batch_size=64, shuffle=True)
    va_l_s = dl(va_s, batch_size=64, shuffle=False)
    te_l_s = dl(te_s, batch_size=64, shuffle=False)

    tr_l_e = dl(tr_e, batch_size=64, shuffle=True)
    va_l_e = dl(va_e, batch_size=64, shuffle=False)
    te_l_e = dl(te_e, batch_size=64, shuffle=False)

    # --- Train Standard ----------------------------
    print("\n--- Training: Standard Readmission ---")
    t_std = Trainer(model=model_std)
    t_std.train(
        train_dataloader=tr_l_s,
        val_dataloader=va_l_s,
        epochs=10,
        monitor="pr_auc",
    )
    m_std = t_std.evaluate(te_l_s)
    _print_metrics("Standard", m_std)

    # --- Train ER ----------------------------------
    print("\n--- Training: ER Readmission ---")
    t_er = Trainer(model=model_er)
    t_er.train(
        train_dataloader=tr_l_e,
        val_dataloader=va_l_e,
        epochs=10,
        monitor="pr_auc",
    )
    m_er = t_er.evaluate(te_l_e)
    _print_metrics("ER", m_er)

    # --- Compare -----------------------------------
    s = m_std["pr_auc"]
    e = m_er["pr_auc"]
    if math.isnan(s) or math.isnan(e):
        print("\nAblation note: PR-AUC undefined "
              "on this tiny split (expected).")
    else:
        d = s - e
        print(
            f"\nAblation result: ER cohort "
            f"PR-AUC is {d * 100:.2f}% "
            f"{'lower' if d > 0 else 'higher'} "
            f"than standard."
        )


if __name__ == "__main__":
    main()
