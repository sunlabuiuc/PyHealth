"""MAS on Mayo PBC: reproduction and ablation study.

Reproduces the Model for Allograft Survival (MAS) of Gao et al., "Predicting
Long-Term Allograft Survival in Liver Transplant Recipients", MLHC 2024
(https://proceedings.mlr.press/v252/gao24a.html), on the public Mayo Clinic
primary biliary cholangitis cohort (``pbcseq``), and runs three ablations.

Pipeline (all PyHealth): ``PBCSeqDataset`` -> ``LongitudinalLiverSurvival``
(counting-process intervals; labs from each interval's starting visit) ->
``MAS`` (linear time-varying Cox, full-batch Breslow partial likelihood) ->
the paper's time-dependent concordance index (TDCI) on held-out patients.

Setup:
    - Five patient-level 70/15/15 splits (seeds 0-4). Feature scaling is fitted
      on the training patients only.
    - MAS is trained for 600 full-batch epochs at learning rate 0.05, with no
      penalty, as in the paper.
    - Metric: mean TDCI over t in {0.5, 1, 3, 5} and delta_t in {1, 3, 5, 7}
      years on the test patients, with a patient-level bootstrap 95% CI.
      Every configuration is also compared with the baseline on the same split
      by a paired bootstrap. Enrollment C (day-0 risk vs total follow-up) is
      reported too.

Configurations:
    - Baseline: MAS-4 on raw labs (bilirubin, albumin, AST, prothrombin time
      as the INR proxy; creatinine and ALT are not in PBC).
    - Lab scale: log-transformed labs.
    - E2, leave one biomarker out: MAS-4 minus each lab.
    - E1, missingness: follow-up labs masked at random at SRTR-like rates
      (AST 0.989, prothrombin time 0.874, albumin 0.704; bilirubin unmasked,
      its rate unknown), then imputed by forward-fill + mean (the paper's
      method), mean only, or mean + missing indicators.
    - E3, label definition: PyHealth's LogisticRegression on a fixed-horizon
      binary label (event within 1, 3, or 5 years; visits censored earlier
      are dropped) instead of a survival model.

Results on Mayo PBC (312 patients; 5 splits, 47 test patients each; mean over
splits, SD in parentheses; delta paired against the baseline on each split;
"CI excl 0" counts the splits whose paired 95% CI for delta excludes 0;
"Enroll. C" is Harrell's C on enrollment visits). Produced by this script
with its default arguments:

    =======================================  =============  ======  =========  =========
    Configuration                            TDCI (SD)       Delta  CI excl 0  Enroll. C
    =======================================  =============  ======  =========  =========
    Baseline: MAS-4, raw labs (paper)        0.789 (0.035)      --         --      0.775
    Log-transformed labs                     0.842 (0.038)  +0.053        3/5      0.842
    E2: without bili                         0.752 (0.040)  -0.036        3/5      0.734
    E2: without albumin                      0.773 (0.078)  -0.016        0/5      0.768
    E2: without ast                          0.792 (0.032)  +0.003        1/5      0.788
    E2: without protime                      0.783 (0.031)  -0.006        0/5      0.760
    E1: masked, forward-fill + mean (paper)  0.792 (0.045)  +0.004        0/5      0.800
    E1: masked, mean only                    0.782 (0.040)  -0.007        0/5      0.752
    E1: masked, mean + missing indicators    0.765 (0.056)  -0.024        0/5      0.770
    E3: binary label, 1-year horizon         0.782 (0.039)  -0.006        0/5      0.785
    E3: binary label, 3-year horizon         0.801 (0.037)  +0.012        2/5      0.809
    E3: binary label, 5-year horizon         0.808 (0.033)  +0.020        2/5      0.815
    =======================================  =============  ======  =========  =========

Findings:
    - Log-transformed labs give the largest gain.
    - Bilirubin carries most of the signal; AST adds nothing once the other
      three labs are included.
    - Forward-fill holds up under heavy SRTR-like missingness.
    - A fixed-horizon classifier ranks patients about as well as the survival
      model, because both are linear in the same four labs.

These are small effects next to the split-to-split spread, with wide CIs.
The paper reports a mean TDCI of 0.819 for MAS on SRTR transplant recipients,
a different cohort, so the numbers are not directly comparable.

Usage:
    # Real data: put pbcseq.csv (R survival::pbcseq) in a directory.
    python examples/pbcseq_liver_survival_mas.py --root /path/to/pbcseq-dir

    # No data needed: the same pipeline on a synthetic pbcseq cohort.
    python examples/pbcseq_liver_survival_mas.py --synthetic

    # Faster: one split, baseline and log labs only.
    python examples/pbcseq_liver_survival_mas.py --synthetic --seeds 0 --quick
"""

import argparse
import logging
import tempfile

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import (
    PatientSplit,
    PBCSeqDataset,
    get_dataloader,
    write_synthetic_pbcseq_csv,
)
from pyhealth.metrics import (
    concordance_index,
    mean_tdci_bootstrap_ci,
    mean_tdci_paired_bootstrap_diff,
    outcomes_from_labels,
    time_dependent_concordance_index,
)
from pyhealth.models import MAS, LogisticRegression
from pyhealth.tasks import LongitudinalLiverSurvival
from pyhealth.trainer import Trainer

MAS4 = ("bili", "albumin", "ast", "protime")
SRTR_MISSING = {"ast": 0.989, "protime": 0.874, "albumin": 0.704}
BASELINE = "Baseline: MAS-4, raw labs (paper)"


def configurations(quick: bool) -> list[tuple[str, str, dict]]:
    """(group, name, task options) for every MAS configuration."""
    rows = [
        ("baseline", BASELINE, {}),
        ("lab scale", "Log-transformed labs", {"log_transform": True}),
    ]
    if quick:
        return rows
    for lab in MAS4:
        others = [f for f in MAS4 if f != lab]
        rows.append(("E2", f"E2: without {lab}", {"feature_set": others}))
    masked = {"inject_missing": SRTR_MISSING}
    rows += [
        ("E1", "E1: masked, forward-fill + mean (paper)", masked),
        ("E1", "E1: masked, mean only", {**masked, "forward_fill": False}),
        (
            "E1",
            "E1: masked, mean + missing indicators",
            {**masked, "forward_fill": False, "missing_indicators": True},
        ),
    ]
    return rows


def full_batch(dataset) -> dict:
    """The whole split as one batch, which keeps the Cox risk sets complete."""
    return next(iter(get_dataloader(dataset, batch_size=len(dataset), shuffle=False)))


def train_mas(train, epochs: int, lr: float, seed: int) -> MAS:
    """Fits MAS on one training split with full-batch Adam."""
    torch.manual_seed(seed)
    model = MAS(train)
    loader = get_dataloader(train, batch_size=len(train), shuffle=False)
    Trainer(model=model, enable_logging=False).train(
        train_dataloader=loader, epochs=epochs, optimizer_params={"lr": lr}
    )
    return model


def mas_risk(model: MAS, batch: dict) -> np.ndarray:
    """MAS log-relative hazard per sample (higher means higher risk)."""
    model.eval()
    with torch.no_grad():
        features = batch["features"].to(model.device)
        return model.linear(features).flatten().cpu().numpy()


def train_logistic(train, epochs: int, seed: int) -> LogisticRegression:
    """Fits PyHealth's LogisticRegression on fixed-horizon binary labels."""
    torch.manual_seed(seed)
    model = LogisticRegression(dataset=train)
    loader = get_dataloader(train, batch_size=len(train), shuffle=False)
    Trainer(model=model, enable_logging=False).train(
        train_dataloader=loader, epochs=epochs, optimizer_params={"lr": 0.01}
    )
    return model


def logistic_risk(model: LogisticRegression, batch: dict) -> np.ndarray:
    """Predicted probability of the event within the horizon, per sample."""
    model.eval()
    with torch.no_grad():
        # forward() requires a label, but it does not affect y_prob.
        label = torch.zeros(len(batch["features"]), 1)
        output = model(features=batch["features"], label=label)
        return output["y_prob"].numpy().ravel()


def enrollment_concordance(batch: dict, risk: np.ndarray) -> float:
    """Harrell's C of each patient's first-interval risk vs total follow-up."""
    intervals = pd.DataFrame(
        batch["survival_interval"].numpy(), columns=["start", "stop", "event"]
    )
    intervals["patient_id"] = batch["patient_id"]
    intervals["risk"] = risk
    intervals = intervals.sort_values("start", kind="stable")
    first = intervals.drop_duplicates("patient_id").set_index("patient_id")
    last = intervals.drop_duplicates("patient_id", keep="last").set_index("patient_id")
    return concordance_index(
        duration=last["stop"], event=last["event"], risk=first.loc[last.index, "risk"]
    )


def score(batch: dict, risk, base_risk, n_boot: int, seed: int) -> dict:
    """Test mean TDCI with its CI, paired delta vs baseline, and enrollment C."""
    landmark = np.asarray(batch["landmark_time"], dtype=float)
    patient = np.asarray(batch["patient_id"])
    event_time, event = outcomes_from_labels(
        batch["survival_interval"].numpy(), landmark, patient
    )
    args = (event_time, event, risk, landmark, patient)
    ci = mean_tdci_bootstrap_ci(*args, n_boot=n_boot, seed=seed)
    diff = mean_tdci_paired_bootstrap_diff(
        event_time, event, risk, base_risk, landmark, patient, n_boot=n_boot, seed=seed
    )
    return {
        "mean_tdci": time_dependent_concordance_index(*args)["mean_tdci"],
        "ci_lower": ci["lower"],
        "ci_upper": ci["upper"],
        "delta": diff["estimate"],
        "delta_lower": diff["lower"],
        "delta_upper": diff["upper"],
        "enrollment_c": enrollment_concordance(batch, risk),
    }


def run_split(dataset, seed: int, args) -> list[dict]:
    """Every configuration on one patient split."""
    split = PatientSplit(ratios=(0.7, 0.15, 0.15), seed=seed)
    baseline_task = LongitudinalLiverSurvival()
    rows, base_risk, base_ids = [], None, None
    for group, name, options in configurations(args.quick):
        task = LongitudinalLiverSurvival(inject_seed=seed, **options)
        train, _, test = dataset.set_task(task, split=split)
        model = train_mas(train, args.epochs, args.lr, seed)
        batch = full_batch(test)
        risk = mas_risk(model, batch)
        if base_risk is None:
            base_risk, base_ids = risk, list(batch["visit_id"])
        # Paired comparisons need the same test samples in the same order.
        assert list(batch["visit_id"]) == base_ids, name
        coefficients = model.linear.weight.detach().flatten().tolist()
        rows.append(
            {"seed": seed, "group": group, "configuration": name, "model": "MAS"}
            | score(batch, risk, base_risk, args.n_boot, seed)
            | {"coefficients": [round(c, 4) for c in coefficients]}
        )
    for horizon in [] if args.quick else (1.0, 3.0, 5.0):
        binary = LongitudinalLiverSurvival(mode="binary_horizon", horizon=horizon)
        train, _, _ = dataset.set_task(binary, split=split)
        # Score the survival test samples with the classifier's own scaling.
        _, _, test = dataset.set_task(
            baseline_task, split=split, input_processors=train.input_processors
        )
        batch = full_batch(test)
        assert list(batch["visit_id"]) == base_ids, horizon
        risk = logistic_risk(train_logistic(train, args.lr_epochs, seed), batch)
        rows.append(
            {
                "seed": seed,
                "group": "E3",
                "configuration": f"E3: binary label, {horizon:g}-year horizon",
                "model": "LogisticRegression",
            }
            | score(batch, risk, base_risk, args.n_boot, seed)
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean over splits, and how many splits' delta CIs exclude 0."""
    excludes = (results["delta_lower"] > 0) | (results["delta_upper"] < 0)
    results = results.assign(delta_ci_excludes_0=excludes.astype(int))
    return results.groupby("configuration", sort=False).agg(
        model=("model", "first"),
        mean_tdci=("mean_tdci", "mean"),
        sd=("mean_tdci", "std"),
        delta=("delta", "mean"),
        delta_ci_excludes_0=("delta_ci_excludes_0", "sum"),
        enrollment_c=("enrollment_c", "mean"),
        splits=("seed", "nunique"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="Directory containing pbcseq.csv.")
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic pbcseq cohort instead of the real data.",
    )
    parser.add_argument("--synthetic-patients", type=int, default=300)
    parser.add_argument("--cache-dir", help="Directory for the dataset cache.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=600, help="MAS epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="MAS learning rate.")
    parser.add_argument("--lr-epochs", type=int, default=500, help="E3 epochs.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--quick", action="store_true", help="Baseline and log labs only."
    )
    parser.add_argument("--out", help="Optional CSV path for per-split results.")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    root = args.root
    if args.synthetic:
        root = tempfile.mkdtemp()
        write_synthetic_pbcseq_csv(root, n_patients=args.synthetic_patients, seed=0)
    dataset = PBCSeqDataset(root=root, cache_dir=args.cache_dir or tempfile.mkdtemp())

    rows = []
    for seed in args.seeds:
        rows += run_split(dataset, seed, args)
        print(f"split {seed} done", flush=True)
    results = pd.DataFrame(rows)
    if args.out:
        results.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\nTest-set mean TDCI over splits (delta paired against the baseline):")
    print(summarize(results).round(3).to_string())


if __name__ == "__main__":
    main()
