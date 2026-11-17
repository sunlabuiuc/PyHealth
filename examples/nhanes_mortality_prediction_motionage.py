"""MotionAge end to end: wearable activity -> mortality risk -> biological age.

Paper: Song, Yin, et al., "MotionAge: A Deep Learning Framework for Biological
Age Prediction from Wearable Activity." https://openreview.net/pdf?id=xGw1o9acWa
Authors' code: https://github.com/jackie-jiaqi-yin/MotionAge (MIT License).

Steps:

1. Data: synthetic NHANES 2003-2004 files (default; no download needed), or
   the real NHANES files under ``--root``.
2. ``NHANESDataset`` -> ``NHANESMortalityPrediction`` (5-year mortality,
   3.5-day windows), split by participant; preprocessing is fitted on train
   only. By default one 60/20/20 split, like the authors'
   ``test_size=0.2, val_size=0.25``. With ``--n-folds k > 1``, participant-level
   k-fold cross-validation as in the paper (Sec. 3.2) and the authors' code:
   test folds stratified on the 5-year label, and a stratified 25% of the
   other folds as validation. The tables then show the mean and standard
   deviation across folds.
3. ``MotionAge`` trained with PyHealth's ``Trainer``. The best epoch is chosen
   by window-level validation AUPRC; the authors select on participant-level
   AUPRC, which ``Trainer`` cannot monitor.
4. Per-participant risk: the mean of the participant's window probabilities.
5. ``MotionAgeMapping`` fitted on training participants (their predictions are
   in-sample, as in the authors' code) turns risk into MotionAge; age
   acceleration is MotionAge minus age.
6. Evaluation on test participants (optionally only those aged ``--min-age``
   or older):

   * C-index C(M, X) of the raw predictor X = age, MotionAge or acceleration
     against the 5-year label M. The paper (Sec. 3.2, Table 1) does not
     mention time-to-event; we read M as the binary label, so C(M, X) equals
     the AUROC of raw X.
   * AUROC of logistic regressions fitted on training participants, for
     [age, sex], [MotionAge, sex] and [acceleration, age, sex], as in the
     authors' ``evaluation/secondary.py``.
   * Extra, not in the paper: Harrell's C on the full linked follow-up
     (``permth_int``, months from interview, the variable the task's 5-year
     label uses; event = ``mortstat``). Skip with ``--no-harrell``.

The defaults (1000 synthetic participants, 7 days, 10 epochs, batch size 32,
one split) run in about 2 minutes on a laptop CPU; each extra fold adds about
as much again. The authors train with batch size 128
for up to 80 epochs; on 1000 synthetic participants that gives too few steps
to learn in a few minutes, so use ``--batch-size 128 --epochs 80`` on real
data. Synthetic data only checks that the pipeline works; its numbers say
nothing about the paper's results.

Usage:
    python examples/nhanes_mortality_prediction_motionage.py
    python examples/nhanes_mortality_prediction_motionage.py \\
        --root ../data/nhanes --min-age 40 --n-folds 5 --out results.json
"""

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from pyhealth.datasets import PatientSplit, Split, get_dataloader
from pyhealth.datasets.nhanes import NHANESDataset
from pyhealth.datasets.nhanes_synthetic import generate_synthetic_nhanes
from pyhealth.metrics.motionage import (
    age_acceleration,
    aggregate_window_predictions,
    evaluate_feature_sets,
    label_concordance,
    participant_fields,
)
from pyhealth.metrics.survival import concordance_index
from pyhealth.models.motionage import MotionAge, MotionAgeMapping
from pyhealth.tasks.nhanes_mortality_prediction import NHANESMortalityPrediction
from pyhealth.trainer import Trainer

SPLIT_NAMES = ("train", "val", "test")
# Raw predictors for the C-index, by display name.
PREDICTORS = {
    "age": "age",
    "motionage": "motion_age",
    "acceleration": "age_acceleration",
}
# Logistic-regression feature sets for the AUROC table (paper / authors).
FEATURE_SETS = {
    "age": ["age", "sex"],
    "motionage": ["motion_age", "sex"],
    "acceleration": ["age_acceleration", "age", "sex"],
}


def follow_up(dataset: NHANESDataset) -> pd.DataFrame:
    """Linked-mortality status and months of follow-up from the interview."""
    events = dataset.global_event_df.select(
        ["patient_id", "nhanes/mortstat", "nhanes/permth_int"]
    ).collect()
    table = events.to_pandas()
    table.columns = ["patient_id", "mortstat", "permth_int"]
    table["patient_id"] = table["patient_id"].astype(str)
    return table


def five_year_labels(
    task: NHANESMortalityPrediction, follow: pd.DataFrame
) -> dict[str, int]:
    """Each participant's label by the task's own rule (``permth_int``).

    Participants whose label is unknown are left out; the task gives them no
    samples, so they never reach a split.
    """
    labels = {}
    for row in follow.itertuples(index=False):
        event = SimpleNamespace(mortstat=row.mortstat, permth_int=row.permth_int)
        label = task._label(event)  # The task's rule, so the folds agree with it.
        if label is not None:
            labels[row.patient_id] = label
    return labels


@dataclass(frozen=True)
class PatientKFold(Split):
    """Fold ``fold`` of a stratified participant-level k-fold split.

    Test participants are fold ``fold`` of ``StratifiedKFold(n_folds,
    shuffle=True, random_state=seed)`` on the label; a stratified
    ``val_ratio`` of the remaining participants (``random_state=seed + fold``)
    is the validation part, the rest training. This follows the authors'
    ``data/partitioning.py``. Parts are ``(train, val, test)``.

    Args:
        n_folds: Number of folds, at least 2.
        fold: Which fold is the test part, ``0 .. n_folds - 1``.
        labels: Maps each participant ID to its binary label.
        val_ratio: Fraction of the non-test participants used for validation.
        seed: Seed for both shuffles.
    """

    n_folds: int
    fold: int
    labels: Mapping[str, int]
    val_ratio: float = 0.25
    seed: int = 0

    def __post_init__(self):
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {self.n_folds}")
        if not 0 <= self.fold < self.n_folds:
            raise ValueError(f"fold must be in [0, {self.n_folds}), got {self.fold}")
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")

    def to_dict(self) -> dict:
        """Describes the split; the label hash keeps cache keys distinct."""
        digest = hashlib.sha1(
            json.dumps(sorted(self.labels.items())).encode()
        ).hexdigest()
        return {
            "kind": "patient_kfold",
            "n_folds": self.n_folds,
            "fold": self.fold,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
            "labels_sha1": digest,
        }

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns the sample indices of the train, val and test parts."""
        patients = sorted(patient_to_index, key=str)
        missing = [p for p in patients if str(p) not in self.labels]
        if missing:
            raise KeyError(f"No label for participants {missing[:5]}")
        y = np.array([self.labels[str(p)] for p in patients])
        folds = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        development, test = list(folds.split(np.zeros(len(y)), y))[self.fold]
        train, val = train_test_split(
            development,
            test_size=self.val_ratio,
            stratify=y[development],
            random_state=self.seed + self.fold,
        )

        def indices(part: np.ndarray) -> np.ndarray:
            out = np.fromiter(
                chain.from_iterable(patient_to_index[patients[i]] for i in part),
                dtype=np.int64,
            )
            out.sort()
            return out

        return [indices(train), indices(val), indices(test)]


def summarize_folds(
    folds: list[dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Stacks each fold's tables and adds the mean and SD across folds.

    Returns:
        ``<name>_folds`` (every fold's rows, with a ``fold`` column) and
        ``<name>`` (mean and SD per row key) for each result table, plus the
        stacked ``participants``.
    """
    keys = {
        "cohort": [],
        "mapping": ["sex"],
        "concordance": ["predictor"],
        "auroc": ["feature_set"],
        "harrell": ["predictor"],
    }
    out = {}
    for name, by in keys.items():
        stacked = pd.concat(
            [tables[name].assign(fold=i) for i, tables in enumerate(folds)],
            ignore_index=True,
        )
        out[f"{name}_folds"] = stacked
        if stacked.empty:
            out[name] = stacked
            continue
        numeric = [
            c
            for c in stacked.select_dtypes("number").columns
            if c not in (*by, "fold", "min_age")
        ]
        if by:
            summary = stacked.groupby(by, sort=False)[numeric].agg(["mean", "std"])
            summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
            summary = summary.reset_index()
        else:  # cohort: one row per fold, no key.
            stats = stacked[numeric].agg(["mean", "std"])
            summary = pd.DataFrame(
                [{f"{c}_{s}": stats.loc[s, c] for c in numeric for s in stats.index}]
            )
            summary.insert(0, "min_age", stacked["min_age"].iloc[0])
        out[name] = summary
    out["participants"] = pd.concat(
        [t["participants"].assign(fold=i) for i, t in enumerate(folds)],
        ignore_index=True,
    )
    return out


def predict_participants(trainer: Trainer, samples, batch_size: int) -> pd.DataFrame:
    """One row per participant: mean window risk, label, age and sex."""
    loader = get_dataloader(samples, batch_size=batch_size, shuffle=False)
    y_true, y_prob, _, patient_ids = trainer.inference(loader, return_patient_ids=True)
    risk = aggregate_window_predictions(patient_ids, y_prob, y_true)
    return risk.merge(participant_fields(samples), on="patient_id", validate="1:1")


def run_split(
    dataset: NHANESDataset,
    task: NHANESMortalityPrediction,
    split: Split,
    args: argparse.Namespace,
    follow: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Trains, calibrates and evaluates MotionAge on one train/val/test split.

    Returns:
        Tables ``cohort``, ``mapping``, ``concordance``, ``auroc``, ``harrell``
        (empty without ``follow``) and ``participants``.
    """
    parts = dataset.set_task(task, split=split)
    torch.manual_seed(args.seed)
    model = MotionAge(parts[0], category_sizes=task.category_sizes, fusion=args.fusion)
    trainer = Trainer(model=model, metrics=["pr_auc", "roc_auc"], enable_logging=False)
    trainer.train(
        get_dataloader(parts[0], batch_size=args.batch_size, shuffle=True),
        get_dataloader(parts[1], batch_size=args.batch_size, shuffle=False),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="pr_auc",
    )

    participants = pd.concat(
        [
            predict_participants(trainer, part, args.batch_size).assign(split=name)
            for name, part in zip(SPLIT_NAMES, parts)
        ],
        ignore_index=True,
    )
    train_rows = participants["split"] == "train"
    mapping = MotionAgeMapping(min_participants=args.min_participants).fit(
        participants.loc[train_rows, "risk"],
        participants.loc[train_rows, "age"],
        participants.loc[train_rows, "sex"],
    )
    participants["motion_age"] = mapping.transform(
        participants["risk"], participants["sex"]
    )
    participants["age_acceleration"] = age_acceleration(
        participants["motion_age"], participants["age"]
    )

    # The age filter applies to the evaluation tables only, not to training
    # or to fitting the mapping.
    in_cohort = (
        participants["age"] >= args.min_age
        if args.min_age is not None
        else pd.Series(True, index=participants.index)
    )
    train = participants[train_rows & in_cohort]
    test = participants[(participants["split"] == "test") & in_cohort]

    cohort = pd.DataFrame(
        [
            {
                "min_age": args.min_age,
                "n_train": len(train),
                "train_deaths": int(train["label"].sum()),
                "n_test": len(test),
                "test_deaths": int(test["label"].sum()),
            }
        ]
    )
    concordance = pd.DataFrame(
        [
            {"predictor": name, "c_index": label_concordance(test["label"], test[col])}
            for name, col in PREDICTORS.items()
        ]
    )
    auroc = evaluate_feature_sets(train, test, FEATURE_SETS, baseline="age")
    mapping_table = pd.DataFrame(
        [
            {"sex": sex, "alpha": alpha, "beta": beta}
            for sex, (alpha, beta) in sorted(mapping.params.items())
        ]
    )
    harrell = pd.DataFrame()
    if follow is not None:
        timed = test.merge(follow, on="patient_id", validate="1:1")
        rows = []
        for name, col in PREDICTORS.items():
            keep = timed[col].notna() & timed["permth_int"].notna()
            rows.append(
                {
                    "predictor": name,
                    "harrell_c": concordance_index(
                        timed.loc[keep, "permth_int"],
                        timed.loc[keep, "mortstat"],
                        timed.loc[keep, col],
                    ),
                    "deaths_any_time": int(timed.loc[keep, "mortstat"].sum()),
                }
            )
        harrell = pd.DataFrame(rows)
    return {
        "cohort": cohort,
        "mapping": mapping_table,
        "concordance": concordance,
        "auroc": auroc,
        "harrell": harrell,
        "participants": participants,
    }


def report(tables: dict[str, pd.DataFrame], n_folds: int = 1) -> None:
    """Prints the result tables (fold means and SDs when ``n_folds > 1``)."""
    # to_dict keeps each column's dtype; a row Series would upcast to float.
    cohort = tables["cohort"].to_dict("records")[0]
    filt = "none" if pd.isna(cohort["min_age"]) else f"age >= {cohort['min_age']:g}"
    print(f"\nEvaluation filter: {filt}")
    if n_folds == 1:
        print(
            f"Test participants: {cohort['n_test']} ({cohort['test_deaths']} "
            f"deaths within 5 years); training participants for the logistic "
            f"fits: {cohort['n_train']} ({cohort['train_deaths']} deaths)"
        )
    else:
        print(f"{n_folds}-fold cross-validation; mean and SD across folds.")
        columns = ["fold", "n_train", "train_deaths", "n_test", "test_deaths"]
        print("Participants per fold after the filter:")
        print(tables["cohort_folds"][columns].to_string(index=False))
    sections = [
        ("MotionAge mapping, logit(risk) = alpha + beta * age", "mapping"),
        ("C-index C(M, X), raw predictor vs 5-year label (test)", "concordance"),
        ("AUROC, logistic regression fitted on train (test)", "auroc"),
        ("Extra: Harrell's C on full follow-up, permth_int (test)", "harrell"),
    ]
    with pd.option_context("display.float_format", "{:.3f}".format):
        for title, key in sections:
            if not tables[key].empty:
                print(f"\n{title}\n{tables[key].to_string(index=False)}")


def save(tables: dict[str, pd.DataFrame], out: str) -> None:
    """Writes the tables to one ``.json`` file, or to CSVs in a directory."""
    path = Path(out)
    if path.suffix.lower() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: json.loads(table.to_json(orient="records"))
            for key, table in tables.items()
        }
        path.write_text(json.dumps(payload, indent=2))
    else:
        path.mkdir(parents=True, exist_ok=True)
        for key, table in tables.items():
            table.to_csv(path / f"{key}.csv", index=False)
    print(f"\nSaved results to {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--root",
        help="Folder with the NHANES files; default: generate synthetic data",
    )
    data.add_argument("--n-participants", type=int, default=1000)
    data.add_argument("--n-days", type=int, default=7)
    data.add_argument("--cache-dir", help="PyHealth cache; default: a temp dir")
    model = parser.add_argument_group("model and training")
    model.add_argument(
        "--covariate-set",
        default="clinical",
        choices=["clinical", "age_sex", "none"],
        help="clinical = MotionAge-FRC, age_sex = -FA, none = wearable only (-F)",
    )
    model.add_argument("--fusion", default="late", choices=["late", "residual"])
    model.add_argument("--epochs", type=int, default=10)
    model.add_argument("--batch-size", type=int, default=32)
    model.add_argument("--lr", type=float, default=1e-3)
    model.add_argument("--seed", type=int, default=0)
    evaluation = parser.add_argument_group("calibration and evaluation")
    evaluation.add_argument(
        "--min-participants",
        type=int,
        default=100,
        help="Minimum training participants per sex to fit the mapping",
    )
    evaluation.add_argument(
        "--min-age",
        type=float,
        default=None,
        help="Evaluate only participants at least this old (paper: 40)",
    )
    evaluation.add_argument(
        "--no-harrell", action="store_true", help="Skip the Harrell's C table"
    )
    evaluation.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="1 = one 60/20/20 split; k > 1 = stratified k-fold CV",
    )
    evaluation.add_argument("--out", help="Save tables: a .json file or a directory")
    args = parser.parse_args(argv)
    if args.n_folds < 1:
        parser.error("--n-folds must be at least 1")
    return args


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    args = parse_args(argv)
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = args.root
        if root is None:
            root = str(Path(tmp) / "synthetic")
            generate_synthetic_nhanes(
                root,
                n_participants=args.n_participants,
                n_days=args.n_days,
                seed=args.seed,
            )
            print(f"Generated {args.n_participants} synthetic participants")
        dataset = NHANESDataset(
            root=root, cache_dir=args.cache_dir or str(Path(tmp) / "cache")
        )
        task = NHANESMortalityPrediction(
            covariate_set=None if args.covariate_set == "none" else args.covariate_set
        )
        follow = None if args.no_harrell else follow_up(dataset)
        if args.n_folds == 1:
            tables = run_split(
                dataset,
                task,
                PatientSplit(ratios=(0.6, 0.2, 0.2), seed=args.seed),
                args,
                follow,
            )
        else:
            labels = five_year_labels(
                task, follow if follow is not None else follow_up(dataset)
            )
            folds = []
            for fold in range(args.n_folds):
                print(f"\n=== Fold {fold + 1} of {args.n_folds} ===")
                split = PatientKFold(args.n_folds, fold, labels, seed=args.seed)
                folds.append(run_split(dataset, task, split, args, follow))
            tables = summarize_folds(folds)
    report(tables, args.n_folds)
    if args.out:
        save(tables, args.out)
    minutes, seconds = divmod(time.time() - start, 60)
    print(f"\nTotal run time: {math.floor(minutes)} min {seconds:.0f} s")
    return tables


if __name__ == "__main__":
    main()
