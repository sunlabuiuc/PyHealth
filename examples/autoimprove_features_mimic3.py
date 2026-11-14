"""Check a hand-written feature and measure what it adds: no search, no LLM.

A feature is written as a spec (JSON), not code. The FeatureSpace validates it,
writes its plain-language definition from the spec itself, checks that it
cannot see the future, and adds it to the samples as an ``engineered`` field.
A paired cross-validation then answers "does this feature help?" with a
measured gain and its standard error.

Task: mortality at the next hospital visit (``MortalityPredictionMIMIC3``) on
PyHealth's public synthetic MIMIC-III (downloaded on first use; no
credentials). The prediction is made at discharge from the current visit, so
the feature may use events dated before that discharge. The synthetic labels
carry little signal, so expect small gains: the example shows the workflow. (A
third of the synthetic stays end before they begin, so the feature's count can
be 0 even for the current stay.)

Usage:
    python examples/autoimprove_features_mimic3.py
"""

from datetime import datetime

import polars as pl

from pyhealth.autoimprove import (
    FeatureSpace,
    PipelineConfig,
    SearchSpace,
    paired_gain,
    patient_folds,
    run_config_cv,
)
from pyhealth.datasets import MIMIC3Dataset, PatientSplit
from pyhealth.tasks import MortalityPredictionMIMIC3

SYNTHETIC_ROOT = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"

SPECS = [{
    "op": "count", "window": 365, "select": {"event_type": "admissions"},
    "rationale": "frequent recent hospital stays mark frailer patients",
    "direction": "+",
}]


def main():
    dataset = MIMIC3Dataset(
        root=SYNTHETIC_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    )
    train, val, _ = dataset.set_task(
        MortalityPredictionMIMIC3(), split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0)
    )

    # The prediction time is discharge from the sample's visit.
    admissions = (
        dataset.global_event_df.filter(pl.col("event_type") == "admissions")
        .select("admissions/hadm_id", "admissions/dischtime").collect()
    )
    discharge_of = {str(h): datetime.fromisoformat(str(d))
                    for h, d in admissions.iter_rows()}

    def discharge(sample) -> datetime:
        return discharge_of[str(sample["hadm_id"])]

    fs = FeatureSpace(
        dataset, time_of=discharge, time_word="discharge",
        nouns={"admissions": "hospital admissions"},
        # an admission's attributes (e.g. its discharge time) are known at discharge
        known_at={"admissions": "dischtime"},
    )
    for row in fs.check(SPECS, train):
        print(f"{row['name']}: {row['definition']}")
        print(f"  rationale: {row['rationale']}; expected direction {row['direction']}")
        print(f"  coverage {row['coverage']}; future-blind: "
              f"{row['future_blind'] or 'passed'}")

    specs = fs.validate_many(SPECS)
    train_fe = fs.augment(train, specs)
    val_fe = fs.augment(val, specs, stats=train_fe.stats)

    space = SearchSpace()
    config = PipelineConfig("MLP", {"embedding_dim": 64, "hidden_dim": 64},
                            {"epochs": 5})
    folds = patient_folds(train, 5)
    without = run_config_cv(config, space, train, val, "roc_auc", folds)
    with_fe = run_config_cv(config, space, train_fe, val_fe, "roc_auc", folds)
    gain, se = paired_gain(with_fe["fold_scores"], without["fold_scores"], True)
    print(f"CV ROC-AUC without the feature {without['metrics']['roc_auc']:.3f}, "
          f"with it {with_fe['metrics']['roc_auc']:.3f}: gain {gain:+.3f} ± {se:.3f}")


if __name__ == "__main__":
    main()
