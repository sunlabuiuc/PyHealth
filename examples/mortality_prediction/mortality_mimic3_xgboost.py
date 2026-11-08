"""In-hospital mortality on synthetic MIMIC-III: XGBoost vs. LogisticRegression and MLP.

XGBoostModel encodes each code field (conditions, procedures, drugs) as
per-sample code counts (bag_of_codes=True), fits once with model.fit, and is
then evaluated with the same Trainer, split and metrics as the neural
baselines. It also prints the global TreeSHAP ranking.

Needs the optional extra: pip install "pyhealth[xgboost]".
Downloads the public synthetic MIMIC-III demo.

Usage:
    python examples/mortality_prediction/mortality_mimic3_xgboost.py
"""

import tempfile

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import MLP, LogisticRegression, XGBoostModel
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer

METRICS = ["pr_auc", "roc_auc"]

if __name__ == "__main__":
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=False,
    )
    samples = base_dataset.set_task(MortalityPredictionMIMIC3())
    train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=0)
    # Unshuffled loaders: tree fits use every row, in a fixed order.
    train_loader = get_dataloader(train, batch_size=256, shuffle=False)
    val_loader = get_dataloader(val, batch_size=256, shuffle=False)
    test_loader = get_dataloader(test, batch_size=256, shuffle=False)

    results = {}

    # Gradient-boosted trees: fit once, then evaluate with the Trainer.
    xgb_model = XGBoostModel(
        samples,
        bag_of_codes=True,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=0,
        scale_pos_weight="balanced",  # inflates probabilities; recalibrate if needed
        early_stopping_rounds=50,
        eval_metric="aucpr",
    )
    xgb_model.fit(train_loader, val_loader)
    results["XGBoost"] = Trainer(model=xgb_model, metrics=METRICS).evaluate(test_loader)

    # Neural baselines on the same split.
    for name, model in [
        ("LogisticRegression", LogisticRegression(dataset=samples)),
        ("MLP", MLP(dataset=samples)),
    ]:
        trainer = Trainer(model=model, metrics=METRICS)
        trainer.train(
            train_dataloader=get_dataloader(train, batch_size=64, shuffle=True),
            val_dataloader=val_loader,
            epochs=20,
            monitor="pr_auc",
        )
        results[name] = trainer.evaluate(test_loader)

    print(f"\n{'model':<20}{'PR-AUC':>8}{'AUROC':>8}")
    for name, scores in results.items():
        print(f"{name:<20}{scores['pr_auc']:>8.3f}{scores['roc_auc']:>8.3f}")

    print("\nTop 10 codes by mean |TreeSHAP| on the test set:")
    for feature, value in list(xgb_model.mean_abs_shap(test_loader).items())[:10]:
        print(f"  {feature:<40}{value:.4f}")
