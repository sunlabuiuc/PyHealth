"""AMA Prediction on MIMIC-III -- Ablation Study.

This script reproduces the Against-Medical-Advice (AMA) discharge
prediction task from:

    Boag, W.; Suresh, H.; Celi, L. A.; Szolovits, P.; Ghassemi, M.
    "Racial Disparities and Mistrust in End-of-Life Care."
    Machine Learning for Healthcare Conference, PMLR, 2018.

The paper predicts AMA discharge using L1-regularized logistic
regression on demographic features (age, gender, race, insurance,
LOS) and three mistrust-proxy scores.  Our PyHealth task reproduces
the same label and demographic feature set.  We use PyHealth's
LogisticRegression model (a single linear layer on feature
embeddings) as the primary model since it is the closest analog
to the paper's approach within PyHealth's pipeline.

The ablation study is structured as follows:

1. **Paper baseline** -- LogisticRegression on demographics only
   (age + LOS + gender + race + insurance), matching the paper's
   BASELINE+RACE configuration.

2. **Feature group comparison** -- demographics-only versus
   clinical-codes-only versus all features combined, showing
   whether ICD codes and prescriptions add predictive value
   beyond demographics.

3. **Model comparison** -- LogisticRegression versus RNN versus
   Transformer on the full feature set, showing whether more
   expressive architectures improve AMA prediction.

All experiments use the synthetic MIMIC-III demo data hosted by
PyHealth so the script is runnable without credentialed access.
Because the demo data contains very few patients and may lack
positive AMA labels, the reported metrics are illustrative only.

Usage:
    python examples/mimic3_ama_prediction_logistic_regression.py
"""

import tempfile

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import LogisticRegression, RNN, Transformer
from pyhealth.tasks import AMAPredictionMIMIC3
from pyhealth.trainer import Trainer

SYNTHETIC_ROOT = (
    "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
)
TABLES = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"]
EPOCHS = 5
BATCH_SIZE = 32
MONITOR = "pr_auc"

DEMO_KEYS = ["demographics", "age", "los"]
CODE_KEYS = ["conditions", "procedures", "drugs"]
ALL_KEYS = DEMO_KEYS + CODE_KEYS


def load_dataset():
    """Load the synthetic MIMIC-III dataset and apply the AMA task."""
    dataset = MIMIC3Dataset(
        root=SYNTHETIC_ROOT,
        tables=TABLES,
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    dataset.stats()
    task = AMAPredictionMIMIC3()
    sample_dataset = dataset.set_task(task)
    return sample_dataset


def run_experiment(sample_dataset, model_cls, model_kwargs, label):
    """Train and evaluate a single configuration.

    Args:
        sample_dataset: The ``SampleDataset`` returned by ``set_task``.
        model_cls: Model class (``LogisticRegression``, ``RNN``, or
            ``Transformer``).
        model_kwargs: Extra keyword arguments forwarded to the model.
        label: Human-readable experiment label for logging.

    Returns:
        Dict of evaluation metrics, or ``None`` if training failed.
    """
    train_ds, val_ds, test_ds = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dl = get_dataloader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = get_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = model_cls(dataset=sample_dataset, **model_kwargs)

    trainer = Trainer(model=model)
    try:
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            epochs=EPOCHS,
            monitor=MONITOR,
        )
        metrics = trainer.evaluate(test_dl)
    except Exception as exc:
        print(f"  [{label}] Training failed: {exc}")
        return None

    print(f"  [{label}] {metrics}")
    return metrics


def main():
    sample_dataset = load_dataset()

    # =================================================================
    # Ablation 1: Paper baseline (LogisticRegression on demographics)
    #
    # This is the closest reproduction of the paper's BASELINE+RACE
    # configuration: age, LOS, gender, race, and insurance fed into
    # a logistic regression model.
    # =================================================================
    print("\n" + "=" * 60)
    print("ABLATION 1: Paper baseline -- LogisticRegression on demographics")
    print("=" * 60)

    run_experiment(
        sample_dataset,
        LogisticRegression,
        {"feature_keys": DEMO_KEYS},
        "LogReg demographics (paper baseline)",
    )

    # =================================================================
    # Ablation 2: Feature group comparison (all using LogisticRegression)
    #
    # Compare demographics-only (paper's features) versus clinical
    # codes only (extension beyond paper) versus all combined.
    # =================================================================
    print("\n" + "=" * 60)
    print("ABLATION 2: Feature groups (LogisticRegression)")
    print("=" * 60)

    configs = [
        (DEMO_KEYS, "demographics only (age+LOS+gender+race+insurance)"),
        (CODE_KEYS, "clinical codes only (conditions+procedures+drugs)"),
        (ALL_KEYS, "all features combined"),
    ]
    for feature_keys, label in configs:
        run_experiment(
            sample_dataset,
            LogisticRegression,
            {"feature_keys": feature_keys},
            label,
        )

    # =================================================================
    # Ablation 3: Model comparison on all features
    #
    # Test whether more expressive architectures improve over the
    # logistic regression baseline when all features are available.
    # =================================================================
    print("\n" + "=" * 60)
    print("ABLATION 3: Model comparison (all features)")
    print("=" * 60)

    model_configs = [
        (LogisticRegression, {}, "LogisticRegression"),
        (RNN, {"hidden_size": 64}, "RNN hidden=64"),
        (Transformer, {"hidden_size": 64}, "Transformer hidden=64"),
    ]

    for model_cls, kwargs, label in model_configs:
        run_experiment(
            sample_dataset,
            model_cls,
            {"feature_keys": ALL_KEYS, **kwargs},
            label,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
