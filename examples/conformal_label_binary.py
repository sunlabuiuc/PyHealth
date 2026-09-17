"""Binary conformal prediction (LABEL) with RNN on synthetic MIMIC-III.

This script trains a binary readmission model, calibrates LABEL using threshold
and APS scores, and evaluates prediction sets on held-out patients.

Run from the repository root:
    python -m examples.conformal_label_binary

The public synthetic dataset is downloaded automatically. Training, validation,
calibration, and testing use separate patients. Results illustrate the workflow
on synthetic data; they are not estimates of clinical performance.
"""

import tempfile

import torch

from pyhealth.calib.predictionset import LABEL
from pyhealth.datasets import (
    MIMIC3Dataset,
    get_dataloader,
    split_by_patient_conformal,
)
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import RNN
from pyhealth.tasks import ReadmissionPredictionMIMIC3
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    torch.manual_seed(42)
    cache_dir = tempfile.TemporaryDirectory()

    # STEP 1: Load dataset
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=cache_dir.name,
        dev=True,
        num_workers=1,
    )
    base_dataset.stats()

    # STEP 2: Set task
    # Must include minors to get any readmission samples on the synthetic dataset
    task = ReadmissionPredictionMIMIC3(exclude_minors=False)
    sample_dataset = base_dataset.set_task(task)

    # STEP 3: Reserve calibration patients separately from model validation.
    train_dataset, val_dataset, cal_dataset, test_dataset = split_by_patient_conformal(
        sample_dataset, [0.5, 0.1, 0.2, 0.2], seed=42
    )
    print(
        f"Samples: train={len(train_dataset)}, validation={len(val_dataset)}, "
        f"calibration={len(cal_dataset)}, test={len(test_dataset)}"
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 4: Define model
    model = RNN(
        dataset=sample_dataset,
    )

    # STEP 5: Train
    # Tiny synthetic splits may contain one class, so monitor accuracy, not AUC.
    trainer = Trainer(model=model, metrics=["accuracy"], enable_logging=False)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="accuracy",
    )

    # STEP 6: Evaluate the base model on held-out test patients.
    print("Base model metrics:", trainer.evaluate(test_dataloader))

    # STEP 7: Calibrate binary prediction sets with either supported score.
    # A 70% target allows a finite quantile with only four calibration samples.
    alpha = 0.3
    for score_type in ("threshold", "aps"):
        predictor = LABEL(model, alpha=alpha, score_type=score_type, random_state=42)
        predictor.calibrate(cal_dataset=cal_dataset)

        # Collect prediction sets explicitly; Trainer.evaluate reports scalar
        # classification metrics but does not collect y_predset.
        y_true, y_prob, _, extra = Trainer(
            model=predictor, device=trainer.device, enable_logging=False
        ).inference(test_dataloader, additional_outputs=["y_predset"])
        metrics = binary_metrics_fn(
            y_true,
            y_prob,
            metrics=[
                "accuracy", "set_size", "rejection_rate", "miscoverage_overall_ps"
            ],
            y_predset=extra["y_predset"],
        )
        print(f"\nLABEL ({score_type}), target coverage: {1 - alpha:.0%}")
        print("Metrics:", metrics)
        print(f"Empirical coverage: {1 - metrics['miscoverage_overall_ps']:.3f}")
        print("Probability shape:", y_prob.shape)  # (N, 1), still binary
        print("Prediction-set shape:", extra["y_predset"].shape)  # (N, 2)
        print("First five sets [class 0, class 1]:")
        print(extra["y_predset"][:5])

    sample_dataset.close()
    cache_dir.cleanup()
