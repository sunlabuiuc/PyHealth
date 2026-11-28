"""Synthetic patient -> decompensation task -> PyHealth DSA pipeline.

This example uses no real MIMIC-III records. It exercises the actual
DecompensationMIMIC3.__call__ patient processing rather than hand-crafting
hazard labels. It is intended for course reproducibility/CI demonstrations.

Run from the PyHealth repository environment with the custom DSA implementation:
    python examples/mortality_prediction/mimic3_decompensation_dsa_full_pipeline.py

This is a task + model integration example, NOT a new BaseDataset contribution.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DSA
from pyhealth.tasks import DecompensationMIMIC3
from pyhealth.trainer import Trainer

SEED = 42
HORIZON = 24
T0 = datetime(2020, 1, 1)


class FakePatient:
    """Minimal synthetic stand-in for a PyHealth Patient event interface."""

    def __init__(self, patient_id: str, events: dict[str, list[Any]]):
        self.patient_id = patient_id
        self.events = events

    def get_events(self, event_type: str) -> list[Any]:
        return self.events.get(event_type, [])


def make_patient(index: int, *, event: bool) -> FakePatient:
    """Create one synthetic adult ICU patient with charted observations."""
    patient_id = f"synthetic-{index:03d}"
    hadm_id = str(1000 + index)
    stay_id = str(2000 + index)
    start = T0 + timedelta(days=index * 3)
    stay_hours = 30.0
    death_hours = 29.5 if event else None

    stay = SimpleNamespace(
        hadm_id=hadm_id,
        icustay_id=stay_id,
        timestamp=start,
        outtime=start + timedelta(hours=stay_hours),
        first_wardid="1",
        last_wardid="1",
        first_careunit="MICU",
        last_careunit="MICU",
    )
    admission = SimpleNamespace(
        hadm_id=hadm_id,
        timestamp=start - timedelta(hours=2),
        dischtime=start + timedelta(hours=40),
        deathtime=(
            start + timedelta(hours=death_hours)
            if death_hours is not None
            else None
        ),
    )
    demographic = SimpleNamespace(
        dob=datetime(1980, 1, 1),
        dod=None,
    )

    # Observations are deliberately simple and wholly synthetic.
    # Distinct hours provide a time series while preserving the MIMIC item-ID
    # mapping used by the actual task implementation.
    charts = [
        SimpleNamespace(
            icustay_id=stay_id,
            hadm_id=hadm_id,
            timestamp=start + timedelta(hours=hour, minutes=5),
            itemid=220045,
            value=str(78 + (index % 4) + hour // 8),
            valueuom="bpm",
            label="Heart Rate",
        )
        for hour in (0, 4, 8, 12, 16, 20, 24)
    ]
    return FakePatient(
        patient_id,
        {
            "icustays": [stay],
            "admissions": [admission],
            "patients": [demographic],
            "chartevents": charts,
            "labevents": [],
        },
    )


def make_sample_dataset(
    samples: list[dict[str, Any]],
    task: DecompensationMIMIC3,
    name: str,
):
    """Apply PyHealth's feature/label processors to task-generated samples."""
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=name,
    )


def make_partition(
    patients: list[FakePatient], task: DecompensationMIMIC3
) -> list[dict[str, Any]]:
    """Generate all hourly records by executing the actual task on patients."""
    samples = [sample for p in patients for sample in task(p)]
    if not samples:
        raise RuntimeError("Task produced no hourly samples")
    labels = {int(sample["y_true"]) for sample in samples}
    if labels != {0, 1}:
        raise RuntimeError(f"Expected both classes per split; found {labels}")
    return samples


def validation_auprc(model: DSA, loader) -> float:
    """Calculate timestep-level AuPRC on held-out synthetic patients."""
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(**batch)
            labels.append(output["y_true"].detach().cpu().numpy().reshape(-1))
            scores.append(output["y_prob"].detach().cpu().numpy().reshape(-1))
    return float(
        average_precision_score(np.concatenate(labels), np.concatenate(scores))
    )


def run_variant(name: str, *, use_bias: bool, use_survtls: bool) -> float:
    """Train/evaluate a DSA variant on task-generated synthetic patient data."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    task = DecompensationMIMIC3(
        horizon_hours=HORIZON,
        min_observation_hours=5,
        survtls_lengthscale=8.0 if use_survtls else None,
    )

    # Split by PATIENT, not hourly sample: no patient's timepoints leak
    # between training, validation, and evaluation.
    patients = [make_patient(i, event=(i % 2 == 0)) for i in range(12)]
    train_patients, val_patients, test_patients = (
        patients[:8], patients[8:10], patients[10:12]
    )
    train_samples = make_partition(train_patients, task)
    val_samples = make_partition(val_patients, task)
    test_samples = make_partition(test_patients, task)
    print(
        f"{name}: patients 8/2/2, hourly samples "
        f"{len(train_samples)}/{len(val_samples)}/{len(test_samples)}"
    )

    train_ds = make_sample_dataset(train_samples, task, f"{name}_train")
    val_ds = make_sample_dataset(val_samples, task, f"{name}_val")
    test_ds = make_sample_dataset(test_samples, task, f"{name}_test")

    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=16, shuffle=False)

    prevalence = None
    if use_bias:
        # Training-only estimates, not validation/test labels.
        hazards = np.asarray(
            [sample["hazard_label"] for sample in train_samples], dtype=float
        )
        masks = np.asarray(
            [sample["hazard_mask"] for sample in train_samples], dtype=float
        )
        prevalence = np.divide(
            (hazards * masks).sum(axis=0),
            masks.sum(axis=0),
            out=np.zeros(HORIZON, dtype=float),
            where=masks.sum(axis=0) > 0,
        ).astype(np.float32)

    model = DSA(
        dataset=train_ds,
        hidden_dim=16,
        max_horizon=HORIZON,
        prediction_horizon=HORIZON - 1,
        num_layers=1,
        dropout=0.0,
        hazard_prevalence=prevalence,
    )
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
    )
    auprc = validation_auprc(model, test_loader)
    print(f"{name:28s} synthetic task-to-model test AuPRC = {auprc:.4f}")
    return auprc


if __name__ == "__main__":
    results = {
        "truncated": run_variant(
            "truncated", use_bias=False, use_survtls=False
        ),
        "truncated+bias": run_variant(
            "truncated+bias", use_bias=True, use_survtls=False
        ),
        "truncated+bias+survTLS": run_variant(
            "truncated+bias+survTLS", use_bias=True, use_survtls=True
        ),
    }
    print("\nSynthetic full-pipeline example completed:")
    for variant, score in results.items():
        print(f"  {variant}: AuPRC={score:.4f}")
    print("These synthetic scores are not MIMIC-III reproduction results.")
