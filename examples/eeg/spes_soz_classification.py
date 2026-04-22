"""Train and evaluate SPES SOZ classifiers on RESPect-style CCEP data.

Reproduces a small Norris et al. (ML4H 2024) style comparison of SPESResNet and
SPESTransformer (divergent vs convergent SPES, with/without distance features).
Pipeline: RESPectCCEPDataset -> SeizureOnsetZoneLocalisation -> split ->
Trainer -> AUROC, AUPRC, and related binary metrics.

Data sources:

    Real data:
        Pass ``--dataset-root`` pointing at a RESPect / OpenNeuro ds004080 layout.
    Synthetic (default):
        If ``--dataset-root`` is omitted, writes minimal CSVs under a temp
        directory.

Command-line arguments:

    --dataset-root (str, optional):
        Path to a RESPect CCEP root (e.g. OpenNeuro ds004080). Default: omitted
        (synthetic mode).
    --synthetic-patients (int):
        Synthetic cohort size when not using ``--dataset-root``. Default: 10.
    --synthetic-electrodes (int):
        Recording contacts per synthetic patient. Default: 20.
    --synthetic-stim-pairs (int):
        Stimulation pairs sampled per synthetic patient. Default: 6.
    --timesteps (int):
        Mean/std response length for synthetic rows. Default: 509.
    --synthetic-annotated-ratio (float):
        Fraction of synthetic patients with SOZ maps. Default: 0.5.
    --synthetic-soz-positive-ratio (float):
        SOZ-positive electrode fraction among annotated patients. Default: 0.144.

Usage:

    From the repo root, synthetic demo (all defaults)::

        python examples/eeg/spes_soz_classification.py

    Real RESPect layout on disk::

        python examples/eeg/spes_soz_classification.py \\
            --dataset-root /path/to/ds004080

    Custom synthetic cohort::

        python examples/eeg/spes_soz_classification.py \\
            --synthetic-patients 8 \\
            --synthetic-electrodes 16 \\
            --synthetic-stim-pairs 5 \\
            --timesteps 400 \\
            --synthetic-annotated-ratio 0.6 \\
            --synthetic-soz-positive-ratio 0.2

Examples:
    >>> cfgs = get_spes_classification_configs()
    >>> {c["name"] for c in cfgs} == {
    ...     "cnn_resnet_divergent_no_features",
    ...     "cnn_resnet_divergent_with_features",
    ...     "cnn_resnet_convergent_no_features",
    ...     "cnn_resnet_convergent_with_features",
    ...     "cnn_transformer_convergent_no_features",
    ...     "cnn_transformer_convergent_with_features",
    ... }
    True
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from pyhealth.datasets import get_dataloader, split_by_patient, split_by_sample
from pyhealth.datasets.respectccep import RESPectCCEPDataset
from pyhealth.models import SPESResNet, SPESTransformer
from pyhealth.tasks.ccep_detect_soz import SeizureOnsetZoneLocalisation
from pyhealth.trainer import Trainer

TIMESTEPS = 509
SEED = 2026
BATCH_SIZE = 2
EPOCHS = 5
SYNTHETIC_N_PATIENTS = 10
SYNTHETIC_N_ELECTRODES = 20
SYNTHETIC_N_STIM_PAIRS = 6
SYNTHETIC_ANNOTATED_PATIENT_RATIO = 0.5
SYNTHETIC_SOZ_POSITIVE_RATIO = 0.144


def _to_json_1d(arr: np.ndarray) -> str:
    """Serialize a 1-D float32 array as compact JSON (RESPect CSV format).

    Args:
        arr: One-dimensional NumPy array.

    Returns:
        JSON string with no extra whitespace, suitable for response_ts columns.
    """
    return json.dumps(arr.astype(np.float32).tolist(), separators=(",", ":"))


def _pick_stim_pairs(
    electrodes: List[str],
    n_pairs: int,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    """Sample up to n_pairs unique stimulation pairs from electrodes.

    Args:
        electrodes: Ordered contact names.
        n_pairs: Desired number of pairs (capped by available combinations).
        rng: NumPy random generator for shuffling.

    Returns:
        List of (stim_a, stim_b) tuples with a before b in electrodes.
    """
    pairs: List[Tuple[str, str]] = []
    for i in range(len(electrodes)):
        for j in range(i + 1, len(electrodes)):
            pairs.append((electrodes[i], electrodes[j]))
    rng.shuffle(pairs)
    return pairs[: max(1, min(n_pairs, len(pairs)))]


def _build_synthetic_respect_table(
    n_patients: int,
    n_electrodes: int,
    n_stim_pairs: int,
    timesteps: int,
    seed: int,
    annotated_patient_ratio: float,
    soz_positive_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic RESPect-compatible participants and CCEP row tables.

    Called from _create_synthetic_dataset_root.

    Args:
        n_patients: Synthetic cohort size (>= 2).
        n_electrodes: Recording contacts per patient (>= 4).
        n_stim_pairs: Target pair count (capped by combinatorics).
        timesteps: Mean/std vector length (>= 200 for SPES models).
        seed: RNG seed.
        annotated_patient_ratio: Fraction of patients with non-trivial SOZ maps.
        soz_positive_ratio: Positive electrode rate among annotated patients.

    Returns:
        Tuple (participants_df, rows_df) for participants.tsv and
        respect_ccep_data-pyhealth.csv.

    Raises:
        ValueError: If arguments are outside supported ranges.
    """

    if n_patients < 2:
        raise ValueError("n_patients must be >= 2.")
    if n_electrodes < 4:
        raise ValueError("n_electrodes must be >= 4.")
    if timesteps < 200:
        raise ValueError("timesteps must be >= 200 for SPES models.")
    if not 0.0 <= annotated_patient_ratio <= 1.0:
        raise ValueError("annotated_patient_ratio must be between 0.0 and 1.0.")
    if not 0.0 < soz_positive_ratio < 1.0:
        raise ValueError("soz_positive_ratio must be between 0.0 and 1.0.")

    rng = np.random.default_rng(seed)
    participants: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    n_annotated = int(round(n_patients * annotated_patient_ratio))
    n_annotated = max(1, min(n_patients - 1, n_annotated))

    for p_idx in range(n_patients):
        participant_id = f"sub-{p_idx + 1:02d}"
        session_id = "ses-1"
        participants.append(
            {
                "participant_id": participant_id,
                "session": session_id,
                "age": int(rng.integers(18, 61)),
                "sex": "M" if p_idx % 2 == 0 else "F",
            }
        )

        electrodes = [f"E{e + 1:02d}" for e in range(n_electrodes)]
        coords: Dict[str, Tuple[float, float, float]] = {
            elec: (
                float(rng.uniform(-40, 40)),
                float(rng.uniform(-40, 40)),
                float(rng.uniform(-40, 40)),
            )
            for elec in electrodes
        }
        is_annotated_patient = p_idx < n_annotated
        if is_annotated_patient:
            n_positive = max(1, int(round(n_electrodes * soz_positive_ratio)))
            positive_indices = set(
                int(i) for i in rng.choice(n_electrodes, size=n_positive, replace=False)
            )
            soz_map = {
                elec: int(i in positive_indices) for i, elec in enumerate(electrodes)
            }
        else:
            # Mimic patients without SOZ-positive annotations.
            soz_map = {elec: 0 for elec in electrodes}
        stim_pairs = _pick_stim_pairs(electrodes=electrodes, n_pairs=n_stim_pairs, rng=rng)

        for rec in electrodes:
            for stim_1, stim_2 in stim_pairs:
                if rec in (stim_1, stim_2):
                    continue
                mean_resp = rng.normal(0.0, 1.0, size=(timesteps,)).astype(np.float32)
                std_resp = np.abs(rng.normal(0.35, 0.1, size=(timesteps,))).astype(np.float32)
                rec_x, rec_y, rec_z = coords[rec]
                rows.append(
                    {
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "run_id": "run-1",
                        "age": participants[-1]["age"],
                        "sex": participants[-1]["sex"],
                        "recording_electrode": rec,
                        "stim_1": stim_1,
                        "stim_2": stim_2,
                        "response_ts": _to_json_1d(mean_resp),
                        "response_ts_std": _to_json_1d(std_resp),
                        "soz_label": int(soz_map[rec]),
                        "recording_x": rec_x,
                        "recording_y": rec_y,
                        "recording_z": rec_z,
                    }
                )

    return pd.DataFrame(participants), pd.DataFrame(rows)


def _create_synthetic_dataset_root(
    base_dir: str,
    n_patients: int,
    n_electrodes: int,
    n_stim_pairs: int,
    timesteps: int,
    seed: int,
    annotated_patient_ratio: float,
    soz_positive_ratio: float,
) -> str:
    """Write synthetic RESPect CSVs under base_dir.

    Called from __main__ when --dataset-root is not set.

    Args:
        base_dir: Existing directory (usually from tempfile.TemporaryDirectory).
        n_patients: Cohort size.
        n_electrodes: Contacts per patient.
        n_stim_pairs: Stimulation pairs per patient.
        timesteps: Response length.
        seed: RNG seed for _build_synthetic_respect_table.
        annotated_patient_ratio: Annotated patient fraction.
        soz_positive_ratio: SOZ-positive electrode fraction.

    Returns:
        base_dir as a string after files are written.

    Raises:
        ValueError: If _build_synthetic_respect_table rejects the arguments.
    """

    root = Path(base_dir)
    participants_df, rows_df = _build_synthetic_respect_table(
        n_patients=n_patients,
        n_electrodes=n_electrodes,
        n_stim_pairs=n_stim_pairs,
        timesteps=timesteps,
        seed=seed,
        annotated_patient_ratio=annotated_patient_ratio,
        soz_positive_ratio=soz_positive_ratio,
    )
    participants_df.to_csv(root / "participants.tsv", sep="\t", index=False)
    rows_df.to_csv(root / "respect_ccep_data-pyhealth.csv", index=False)
    return str(root)


def get_spes_classification_configs() -> List[Dict[str, Any]]:
    """Return ordered model configurations for the benchmark sweep.

    Each dict has name, model_type (spes_resnet or spes_transformer),
    paradigm (divergent or convergent), and include_distance.

    Returns:
        List of configs passed to build_spes_classification_model and
        run_spes_soz_classification.

    Examples:
        >>> get_spes_classification_configs()[0]["model_type"]
        'spes_resnet'
    """

    return [
        {
            "name": "cnn_resnet_divergent_no_features",
            "model_type": "spes_resnet",
            "paradigm": "divergent",
            "include_distance": False,
        },
        {
            "name": "cnn_resnet_divergent_with_features",
            "model_type": "spes_resnet",
            "paradigm": "divergent",
            "include_distance": True,
        },
        {
            "name": "cnn_resnet_convergent_no_features",
            "model_type": "spes_resnet",
            "paradigm": "convergent",
            "include_distance": False,
        },
        {
            "name": "cnn_resnet_convergent_with_features",
            "model_type": "spes_resnet",
            "paradigm": "convergent",
            "include_distance": True,
        },
        {
            "name": "cnn_transformer_convergent_no_features",
            "model_type": "spes_transformer",
            "paradigm": "convergent",
            "include_distance": False,
        },
        {
            "name": "cnn_transformer_convergent_with_features",
            "model_type": "spes_transformer",
            "paradigm": "convergent",
            "include_distance": True,
        },
    ]


def build_spes_classification_model(config: Dict[str, Any], dataset):
    """Build one model instance for a sweep entry.

    Called from run_spes_soz_classification for each config.

    Args:
        config: Must include model_type and include_distance (see
            get_spes_classification_configs).
        dataset: Task-processed dataset (same instance passed to Trainer).

    Returns:
        SPESResNet or SPESTransformer with demo-sized hyperparameters.

    Raises:
        ValueError: If model_type is unknown.
    """

    include_distance = bool(config["include_distance"])
    if config["model_type"] == "spes_resnet":
        return SPESResNet(
            dataset=dataset,
            input_channels=4,
            noise_std=0.0,
            include_distance=include_distance,
        )
    if config["model_type"] == "spes_transformer":
        return SPESTransformer(
            dataset=dataset,
            mean=True,
            std=True,
            conv_embedding=True,
            mlp_embedding=True,
            num_layers=1,
            embedding_dim=16,
            random_channels=2,
            noise_std=0.0,
            include_distance=include_distance,
        )
    raise ValueError(f"Unsupported model_type: {config['model_type']}")


def _build_task_dataset_from_root(
    dataset_root: str,
    paradigm: str,
):
    """Load RESPect data and attach the SOZ localisation task.

    Called from run_spes_soz_classification before splitting.

    Args:
        dataset_root: Root passed to RESPectCCEPDataset.
        paradigm: spes_mode for SeizureOnsetZoneLocalisation (divergent or
            convergent).

    Returns:
        SampleDataset from set_task, ready for get_dataloader.
    """

    base_dataset = RESPectCCEPDataset(root=dataset_root)
    task = SeizureOnsetZoneLocalisation(spes_mode=paradigm)
    return base_dataset.set_task(task)


def _compute_paper_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute AUROC, AUPRC, sensitivity, specificity, and Youden index.

    AUROC is nan when y_true contains only one class. Called from
    run_spes_soz_classification on test predictions.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Decision threshold on y_prob. Default: 0.5.

    Returns:
        Metric name to float value (including auroc, auprc, sensitivity,
        specificity, youden).
    """

    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    youden = sensitivity + specificity - 1.0

    has_both_classes = np.unique(y_true).size == 2
    auroc = float(sk_metrics.roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")

    return {
        "auroc": auroc,
        "auprc": float(sk_metrics.average_precision_score(y_true, y_prob)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden": float(youden),
    }


def run_spes_soz_classification(dataset_root: str) -> List[Dict[str, float]]:
    """Train/evaluate every entry in get_spes_classification_configs.

    For each config: rebuild the task dataset from dataset_root, split
    train/val/test (patient-level when enough patients exist), fit with
    Trainer, then score on the test loader.

    Args:
        dataset_root: RESPect root with participants.tsv and
            respect_ccep_data-pyhealth.csv (real or synthetic).

    Returns:
        One dict per config: config (name string) plus metrics from
        _compute_paper_metrics and loss.
    """

    results: List[Dict[str, float]] = []
    for config in get_spes_classification_configs():
        sample_dataset = _build_task_dataset_from_root(
            dataset_root=dataset_root,
            paradigm=str(config["paradigm"]),
        )
        patient_count = int(len(sample_dataset.patient_to_index))
        if patient_count >= 3:
            train_dataset, val_dataset, test_dataset = split_by_patient(
                sample_dataset, [0.6, 0.2, 0.2], seed=SEED
            )
        else:
            # For tiny cohorts (e.g., one patient after filtering), patient-level
            # splitting can produce empty train/val sets. Fall back to sample-level
            # splitting to keep the script runnable for experimentation.
            train_dataset, val_dataset, test_dataset = split_by_sample(
                sample_dataset, [0.6, 0.2, 0.2], seed=SEED
            )
        train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = build_spes_classification_model(config=config, dataset=sample_dataset)
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=EPOCHS,
            monitor=None,
        )
        y_true_all, y_prob_all, loss_mean = trainer.inference(test_loader)
        metrics = _compute_paper_metrics(y_true=y_true_all, y_prob=y_prob_all)
        metrics["loss"] = float(loss_mean)

        summary = {"config": config["name"]}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                summary[key] = float(value)
        results.append(summary)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "SPES SOZ classification experiments. Use --dataset-root for a real "
            "RESPectCCEPDataset + SeizureOnsetZoneLocalisation run."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional path to RESPect CCEP (OpenNeuro ds004080) root.",
    )
    parser.add_argument(
        "--synthetic-patients",
        type=int,
        default=SYNTHETIC_N_PATIENTS,
        help="Number of synthetic patients when --dataset-root is omitted.",
    )
    parser.add_argument(
        "--synthetic-electrodes",
        type=int,
        default=SYNTHETIC_N_ELECTRODES,
        help="Number of electrodes per synthetic patient.",
    )
    parser.add_argument(
        "--synthetic-stim-pairs",
        type=int,
        default=SYNTHETIC_N_STIM_PAIRS,
        help="Number of stimulation pairs sampled per synthetic patient.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS,
        help="Response timeseries length for synthetic generation.",
    )
    parser.add_argument(
        "--synthetic-annotated-ratio",
        type=float,
        default=SYNTHETIC_ANNOTATED_PATIENT_RATIO,
        help="Fraction of synthetic patients with SOZ-positive annotations.",
    )
    parser.add_argument(
        "--synthetic-soz-positive-ratio",
        type=float,
        default=SYNTHETIC_SOZ_POSITIVE_RATIO,
        help="SOZ-positive electrode ratio for annotated synthetic patients.",
    )
    args = parser.parse_args()

    if args.dataset_root:
        classification_results = run_spes_soz_classification(dataset_root=args.dataset_root)
        print("SPES SOZ Classification Results (real dataset source)")
    else:
        with tempfile.TemporaryDirectory(prefix="spes_synth_respect_") as tmp_dir:
            synthetic_root = _create_synthetic_dataset_root(
                base_dir=tmp_dir,
                n_patients=int(args.synthetic_patients),
                n_electrodes=int(args.synthetic_electrodes),
                n_stim_pairs=int(args.synthetic_stim_pairs),
                timesteps=int(args.timesteps),
                seed=SEED,
                annotated_patient_ratio=float(args.synthetic_annotated_ratio),
                soz_positive_ratio=float(args.synthetic_soz_positive_ratio),
            )
            classification_results = run_spes_soz_classification(
                dataset_root=synthetic_root
            )
        print("SPES SOZ Classification Results (synthetic dataset source)")

    for row in classification_results:
        print(row)

