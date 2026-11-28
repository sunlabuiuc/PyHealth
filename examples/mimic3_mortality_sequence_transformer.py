"""Ablation of Sequence Transformer Networks on MIMIC-III ICU mortality

Contributors:
    Matthew Nguyen (mnguyen358, mnguyen358@gatech.edu)
    Alireza Ghiam (aghiam3, aghiam3@gatech.edu)

Paper:
    Jeeheh Oh, Jiaxuan Wang and Jenna Wiens. "Learning to Exploit Invariances
    in Clinical Time-Series Data using Sequence Transformer Networks."
    MLHC 2018. https://proceedings.mlr.press/v85/oh18a.html

Experiment:
The paper compares a 1D CNN with the same CNN behind a Sequence Transformer,
and splits the transformer into its temporal part (time warp and theta) and
magnitude part (value shift/scale and phi).

- Data: MIMIC-III demo (100 patients, https://physionet.org/content/mimiciii-demo/1.4/)
- Task: YAIB ICU mortality, bins 0-24 (25 hourly bins). The last bin covers
  hours 24-25. The paper uses 48 hours to predict in-hospital mortality.
- Inputs: dynamic measurements and missing-value indicators. Height/weight
  are left out because the YAIB static values can include later measurements.
- Training: batch 15, Adam lr 1e-3, 10 epochs. Keep the model with the lowest
  validation loss; the paper selected by validation AUROC.
- Evaluation: 60/20/20 patient split, preprocessing fitted on training
  patients only, repeated over 10 random splits. All configurations start
  with the same CNN weights and training sample order for each split.
- Ablation: temporal only, magnitude only, and both, as in the paper.
  Also compares dropout 0, 0.2, and 0.5 for the full STN.
- Matched-dropout study: CNN and full STN both use dropout 0.2, isolating
  the transformation from the original configurations' dropout difference.

Results:
Writes each run to JSON and prints the mean and SD. See README_STN.md for
the results without static features. AUPR here is average precision.

The splits overlap, so the SD is not a confidence interval. The demo is small
and contains patients who eventually died. The task also differs from the
paper, so the scores are not directly comparable. YAIB still uses gaps in
measurements over the whole stay to select patients. This tests the model
on a retrospective dataset; it does not validate use on new ICU patients.

Usage:

    python examples/mimic3_mortality_sequence_transformer.py \\
        --root /path/to/mimic-iii-clinical-database-demo-1.4

    Add --study matched-dropout for the two-configuration control.

"""

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader
from pyhealth.models import SequenceTransformerNetwork
from pyhealth.tasks import YAIBMortalityMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

# Folder with the MIMIC-III CSV files (the unzipped demo, or full MIMIC-III).
# Set it here, or pass --root on the command line.
MIMIC_ROOT = "mimic-iii-clinical-database-demo-1.4"

SEQ_LEN = YAIBMortalityMIMIC3.OBSERVATION_HOURS + 1  # hourly bins 0-24

CONFIGS = {  # rows 1-4: the paper's Table 2; rows 5-6: dropout of the full model
    "baseline CNN": {"temporal": False, "magnitude": False, "dropout": 0.3},
    "temporal only": {"temporal": True, "magnitude": False, "dropout": 0.2},
    "magnitude only": {"temporal": False, "magnitude": True, "dropout": 0.2},
    "STN (both)": {"temporal": True, "magnitude": True, "dropout": 0.2},
    "STN, dropout 0": {"temporal": True, "magnitude": True, "dropout": 0.0},
    "STN, dropout 0.5": {"temporal": True, "magnitude": True, "dropout": 0.5},
}

MATCHED_DROPOUT_CONFIGS = {
    "CNN, dropout 0.2": {"temporal": False, "magnitude": False, "dropout": 0.2},
    "STN (both)": {"temporal": True, "magnitude": True, "dropout": 0.2},
}


class DynamicOnlyMortality(YAIBMortalityMIMIC3):
    """YAIB mortality with only the dynamic features.

    YAIB still computes static values to select patients, but the model
    only uses the dynamic field.
    """

    task_name = "SequenceTransformerDynamicOnlyMortalityMIMIC3"
    input_schema: ClassVar[dict[str, Any]] = {
        "dynamic": deepcopy(YAIBMortalityMIMIC3.input_schema["dynamic"])
    }


def run(train, val, test, config, seed, initial_cnn):
    """Trains one configuration and returns test AUROC, AUPR, theta-I, phi-I"""
    set_seed(seed)
    model = SequenceTransformerNetwork(train, seq_len=SEQ_LEN, **config)
    model.cnn.load_state_dict(initial_cnn)
    # Same training seed after building each model
    set_seed(seed + 10000)
    trainer = Trainer(model=model, device="cpu", metrics=[], enable_logging=False)
    trainer.train(
        train_dataloader=get_dataloader(train, 15, shuffle=True, seed=seed),
        val_dataloader=get_dataloader(val, 15),
        epochs=10,
        optimizer_params={"lr": 1e-3},
        monitor="loss",
        monitor_criterion="min",
        keep_best_in_memory=True,
    )
    y_true, y_prob, _ = trainer.inference(get_dataloader(test, 15))

    model.eval()
    with torch.no_grad():
        out = model.transform(**next(iter(get_dataloader(test, len(test)))))
    identity = torch.tensor([1.0, 0.0])
    theta = (out["theta"] - identity).abs().mean().item()
    phi = (out["phi"] - identity).abs().mean().item()

    if len(np.unique(y_true)) < 2:  # if it lacks deaths
        return np.nan, np.nan, theta, phi
    auroc = roc_auc_score(y_true.ravel(), y_prob.ravel())
    aupr = average_precision_score(y_true.ravel(), y_prob.ravel())
    return auroc, aupr, theta, phi


def split_details(parts):
    """Checks patient splits and counts stays and labels. Does not save IDs."""
    patient_sets = [set(part.patient_to_index) for part in parts]
    if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Patient overlap between training, validation and test")
    details = {}
    for name, part, patients in zip(("train", "val", "test"), parts, patient_sets):
        labels = [int(sample["mortality"].item()) for sample in part]
        details[name] = {
            "stays": len(part),
            "patients": len(patients),
            "positive_labels": sum(labels),
            "negative_labels": len(labels) - sum(labels),
        }
        if not labels:
            raise ValueError(f"{name} partition is empty")
    return details


def summarize(scores):
    """Mean, SD, and number of valid runs for each metric"""
    summary = {}
    for name, runs in scores.items():
        values = np.asarray(runs, dtype=float)
        if not len(values):
            continue
        result = {}
        for column, metric in enumerate(
            ("auroc", "aupr_ap", "theta_minus_I", "phi_minus_I")
        ):
            valid = values[:, column][np.isfinite(values[:, column])]
            result[metric] = {
                "n": len(valid),
                "mean": float(valid.mean()) if len(valid) else None,
                "sd": float(valid.std(ddof=1)) if len(valid) > 1 else None,
            }
        summary[name] = result
    return summary


def main():
    parser = argparse.ArgumentParser(description="STN ablation on MIMIC-III")
    parser.add_argument("--root", default=MIMIC_ROOT, help="MIMIC-III CSV folder")
    parser.add_argument("--seeds", type=int, default=10, help="random splits")
    parser.add_argument(
        "--study", choices=("ablation", "matched-dropout"), default="ablation"
    )
    parser.add_argument("--output", type=Path, help="JSON path; defaults by study")
    parser.add_argument("--cache-dir", type=Path, help="optional dataset cache folder")
    args = parser.parse_args()
    if not Path(args.root).expanduser().is_dir():
        parser.error(f"MIMIC-III folder not found: {args.root!r}")
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    configs = CONFIGS if args.study == "ablation" else MATCHED_DROPOUT_CONFIGS
    if args.output is None:
        filename = (
            "mimic3_stn_dynamic_only.json"
            if args.study == "ablation"
            else "mimic3_stn_matched_dropout.json"
        )
        args.output = Path("results") / filename

    kwargs = {"cache_dir": str(args.cache_dir)} if args.cache_dir else {}
    dataset = MIMIC3Dataset(
        root=str(Path(args.root).expanduser()),
        tables=["labevents", "chartevents", "outputevents", "transfers"],
        **kwargs,
    )
    scores = {name: [] for name in configs}
    output = {
        "status": "running",
        "study": args.study,
        "created_utc": datetime.now(UTC).isoformat(),
        "example_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task": DynamicOnlyMortality.task_name,
        "outcome": "ICU mortality, not the paper's in-hospital mortality",
        "input": "dynamic only; 25 hourly bins numbered 0 through 24",
        "cohort_caveat": "YAIB retrospective whole-stay measurement-gap eligibility retained",
        "split_ratios": [0.6, 0.2, 0.2],
        "seeds": list(range(args.seeds)),
        "epochs": 10,
        "batch_size": 15,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "checkpoint": "minimum validation loss; restored from memory",
        "paired_initial_classifier": True,
        "aupr_definition": "sklearn average_precision_score",
        "configs": configs,
        "versions": {
            name: version(name)
            for name in ("pyhealth", "torch", "numpy", "scikit-learn")
        },
        "splits": [],
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for seed in range(args.seeds):
        train, val, test = dataset.set_task(
            DynamicOnlyMortality(),
            split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=seed),
            num_workers=1,
        )
        output["splits"].append({"seed": seed, **split_details((train, val, test))})
        set_seed(seed)
        reference = SequenceTransformerNetwork(
            train, seq_len=SEQ_LEN, **CONFIGS["baseline CNN"]
        )
        initial_cnn = deepcopy(reference.cnn.state_dict())
        del reference
        for name, config in configs.items():
            result = run(train, val, test, config, seed, initial_cnn)
            scores[name].append(result)
            output["runs"].append(
                {
                    "seed": seed,
                    "configuration": name,
                    **{
                        metric: float(value) if np.isfinite(value) else None
                        for metric, value in zip(
                            ("auroc", "aupr_ap", "theta_minus_I", "phi_minus_I"), result
                        )
                    },
                }
            )
            output["summary"] = summarize(scores)
            args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")

    output["status"] = "complete"
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"\nTest mean SD over {args.seeds} random patient splits:")
    print(f"{'Configuration':18s} {'AUROC':15s} {'AUPR':15s} theta-I  phi-I")
    for name, runs in scores.items():
        auroc, aupr, theta, phi = np.array(runs).T
        auroc_s = f"{np.nanmean(auroc):.3f} ({np.nanstd(auroc, ddof=1):.3f})"
        aupr_s = f"{np.nanmean(aupr):.3f} ({np.nanstd(aupr, ddof=1):.3f})"
        transforms = f"{theta.mean():9.3f}  {phi.mean():7.3f}"
        print(f"{name:18s} {auroc_s:15s} {aupr_s:15s} {transforms}")


if __name__ == "__main__":
    main()
