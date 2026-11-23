"""Paired H1 ablation with PyHealth Trainer, standard projection and refit.

Contributor: Chia Yang Tsai (UIUC NetID: ct68).
Paper: Sethi et al., ProtoECGNet, MLHC 2025.
https://proceedings.mlr.press/v298/sethi25a.html

Default: a short CPU-only synthetic demonstration, with no downloads. The two
arms use identical initialization, batch order, noise stream and epoch budgets.
Metrics are synthetic pipeline checks, not research or clinical findings.

Real data: pass --root pointing to PTB-XL v1.0.3. This explicitly starts NEW
training through Trainer on folds 1-8 and validation on fold 9. Fold 10 waveforms
are never loaded. This example does not reproduce the historical custom-Colab
training run, select new held-out models, or overwrite its checkpoints.

Examples:
    python examples/ptbxl_multilabel_classification_protoecgnet_h1.py
    python examples/ptbxl_multilabel_classification_protoecgnet_h1.py \
        --root /data/ptb-xl/1.0.3 --train-limit 128 --validation-limit 64 \
        --output-dir /tmp/protoecgnet_h1_demo
"""

import argparse
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyhealth.datasets import PTBXLDataset, SampleDataset, get_dataloader
from pyhealth.datasets.ptbxl import parse_scp_codes
from pyhealth.models import ProtoECGNet, ProtoECGNetH1
from pyhealth.models.protoecgnet_h1_checkpoint import bank_state_digest
from pyhealth.models.protoecgnet_h1_data import (
    PTBXL_MORPHOLOGY_LABEL_ORDER,
    PairedECGLoader,
    create_h1_ptbxl_dataset,
)
from pyhealth.trainer import Trainer
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

TRAIN_NOISE_STREAM = 2026111801
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class _ObservedPairedLoader(PairedECGLoader):
    """Fingerprint the actual clean/noisy batches consumed by Trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_hashes = {}
        self.consumed_batches = {}

    def __iter__(self):
        epoch = self.next_epoch
        return self._observe(super().__iter__(), epoch)

    def _observe(self, batches, epoch):
        digest = hashlib.sha256()
        for count, batch in enumerate(batches, start=1):
            digest.update(json.dumps([int(eid) for eid in batch["ecg_id"]]).encode())
            for key in ("signal", "noisy_signal", "labels"):
                value = batch[key].detach().cpu().contiguous()
                digest.update(
                    json.dumps([key, str(value.dtype), list(value.shape)]).encode()
                )
                digest.update(value.numpy().tobytes())
            self.stream_hashes[str(epoch)] = digest.hexdigest()
            self.consumed_batches[str(epoch)] = count
            yield batch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_splits(seed: int) -> tuple[SampleDataset, SampleDataset, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    splits = []
    train_targets = None
    for split, count, start, fold in [("train", 8, 1, 1), ("validation", 4, 101, 9)]:
        signals = torch.randn(count, 12, 1000, generator=generator) * 0.05
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]).repeat(
            count // 4, 1
        )
        time = torch.arange(1000) / 100
        signals[:, 0] += targets[:, 0, None] * torch.sin(2 * torch.pi * 3 * time)
        signals[:, 1] += targets[:, 1, None] * torch.cos(2 * torch.pi * 5 * time)
        dataset = create_h1_ptbxl_dataset(
            signals,
            targets,
            ecg_ids=list(range(start, start + count)),
            patient_ids=list(range(start, start + count)),
            folds=[fold] * count,
            split=split,
            label_names=["B", "A"],
        )
        splits.append(dataset)
        if split == "train":
            train_targets = targets
    return splits[0], splits[1], train_targets


def _ptbxl_splits(
    root: Path, output: Path, train_limit: int, val_limit: int
) -> tuple[SampleDataset, SampleDataset, Tensor]:
    """Reuse native PTBXLDataset; adapt the fixed morphology view explicitly.

    The existing general task skips all-zero targets and reads float32 before
    filtering. Here all SCP keys (including likelihood zero) are positives,
    only the 52 fixed morphology labels are retained, and high-pass filtering
    precedes the float32 cast. No extra standardization is fitted or applied.
    """
    import wfdb
    from scipy.signal import butter, filtfilt

    native = PTBXLDataset(
        root=str(root),
        sampling_rate=100,
        metadata_cache_dir=output / "metadata_cache",
        cache_dir=output / "dataset_cache",
        num_workers=1,
    )
    descriptors = {"train": [], "validation": []}
    for patient_id in native.unique_patient_ids:
        patient = native.get_patient(patient_id)
        for event in patient.get_events(event_type="records"):
            fold = int(float(event.strat_fold))
            if fold not in range(1, 10):
                continue  # Skip held-out records BEFORE waveform access.
            split = "validation" if fold == 9 else "train"
            descriptors[split].append(
                (int(float(event.record_id)), int(patient_id), fold, event)
            )
    labels = list(PTBXL_MORPHOLOGY_LABEL_ORDER)
    b, a = butter(1, 0.5 / (100 / 2), btype="high")
    built = []
    train_targets = None
    for split, limit in [("train", train_limit), ("validation", val_limit)]:
        records = sorted(descriptors[split], key=lambda record: record[0])
        if limit:
            records = records[:limit]
        if not records:
            raise ValueError(f"No {split} ECGs selected.")
        signals = []
        targets = []
        for _, _, _, event in records:
            if int(event.sampling_rate) != 100:
                raise ValueError("Expected 100 Hz.")
            signal, fields = wfdb.rdsamp(event.signal_file)
            if signal.shape != (1000, 12) or fields["sig_name"] != LEADS:
                raise ValueError(
                    "Expected 1000 samples and the canonical 12-lead order."
                )
            signals.append(
                np.ascontiguousarray(
                    filtfilt(b, a, signal.T, axis=-1), dtype=np.float32
                )
            )
            codes = parse_scp_codes(event.scp_codes)
            targets.append([float(label in codes) for label in labels])
        y = torch.tensor(targets, dtype=torch.float32)
        built.append(
            create_h1_ptbxl_dataset(
                torch.from_numpy(np.stack(signals)),
                y,
                ecg_ids=[r[0] for r in records],
                patient_ids=[r[1] for r in records],
                folds=[r[2] for r in records],
                split=split,
                label_names=labels,
            )
        )
        if split == "train":
            train_targets = y
    train_patients = {built[0][i]["patient_id"] for i in range(len(built[0]))}
    validation_patients = {built[1][i]["patient_id"] for i in range(len(built[1]))}
    if train_patients & validation_patients:
        raise ValueError("Patient overlap between splits.")
    return built[0], built[1], train_targets


def _tensor_state(
    model: ProtoECGNetH1, *, feature_only: bool = False
) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, Tensor)
        and (
            not feature_only
            or name.startswith(
                (
                    "branches.morphology.backbone.",
                    "branches.morphology.add_on.",
                    "branches.morphology.prototype_vectors",
                )
            )
        )
    }


def _equal_state(model: ProtoECGNetH1, expected: dict[str, Tensor]) -> bool:
    current = model.state_dict()
    return all(
        torch.equal(current[name].detach().cpu(), value)
        for name, value in expected.items()
    )


def _classification(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    positives = y.sum(0)
    negatives = len(y) - positives
    eligible = (positives > 0) & (negatives > 0)
    return {
        "macro_auroc": float(
            roc_auc_score(y[:, eligible], p[:, eligible], average="macro")
        )
        if eligible.any()
        else None,
        "macro_ap": float(
            average_precision_score(y[:, eligible], p[:, eligible], average="macro")
        )
        if eligible.any()
        else None,
        "eligible_labels": int(eligible.sum()),
        "total_labels": y.shape[1],
        "positive_support": positives.astype(int).tolist(),
        "negative_support": negatives.astype(int).tolist(),
    }


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    """Run both new-training arms; save local results and reloadable checkpoints."""
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    synthetic = args.root is None
    if synthetic:
        train, validation, targets = _synthetic_splits(args.seed)
    else:
        train, validation, targets = _ptbxl_splits(
            args.root.resolve(), output, args.train_limit, args.validation_limit
        )
    weights, cooc = ProtoECGNet.compute_label_statistics(targets)
    options = {
        "proto_dim": 16 if synthetic else 512,
        "prototypes_per_label": 2 if synthetic else 18,
        "class_weights": weights,
        "label_cooccurrence": cooc,
    }
    parent = ProtoECGNetH1(train, **options)
    parent_state = copy.deepcopy(parent.state_dict())
    labels = list(parent.label_names)
    del parent
    report = {
        "scope": "synthetic_pipeline_demo"
        if synthetic
        else "new_ptbxl_fold9_training_example",
        "research_formal_training_reproduced": False,
        "fold10_used": False,
        "device": "cpu",
        "seed": args.seed,
        "joint_epochs": args.joint_epochs,
        "refit_epochs": args.refit_epochs,
        "training_noise_stream": TRAIN_NOISE_STREAM,
        "noise_alpha": 0.01,
        "label_names": labels,
        "train_records": len(train),
        "validation_records": len(validation),
        "arms": {},
    }
    for arm, coefficient in [
        ("paired_lambda0", 0.0),
        ("paired_consistency", args.lambda_cons),
    ]:
        model = ProtoECGNetH1(train, **options)
        model.load_state_dict(parent_state, strict=True)
        model.lambda_cons = coefficient
        trainer = Trainer(model=model, device="cpu", enable_logging=False)
        pairs = _ObservedPairedLoader(
            train,
            batch_size=args.batch_size,
            stream_seed=TRAIN_NOISE_STREAM,
            shuffle=True,
            seed=args.seed,
        )
        initial_bank_hash = bank_state_digest(
            model.state_dict(), labels, model.get_extra_state()["architecture"]
        )
        initial_classifier = (
            model.branches["morphology"].classifier.weight.detach().clone()
        )
        initial_classifier_hash = hashlib.sha256(
            initial_classifier.cpu().numpy().tobytes()
        ).hexdigest()
        trainer.train(
            pairs,
            epochs=args.joint_epochs,
            optimizer_params={"lr": args.learning_rate},
            weight_decay=0.0,
            max_grad_norm=None,
            load_best_model_at_last=False,
        )
        classifier_frozen = torch.equal(
            initial_classifier, model.branches["morphology"].classifier.weight
        )
        if not classifier_frozen:
            raise RuntimeError("Joint stage updated the frozen classifier.")
        if pairs.next_epoch != 1 + args.joint_epochs:
            raise RuntimeError("Noise epoch did not advance by full passes.")
        train_clean = get_dataloader(train, batch_size=args.batch_size, shuffle=False)
        # Native dataset IDs are ascending; standard projection captures sources.
        model.project_prototypes(
            train_clean,
            source_context={
                "example": "paired_H1_ablation",
                "preprocessing": "synthetic"
                if synthetic
                else "reference_remove_baseline_wander_100hz_no_standardization",
                "label_order": labels,
                "data_origin": "synthetic" if synthetic else "PTBXLDataset v1.0.3",
            },
        )
        model.set_stage("classifier")
        frozen = _tensor_state(model, feature_only=True)
        trainer.train(
            train_clean,
            epochs=args.refit_epochs,
            optimizer_params={"lr": args.learning_rate},
            weight_decay=0.0,
            max_grad_norm=None,
            load_best_model_at_last=False,
        )
        features_frozen = _equal_state(model, frozen)
        if not features_frozen:
            raise RuntimeError("Refit changed features, prototypes or BN buffers.")
        model.eval()
        provenance = model.get_projection_provenance()
        if provenance is None:
            raise RuntimeError("Missing bank-linked projection sources.")
        checkpoint = output / f"{arm}.ckpt"
        trainer.save_ckpt(str(checkpoint))
        restored = ProtoECGNetH1(train, **options)
        restored_trainer = Trainer(model=restored, device="cpu", enable_logging=False)
        restored_trainer.load_ckpt(str(checkpoint))
        restored.eval()
        probe = next(
            iter(get_dataloader(validation, batch_size=args.batch_size, shuffle=False))
        )
        with torch.no_grad():
            actual = model(**probe)
            reloaded = restored(**probe)
        if any(
            not torch.equal(actual[key], reloaded[key])
            for key in ("logit", "similarity", "y_prob")
        ):
            raise RuntimeError("Native checkpoint reload changed predictions.")
        if restored.get_projection_provenance() != provenance:
            raise RuntimeError("Reload lost source binding.")
        y, p, loss = trainer.inference(
            get_dataloader(validation, batch_size=args.batch_size, shuffle=False)
        )
        report["arms"][arm] = {
            "lambda_cons": coefficient,
            **_classification(y, p),
            "validation_refit_loss": loss,
            "joint_classifier_frozen": classifier_frozen,
            "refit_feature_state_frozen": features_frozen,
            "noise_epochs_completed": pairs.next_epoch - 1,
            "projection_covered": provenance["covered_prototypes"],
            "projection_total": len(provenance["source_records"]),
            "strict_reload": True,
            "checkpoint_sha256": _sha256(checkpoint),
        }
        report["arms"][arm].update(
            {
                "initial_bank_state_sha256": initial_bank_hash,
                "initial_classifier_sha256": initial_classifier_hash,
                "consumed_batch_stream_sha256": pairs.stream_hashes,
                "consumed_batches_per_epoch": pairs.consumed_batches,
            }
        )
        if any(count != len(pairs) for count in pairs.consumed_batches.values()):
            raise RuntimeError("Trainer consumed an incomplete noise epoch.")
        terms = model.prototype_contributions(probe[model.signal_key])["morphology"][
            0, 0
        ].cpu()
        illustrative = {
            "scope": "First validation ECG and first vocabulary label; not formal research case analysis.",
            "ecg_id": int(probe["ecg_id"][0]),
            "label": labels[0],
            "logit": actual["logit"][0, 0].item(),
            "probability": actual["y_prob"][0, 0].item(),
        }
        for sign in ("positive", "negative"):
            magnitude = terms if sign == "positive" else -terms
            selected = torch.argsort(magnitude, descending=True)[
                : min(5, len(terms))
            ].tolist()
            illustrative[sign + "_sources"] = [
                {
                    "prototype_id": index,
                    "signed_contribution": terms[index].item(),
                    "source": provenance["source_records"][index],
                }
                for index in selected
                if magnitude[index] > 0
            ]
        (output / f"{arm}_illustrative_evidence.json").write_text(
            json.dumps(illustrative, indent=2) + "\n"
        )
        np.savez(output / f"{arm}_validation.npz", y_true=y, y_prob=p)
    first, second = report["arms"].values()
    for key in (
        "initial_bank_state_sha256",
        "initial_classifier_sha256",
        "consumed_batch_stream_sha256",
        "consumed_batches_per_epoch",
    ):
        if first[key] != second[key]:
            raise RuntimeError(f"Paired ablation mismatch: {key}")
    report["matched_initialization_and_consumed_batches"] = True
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    table = [
        "| Arm | Lambda | Validation macro AUROC | Validation macro AP | Eligible labels |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, m in report["arms"].items():
        table.append(
            f"| {arm} | {m['lambda_cons']:g} | {m['macro_auroc']} | {m['macro_ap']} | {m['eligible_labels']}/{m['total_labels']} |"
        )
    (output / "comparison.md").write_text("\n".join(table) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="PTB-XL v1.0.3 root; omission runs synthetic CPU demo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pyhealth_protoecgnet_h1_demo"),
        help="NEW output directory; existing directories are never overwritten.",
    )
    parser.add_argument("--joint-epochs", type=int, default=2)
    parser.add_argument("--refit-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=598)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-cons", type=float, default=117608.0)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=128,
        help="Real-data subset size; 0 uses all training records.",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=64,
        help="Real-data subset size; 0 uses all fold-9 records.",
    )
    args = parser.parse_args()
    if min(args.joint_epochs, args.refit_epochs, args.batch_size) < 1:
        parser.error("Epochs and batch size must be positive.")
    if min(args.train_limit, args.validation_limit) < 0:
        parser.error("Subset limits must be nonnegative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if not math.isfinite(args.lambda_cons) or args.lambda_cons < 0:
        parser.error("Consistency coefficient must be finite and nonnegative.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32).")
    logging.basicConfig(level=logging.WARNING)
    run_demo(args)


if __name__ == "__main__":
    main()
