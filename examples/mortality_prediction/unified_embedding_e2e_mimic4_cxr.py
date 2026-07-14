"""End-to-end protocol runner for Unified Embedding on MIMIC-IV + CXR.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on MIMIC-IV mortality using
clinical notes, ICD codes, lab values, and chest X-rays.

Example
-------
    python examples/mortality_prediction/unified_embedding_e2e_mimic4_cxr.py \\
      --ehr-root /shared/rsaas/physionet.org/files/mimiciv/2.2 \\
      --note-root /shared/rsaas/physionet.org/files/mimic-note \\
      --cxr-root /shared/rsaas/physionet.org/files/MIMIC-CXR \\
      --task clinical_notes_icd_labs_cxr \\
      --model ehrmamba --num-layers 2 \\
      --epochs 10 --batch-size 8 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import MLP, RNN, Transformer, UnifiedMultimodalEmbeddingModel
from pyhealth.models.bottleneck_transformer import BottleneckTransformer
from pyhealth.models.ehrmamba import EHRMamba
from pyhealth.models.jamba_ehr import JambaEHR
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.tasks.multimodal_mimic4 import (
    ClinicalNotesICDLabsMIMIC4,
    ClinicalNotesICDLabsCXRMIMIC4,
)
from pyhealth.trainer import Trainer


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]
    note_tables = None

    # Notes required for any task that includes clinical text
    if args.task in ("clinical_notes_icd_labs", "clinical_notes_icd_labs_cxr"):
        if not args.note_root:
            raise ValueError(f"--task {args.task} requires --note-root.")
        note_tables = ["discharge", "radiology"]

    # CXR metadata tables only loaded for the CXR task
    cxr_kwargs = {}
    if args.task == "clinical_notes_icd_labs_cxr":
        if not args.cxr_root:
            raise ValueError("--task clinical_notes_icd_labs_cxr requires --cxr-root.")
        cxr_kwargs = dict(
            cxr_root=args.cxr_root,
            cxr_variant=args.cxr_variant,
            cxr_tables=["metadata", "negbio", "chexpert", "split"],
        )

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev,
        num_workers=args.num_workers,
        **cxr_kwargs,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "stagenet":
        return MortalityPredictionStageNetMIMIC4()
    if args.task == "clinical_notes_icd_labs":
        return ClinicalNotesICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task == "clinical_notes_icd_labs_cxr":
        return ClinicalNotesICDLabsCXRMIMIC4(window_hours=args.observation_window_hours)
    raise ValueError(f"Unknown task: {args.task}")


def _split_dataset(dataset: Any, seed: int) -> Tuple[Any, Any, Any]:
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=seed)
    if len(train_ds) == 0 or len(test_ds) == 0:
        train_ds, val_ds, test_ds = split_by_sample(dataset, [0.8, 0.1, 0.1], seed=seed)
    return train_ds, val_ds, test_ds


def _build_model(args: argparse.Namespace, sample_dataset: Any):
    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
    )

    if args.model == "mlp":
        return MLP(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            unified_embedding=unified,
        )
    if args.model == "rnn":
        return RNN(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            unified_embedding=unified,
            rnn_type=args.rnn_type,
            num_layers=args.rnn_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
        )
    if args.model == "transformer":
        return Transformer(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            unified_embedding=unified,
        )
    if args.model == "bottleneck_transformer":
        return BottleneckTransformer(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            bottlenecks_n=args.bottlenecks_n,
            fusion_startidx=args.fusion_startidx,
            num_layers=args.num_layers,
            heads=args.heads,
            dropout=args.dropout,
            unified_embedding=unified,
        )
    if args.model == "ehrmamba":
        return EHRMamba(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            state_size=args.mamba_state_size,
            conv_kernel=args.mamba_conv_kernel,
            dropout=args.dropout,
            unified_embedding=unified,
        )
    if args.model == "jambaehr":
        return JambaEHR(
            dataset=sample_dataset,
            embedding_dim=args.embedding_dim,
            num_transformer_layers=args.jamba_transformer_layers,
            num_mamba_layers=args.jamba_mamba_layers,
            heads=args.heads,
            dropout=args.dropout,
            state_size=args.mamba_state_size,
            conv_kernel=args.mamba_conv_kernel,
            unified_embedding=unified,
        )
    raise ValueError(f"Unknown model: {args.model}")


def _write_predictions(
    output_csv: Path,
    patient_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    y_true_flat = y_true.reshape(-1).tolist()
    y_prob_flat = y_prob.reshape(-1).tolist()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patient_id", "y_true", "y_prob", "y_pred_threshold_0_5"],
        )
        writer.writeheader()
        for idx, prob in enumerate(y_prob_flat):
            writer.writerow(
                {
                    "patient_id": patient_ids[idx],
                    "y_true": int(y_true_flat[idx]),
                    "y_prob": float(prob),
                    "y_pred_threshold_0_5": int(float(prob) >= 0.5),
                }
            )


def run(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust settings."
        )

    train_ds, val_ds, test_ds = _split_dataset(sample_dataset, seed=args.seed)
    model = _build_model(args, sample_dataset)

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
        if len(test_ds) > 0
        else None
    )

    exp_name = f"{args.model}_seed{args.seed}"
    output_dir = Path(args.output_dir)

    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
        device=args.device,
        enable_logging=True,
        output_path=str(output_dir),
        exp_name=exp_name,
    )

    # BT uses tighter optimizer settings to stabilize training on full MIMIC-IV
    effective_lr = args.lr
    effective_max_grad_norm = args.max_grad_norm
    optimizer_params = {}

    if args.model == "bottleneck_transformer":
        if effective_lr is None:
            effective_lr = 1e-4
        if effective_max_grad_norm is None:
            effective_max_grad_norm = 0.5
        optimizer_params["eps"] = args.adam_eps if args.adam_eps is not None else 1e-6
    else:
        if effective_lr is None:
            effective_lr = 1e-3
        if args.adam_eps is not None:
            optimizer_params["eps"] = args.adam_eps

    optimizer_params["lr"] = effective_lr

    if args.epochs > 0 and len(train_ds) > 0:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=effective_max_grad_norm,
            monitor="pr_auc",
            load_best_model_at_last=True,
        )

    inference_loader = test_loader or val_loader or train_loader
    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    output_csv = output_dir / exp_name / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV + CXR with any of six backbone models."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cxr-root", type=str, default=None)
    parser.add_argument("--cxr-variant", type=str, default="sunlab", choices=["default", "sunlab"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e_cxr")

    parser.add_argument(
        "--task", type=str, default="clinical_notes_icd_labs_cxr",
        choices=["stagenet", "clinical_notes_icd_labs", "clinical_notes_icd_labs_cxr"],
    )
    parser.add_argument(
        "--model", type=str, default="rnn",
        choices=["mlp", "rnn", "transformer", "bottleneck_transformer", "ehrmamba", "jambaehr"],
    )

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--adam-eps", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", action="store_true")

    parser.add_argument("--observation-window-hours", type=int, default=24)

    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    parser.add_argument("--bottlenecks-n", type=int, default=4)
    parser.add_argument("--fusion-startidx", type=int, default=1)

    parser.add_argument("--max-grad-norm", type=float, default=None)

    parser.add_argument("--mamba-state-size", type=int, default=16)
    parser.add_argument("--mamba-conv-kernel", type=int, default=4)
    parser.add_argument("--jamba-transformer-layers", type=int, default=2)
    parser.add_argument("--jamba-mamba-layers", type=int, default=6)

    return parser.parse_args()


def print_vram_summary(device: Optional[str] = None) -> None:
    """Print peak allocated and total VRAM for the given CUDA device."""
    if not torch.cuda.is_available():
        print("VRAM summary: CUDA not available.")
        return
    idx = torch.device(device).index if device and device.startswith("cuda") else 0
    if idx is None:
        idx = 0
    allocated_mb = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
    print(f"VRAM summary (cuda:{idx}): peak_allocated={allocated_mb:.0f} MB / total={total_mb:.0f} MB")


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")
    print_vram_summary(args.device)
