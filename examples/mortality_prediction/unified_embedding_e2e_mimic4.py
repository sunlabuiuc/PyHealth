"""End-to-end protocol runner for Unified Embedding on MIMIC-IV.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer) on a MIMIC-IV mortality task, then writes per-sample
predictions to CSV.

Tasks
-----
--task stagenet (default)
    MortalityPredictionStageNetMIMIC4: ICD codes + 10-dim lab vectors,
    patient-level samples aggregated across all admissions.

--task clinical_notes_icd_labs
    ClinicalNotesICDLabsMIMIC4: discharge/radiology notes + ICD + labs.
    Requires --note-root.

Example
-------
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --task stagenet \\
      --model transformer \\
      --heads 4 --num-layers 2 \\
      --dev --device cpu \\
      --epochs 10 --batch-size 32 --lr 1e-3 \\
      --output-dir ./output/unified_e2e
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import MLP, RNN, Transformer, UnifiedMultimodalEmbeddingModel
from pyhealth.models.bottleneck_transformer import BottleneckTransformer
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesICDLabsMIMIC4
from pyhealth.trainer import Trainer


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]
    note_tables = None

    if args.task == "clinical_notes_icd_labs":
        if not args.note_root:
            raise ValueError("--task clinical_notes_icd_labs requires --note-root.")
        note_tables = ["discharge", "radiology"]

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev,
        num_workers=args.num_workers,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "stagenet":
        return MortalityPredictionStageNetMIMIC4()
    if args.task == "clinical_notes_icd_labs":
        return ClinicalNotesICDLabsMIMIC4(window_hours=args.observation_window_hours)
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

    trainer = Trainer(
        model=model,
        metrics=["accuracy"],
        device=args.device,
        enable_logging=False,
    )

    if args.epochs > 0 and len(train_ds) > 0:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params={"lr": args.lr},
            monitor=None,
            load_best_model_at_last=False,
        )

    inference_loader = test_loader or val_loader or train_loader
    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    output_dir = Path(args.output_dir)
    output_csv = output_dir / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV with any of four sequence heads."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")

    parser.add_argument(
        "--task",
        type=str,
        choices=["stagenet", "clinical_notes_icd_labs"],
        default="stagenet",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rnn", "transformer", "bottleneck_transformer"],
        default="rnn",
    )

    # Shared embedding / training
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", action="store_true")

    # Task-specific
    parser.add_argument("--observation-window-hours", type=int, default=24)

    # RNN-specific
    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    # Transformer / BottleneckTransformer shared
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

    # BottleneckTransformer-specific
    parser.add_argument("--bottlenecks-n", type=int, default=4)
    parser.add_argument("--fusion-startidx", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")
