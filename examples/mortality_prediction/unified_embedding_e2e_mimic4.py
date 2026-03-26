"""End-to-end protocol runner for Unified Embedding + MLP/RNN on MIMIC-IV.

Objective
---------
Establish a reproducible E2E (End-to-end) test path for unified multimodal embedding
with both MLP and RNN heads:

1. ingest MIMIC-IV data,
2. construct multimodal task samples (MultimodalMortalityHorizonMIMIC4),
3. train/infer with UnifiedMultimodalEmbeddingModel,
4. emit concrete prediction rows.

Task
----
pyhealth.tasks.MultimodalMortalityHorizonMIMIC4

- Observation window: configurable (default 24h)
- Prediction horizon: configurable (default 12h)
- Label: in-hospital mortality within horizon
- Inputs:
    - icd_codes (diagnoses + procedures, StageNet)
    - labs (10-category lab vectors, StageNet tensor)
    - optional notes (discharge/radiology, tuple-time-text)

Criteria
-------------------
For each model head (MLP and RNN):

1. dataset.set_task(MultimodalMortalityHorizonMIMIC4(...)) returns non-empty samples.
2. A forward pass on a batch returns y_prob and loss.
3. Inference returns aligned arrays of patient_id, y_true, y_prob.
4. Predictions are written to CSV with one row per sample.

Reference Test
--------------
tests/core/test_unified_e2e_mimic4.py validates the end-to-end path.

    pytest tests/core/test_unified_e2e_mimic4.py -v

Run on Full MIMIC-IV
--------------------
RNN head:

    PYTHONPATH=. PYHEALTH_DISABLE_DASK_DISTRIBUTED=1 \\
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --model rnn \\
      --observation-window-hours 24 \\
      --prediction-horizon-hours 12 \\
      --epochs 3 \\
      --batch-size 64 \\
      --output-dir ./output/unified_e2e

MLP head:

    PYTHONPATH=. PYHEALTH_DISABLE_DASK_DISTRIBUTED=1 \\
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --model mlp \\
      --observation-window-hours 24 \\
      --prediction-horizon-hours 12 \\
      --epochs 3 \\
      --batch-size 64 \\
      --output-dir ./output/unified_e2e

Outputs
----------------
- <output-dir>/predictions_rnn.csv
- <output-dir>/predictions_mlp.csv

Columns: patient_id, y_true, y_prob, y_pred_threshold_0_5
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
from pyhealth.models import MLP, RNN, UnifiedMultimodalEmbeddingModel
from pyhealth.tasks import MultimodalMortalityHorizonMIMIC4
from pyhealth.trainer import Trainer


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]
    note_tables = ["discharge", "radiology"] if args.include_notes else None

    if args.include_notes and not args.note_root:
        raise ValueError("--include-notes requires --note-root.")

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if args.include_notes else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev,
        num_workers=args.num_workers,
    )


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

    task = MultimodalMortalityHorizonMIMIC4(
        observation_window_hours=args.observation_window_hours,
        prediction_horizon_hours=args.prediction_horizon_hours,
        include_notes=args.include_notes,
        tokenizer_model=args.tokenizer_model,
        min_age=args.min_age,
        padding=args.padding,
    )
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust window settings."
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
        description="Run E2E unified embedding + MLP/RNN protocol on MIMIC-IV."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")

    parser.add_argument("--model", type=str, choices=["mlp", "rnn"], default="rnn")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--observation-window-hours", type=int, default=24)
    parser.add_argument("--prediction-horizon-hours", type=int, default=12)
    parser.add_argument("--min-age", type=int, default=18)
    parser.add_argument("--padding", type=int, default=0)

    parser.add_argument("--include-notes", action="store_true")
    parser.add_argument("--tokenizer-model", type=str, default="bert-base-uncased")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")
