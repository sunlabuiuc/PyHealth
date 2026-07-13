"""End-to-end protocol runner for Unified Embedding on MIMIC-IV.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on a MIMIC-IV mortality task,
then writes per-sample predictions to CSV.

Tasks
-----
--task stagenet (default)
    MortalityPredictionStageNetMIMIC4: ICD codes + 10-dim lab vectors,
    patient-level samples aggregated across all admissions.

--task icd_labs
    ICDLabsMIMIC4: ICD codes + 10-dim lab vectors via the unified
    multimodal pipeline.  No notes required.

--task clinical_notes_icd_labs
    ClinicalNotesICDLabsMIMIC4: discharge/radiology notes + ICD + labs.
    Requires --note-root.  Legacy; ICD codes are discharge-coded (leakage).

--task notes_labs (recommended for multimodal)
    NotesLabsMIMIC4: admission-context note sections + labs, no ICD codes.
    Extracts Chief Complaint, HPI, PMH, Medications on Admission from the
    discharge note — text available at admission time, ~90%+ coverage.
    Requires --note-root.
    Add --freeze-encoder to freeze Bio_ClinicalBERT and train only the
    backbone; cuts BERT VRAM by ~50%, useful on smaller GPUs (≤24 GB).
    Add --icd-codes to include discharge-coded ICD codes (ablation only).

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

    # EHRMamba on full dataset (no --dev):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task clinical_notes_icd_labs --model ehrmamba \\
      --embedding-dim 128 --num-layers 2 --seed 42

    # JambaEHR:
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task clinical_notes_icd_labs --model jambaehr \\
      --embedding-dim 128 --jamba-transformer-layers 2 --jamba-mamba-layers 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    sample_balanced,
    sample_oversample,
    sample_weighted,
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
    ICDLabsMIMIC4,
    NotesLabsMIMIC4,
)
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]
    note_tables = None

    if args.task == "clinical_notes_icd_labs":
        if not args.note_root:
            raise ValueError("--task clinical_notes_icd_labs requires --note-root.")
        note_tables = ["discharge", "radiology"]

    if args.task == "icd_labs":
        ehr_tables = ["diagnoses_icd", "procedures_icd", "labevents"]

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev if args.dev else False,
        num_workers=args.num_workers,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "stagenet":
        return MortalityPredictionStageNetMIMIC4()
    if args.task == "icd_labs":
        return ICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task == "clinical_notes_icd_labs":
        return ClinicalNotesICDLabsMIMIC4(window_hours=args.observation_window_hours)
    if args.task == "notes_labs":
        return NotesLabsMIMIC4(
            window_hours=args.observation_window_hours,
            include_icd=args.icd_codes,
            include_vitals=args.include_vitals,
        )
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
        freeze_text_encoder=args.freeze_encoder,
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


def _compute_pos_weight(train_ds, label_key: str = "mortality") -> float:
    """Count pos/neg in train_ds and return n_neg/n_pos for BCE pos_weight."""
    n_pos = n_neg = 0
    for i in range(len(train_ds)):
        sample = train_ds[i]
        label = sample.get(label_key, 0)
        if hasattr(label, "__iter__"):
            label = next(iter(label))
        if float(label) > 0.5:
            n_pos += 1
        else:
            n_neg += 1
    if n_pos == 0:
        return 1.0
    # Cap at 10: n_neg/n_pos ≈ 37 on MIMIC-IV mortality is too extreme with
    # typical LRs and causes training oscillation. 10 still strongly corrects
    # for imbalance while keeping gradient magnitudes tractable.
    return min(10.0, n_neg / n_pos)


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed)

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust settings."
        )

    train_ds, val_ds, test_ds = _split_dataset(sample_dataset, seed=args.seed)

    label_key = list(sample_dataset.output_schema.keys())[0]

    # Resolve effective sampling strategy.
    # --balanced-sampling / --balanced-ratio are legacy aliases for undersample.
    strategy = args.sampling_strategy
    if args.balanced_sampling and strategy == "none":
        strategy = "undersample"

    if strategy == "undersample":
        ratio = args.balanced_ratio
        print(f"[sampling] Undersampling negatives -> pos:neg 1:{ratio}")
        train_ds = sample_balanced(train_ds, ratio=ratio, seed=args.seed, label_key=label_key)
        print(f"[sampling] Training size after undersample: {len(train_ds)}")

    elif strategy == "oversample":
        ratio = args.balanced_ratio
        print(f"[sampling] Oversampling positives -> pos:neg 1:{ratio}")
        train_ds = sample_oversample(train_ds, ratio=ratio, seed=args.seed, label_key=label_key)
        print(f"[sampling] Training size after oversample: {len(train_ds)}")

    elif strategy == "weighted":
        print("[sampling] Weighted resampling (class-proportional, with replacement)")
        train_ds = sample_weighted(train_ds, seed=args.seed, label_key=label_key)
        print(f"[sampling] Training size after weighted resample: {len(train_ds)}")

    model = _build_model(args, sample_dataset)

    # Apply class-imbalance correction via BCE pos_weight.
    # pos_weight = n_neg / n_pos so the rare positive class gets proportionally
    # higher gradient signal, preventing all-negative collapse (F1=0).
    if args.pos_weight is not None:
        pw_value = args.pos_weight
    else:
        print(f"[pos_weight] Computing class balance from {len(train_ds)} training samples...")
        pw_value = _compute_pos_weight(train_ds, label_key=label_key)
    print(f"[pos_weight] Using pos_weight={pw_value:.2f} for binary BCE loss.")
    model._pos_weight = torch.tensor([pw_value], dtype=torch.float32)

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

    # Experiment name encodes model + seed for easy log separation
    exp_name = f"{args.model}_seed{args.seed}"
    output_dir = Path(args.output_dir)

    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1", "f1_opt", "accuracy"],
        device=args.device,
        enable_logging=True,
        output_path=str(output_dir),
        exp_name=exp_name,
    )

    # BottleneckTransformer is more fragile on full MIMIC-IV with no warmup.
    # Use safer defaults unless explicitly overridden from CLI.
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
        # All non-BT models: 1e-4 (was 1e-3). With pos_weight correction,
        # effective gradient magnitude for positives is ~10x higher, so a
        # smaller LR is needed to avoid training oscillation.
        if effective_lr is None:
            effective_lr = 1e-4
        # Universal grad clipping: prevents runaway updates from the weighted
        # positive-class loss (pos_weight ≈ 10 scales positive gradients 10x).
        if effective_max_grad_norm is None:
            effective_max_grad_norm = 1.0
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
            patience=args.patience,
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
        description="Run E2E unified embedding on MIMIC-IV with any of six sequence heads."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")

    parser.add_argument(
        "--task",
        type=str,
        choices=["stagenet", "icd_labs", "clinical_notes_icd_labs"],
        default="stagenet",
        help=(
            "notes_labs: admission-context text (CC/HPI/PMH/MedsOnAdm) + labs. "
            "No ICD codes (discharge-coded = leakage). Recommended for multimodal."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rnn", "transformer", "bottleneck_transformer",
                 "ehrmamba", "jambaehr"],
        default="rnn",
    )

    # Shared embedding / training
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Learning rate. Default is 1e-4 for all models. "
            "(Previously 1e-3 for mlp/rnn/transformer/ehrmamba/jambaehr — "
            "reduced after pos_weight correction caused oscillation at 1e-3.)"
        ),
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=None,
        help=(
            "Adam epsilon. Default is model-specific: 1e-8 for non-BT models, "
            "1e-6 for bottleneck_transformer."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--dev",
        nargs="?",
        type=int,
        const=1000,
        default=0,
        help=(
            "Dev mode: limit dataset to N patients for fast iteration. "
            "--dev (no value) defaults to 1000 patients. "
            "--dev 5000 limits to 5000. Omit for full dataset."
        ),
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help=(
            "BCE pos_weight for the positive class (float). "
            "Default: auto-computed as n_neg/n_pos from training split. "
            "Set to 1.0 to disable class-imbalance correction."
        ),
    )

    # Task-specific
    parser.add_argument("--observation-window-hours", type=int, default=24)
    parser.add_argument(
        "--icd-codes",
        action="store_true",
        default=False,
        help=(
            "Include discharge-coded ICD codes in notes_labs task. "
            "Default: off (ICD codes are coded at discharge and constitute "
            "data leakage for in-hospital mortality prediction). "
            "Enable only for ablation / legacy comparison experiments."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help=(
            "Freeze pretrained BERT text encoder weights and train only the "
            "downstream backbone (MLP/RNN/Transformer head + projection layer). "
            "Reduces VRAM by ~50% for the text branch; useful when GPU memory "
            "is limited or for faster iteration on backbone architectures."
        ),
    )
    parser.add_argument(
        "--include-vitals",
        action="store_true",
        default=False,
        help=(
            "Include ICU vital signs (HeartRate, SysBP, DiasBP, MeanBP, "
            "RespRate, SpO2, Temperature) from chartevents as an additional "
            "modality alongside labs and notes. Adds chartevents to EHR tables."
        ),
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        default=False,
        help=(
            "Undersample the majority (negative) class in training to improve "
            "PR-AUC on imbalanced datasets. Uses sample_balanced() to create a "
            "1:--balanced-ratio pos:neg training set."
        ),
    )
    parser.add_argument(
        "--balanced-ratio",
        type=float,
        default=1.0,
        help=(
            "Negatives per positive in the balanced training set. "
            "Default: 1.0 (equal pos/neg). Used with undersample and oversample strategies."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "undersample", "oversample", "weighted"],
        help=(
            "Training-set class balance strategy. "
            "'none': no resampling (default). "
            "'undersample': drop majority-class (neg) samples via sample_balanced(). "
            "'oversample': duplicate minority-class (pos) samples via sample_oversample(). "
            "'weighted': class-proportional resampling w/ replacement via sample_weighted(). "
            "--balanced-sampling is a legacy alias for 'undersample'."
        ),
    )

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

    # Training stability
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help=(
            "Gradient clipping max norm. Default is model-specific: None for "
            "non-BT models, 0.5 for bottleneck_transformer."
        ),
    )

    # Mamba / JambaEHR-specific
    parser.add_argument("--mamba-state-size", type=int, default=16,
                        help="SSM state size for EHRMamba and JambaEHR blocks.")
    parser.add_argument("--mamba-conv-kernel", type=int, default=4,
                        help="Causal conv kernel size for EHRMamba and JambaEHR blocks.")
    parser.add_argument("--jamba-transformer-layers", type=int, default=2,
                        help="Number of Transformer (attention) layers in JambaEHR.")
    parser.add_argument("--jamba-mamba-layers", type=int, default=6,
                        help="Number of Mamba (SSM) layers in JambaEHR.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")
