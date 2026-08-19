"""End-to-end protocol runner for Unified Embedding on MIMIC-IV.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on a MIMIC-IV mortality task,
then writes per-sample predictions to CSV.

Tasks
-----
--task labs (default)
    LabsMIMIC4: 10-dim lab vectors only.

--task notes_labs (recommended for multimodal)
    NotesLabsMIMIC4: notes + 10-dim lab vectors.

--task notes_labs_cxr
    NotesLabsCXRMIMIC4: notes + labs + chest-xray.

Example
-------
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --task labs \\
      --model transformer \\
      --heads 4 --num-layers 2 \\
      --dev --device cpu \\
      --epochs 10 --batch-size 32 --lr 1e-3 \\
      --output-dir ./output/unified_e2e

    # EHRMamba on full dataset (no --dev):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task notes_labs --model ehrmamba \\
      --embedding-dim 128 --num-layers 2 --seed 42

    # JambaEHR:
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task notes_labs --model jambaehr \\
      --embedding-dim 128 --jamba-transformer-layers 2 --jamba-mamba-layers 2
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

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
from pyhealth.processors import fit_lab_standardizer, lab_standardizer_fit_scope
from pyhealth.tasks.multimodal_mimic4 import (
    LabsMIMIC4,
    NotesLabsCXRMIMIC4,
    NotesLabsMIMIC4,
)
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed, write_run_config


class WandbLogger:

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: Optional[str],
        run_name: str,
        tags: list[str],
        config: Dict[str, Any],
    ) -> None:
        self.enabled = enabled
        self._run = None
        if self.enabled:
            import wandb

            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                config=config,
            )

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            self._run.log(data, step=step)

    def finish(self) -> None:
        if self.enabled:
            self._run.finish()


def _build_base_dataset(args: argparse.Namespace) -> MIMIC4Dataset:
    ehr_tables = ["labevents"]
    note_tables = None
    cxr_kwargs = {}

    if args.task == "notes_labs":
        if not args.note_root:
            raise ValueError("--task notes_labs requires --note-root.")
        note_tables = ["discharge", "radiology"]

    if args.task == "notes_labs_cxr":
        if not args.note_root:
            raise ValueError("--task notes_labs_cxr requires --note-root.")
        if not args.cxr_root:
            raise ValueError("--task notes_labs_cxr requires --cxr-root.")
        note_tables = ["discharge", "radiology"]
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
        dev=args.dev if args.dev else False,
        num_workers=args.num_workers,
        **cxr_kwargs,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "notes_labs":
        return NotesLabsMIMIC4(
            window_hours=args.observation_window_hours,
        )
    if args.task == "notes_labs_cxr":
        return NotesLabsCXRMIMIC4(
            window_hours=args.observation_window_hours,
        )
    if args.task == "labs":
        return LabsMIMIC4(window_hours=args.observation_window_hours)
    raise ValueError(f"Unknown task: {args.task}")


def _split_dataset(dataset: Any, seed: int) -> Tuple[Any, Any, Any, str]:
    """Split by patient, falling back to by-sample only if that yields nothing.

    The fallback is leaky: a patient with several admissions can then land in
    both train and test, which inflates the metrics. It only triggers on tiny
    cohorts, but it must not trigger silently, so the mode is returned and
    recorded alongside the run's results.
    """
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=seed)
    if len(train_ds) == 0 or len(test_ds) == 0:
        warnings.warn(
            "split_by_patient produced an empty split, falling back to "
            "split_by_sample. The same patient may now appear in train and "
            "test, so these metrics are optimistic and not comparable to "
            "patient-split runs.",
            RuntimeWarning,
            stacklevel=2,
        )
        train_ds, val_ds, test_ds = split_by_sample(dataset, [0.8, 0.1, 0.1], seed=seed)
        return train_ds, val_ds, test_ds, "by_sample_fallback_leaky"
    return train_ds, val_ds, test_ds, "by_patient"


def _build_model(
    args: argparse.Namespace,
    sample_dataset: Any,
    numeric_standardizers: dict[str, Any] | None = None,
):
    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
        freeze_text_encoder=args.freeze_encoder,
        numeric_standardizers=numeric_standardizers,
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
    set_seed(args.seed)

    base_dataset = _build_base_dataset(args)
    task = _build_task(args)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or adjust settings."
        )

    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_ds, val_ds, test_ds, split_mode = _split_dataset(
        sample_dataset, seed=split_seed
    )

    numeric_standardizers: dict[str, Any] = {}
    if "labs" in sample_dataset.input_processors and not args.no_lab_standardization:
        if "labs_mask" not in sample_dataset.input_processors:
            raise RuntimeError(
                "Lab standardisation requires the labs_mask observation field."
            )
        lab_standardizer = fit_lab_standardizer(
            train_ds,
            value_field="labs",
            fit_scope=lab_standardizer_fit_scope(train_ds, value_field="labs"),
        )
        numeric_standardizers["labs"] = lab_standardizer
        print(
            "[lab-standardization] fitted on train split only: "
            f"counts={lab_standardizer.observed_count.tolist()} "
            f"mean={lab_standardizer.mean.tolist()} std={lab_standardizer.std.tolist()}"
        )
    elif "labs" in sample_dataset.input_processors:
        print("[lab-standardization] disabled; reproducing raw-lab baseline.")

    model = _build_model(args, sample_dataset, numeric_standardizers)

    loader_kwargs = {
        "num_workers": args.loader_num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": (
            args.prefetch_factor if args.loader_num_workers > 0 else None
        ),
    }
    train_loader = get_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = (
        get_dataloader(
            val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
        )
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        get_dataloader(
            test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
        )
        if len(test_ds) > 0
        else None
    )

    if test_loader is not None:
        inference_loader, eval_split = test_loader, "test"
    elif val_loader is not None:
        inference_loader, eval_split = val_loader, "val"
        warnings.warn(
            "No test split available; reporting predictions from the VALIDATION "
            "split. These are not test metrics.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        inference_loader, eval_split = train_loader, "train"
        warnings.warn(
            "No test or validation split available; reporting predictions from "
            "the TRAINING split. These metrics are held-in and meaningless as a "
            "generalisation estimate.",
            RuntimeWarning,
            stacklevel=2,
        )

    # The task MUST be in the name. Without it, two arms of the same comparison
    # at the same seed, for example --task labs and --task notes_labs, resolve
    # to one directory and the second run overwrites the first.
    exp_name = f"{args.task}_{args.model}_seed{args.seed}"
    output_dir = Path(args.output_dir)

    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or exp_name,
        tags=args.wandb_tags.split(",") if args.wandb_tags else [args.task, args.model],
        config=vars(args),
    )

    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1", "accuracy"],
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
        if effective_lr is None:
            effective_lr = 1e-4
        if effective_max_grad_norm is None:
            effective_max_grad_norm = 1.0
        if args.adam_eps is not None:
            optimizer_params["eps"] = args.adam_eps

    optimizer_params["lr"] = effective_lr

    write_run_config(
        str(output_dir / exp_name),
        {
            **vars(args),
            "resolved_lr": effective_lr,
            "resolved_max_grad_norm": effective_max_grad_norm,
            "split_mode": split_mode,
            "resolved_split_seed": split_seed,
            "eval_split": eval_split,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "lab_standardization": bool(numeric_standardizers),
        },
    )

    if args.epochs > 0 and len(train_ds) > 0:
        metrics_history = trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=effective_max_grad_norm,
            monitor="pr_auc",
            load_best_model_at_last=True,
            patience=args.patience,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
        )
        for epoch_record in metrics_history:
            wandb_logger.log(epoch_record, step=epoch_record["epoch"])

    if wandb_logger.enabled and test_loader is not None:
        test_scores = trainer.evaluate(test_loader)
        wandb_logger.log({f"test_{k}": v for k, v in test_scores.items()})

    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    output_csv = output_dir / exp_name / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob)

    wandb_logger.finish()

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV with any of six sequence heads."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cxr-root", type=str, default=None)
    parser.add_argument("--cxr-variant", type=str, default="sunlab", choices=["default", "sunlab"])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")

    parser.add_argument(
        "--task",
        type=str,
        choices=["labs", "notes_labs", "notes_labs_cxr"],
        default="labs",
        help=(
            "notes_labs: admission-context text (CC/HPI/PMH/MedsOnAdm) + labs. "
            "Recommended for multimodal. "
            "notes_labs_cxr: notes_labs plus in-window chest X-rays; requires "
            "--note-root and --cxr-root."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "rnn", "transformer", "bottleneck_transformer",
                 "ehrmamba", "jambaehr"],
        default="rnn",
    )

    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        "--learning_rate",
        dest="lr",
        type=float,
        default=None,
        help="Learning rate. Default is 1e-4 for all models.",
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
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable automatic mixed precision training to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--amp-dtype",
        "--amp_dtype",
        dest="amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype when --use-amp is set. bf16 is more stable (default).",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--loader-num-workers",
        type=int,
        default=0,
        help="Worker processes for runtime batch loading/collation.",
    )
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--persistent-workers", action="store_true", default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        "--split_seed",
        dest="split_seed",
        type=int,
        default=None,
        help=(
            "Patient-split RNG seed. Defaults to --seed. Set separately when "
            "a tiny --dev cohort puts no positives in val/test."
        ),
    )
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
        "--patients",
        type=int,
        default=None,
        help=(
            "Not a supported flag. Do not pass this. Use --dev N for a "
            "patient-limited smoke, and omit both for the full table."
        ),
    )
    parser.add_argument(
        "--observation-window-hours",
        type=int,
        default=None,
        help=(
            "If set, collect labs/CXR/radiology only this many hours from each "
            "admission. Default: full stay (through discharge)."
        ),
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        default=False,
        help=(
            "Freeze pretrained BERT text encoder weights and train only the "
            "downstream backbone (RNN/Transformer head + projection layer). "
        ),
    )
    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--num-layers",
        "--num_layers",
        dest="num_layers",
        type=int,
        default=2,
    )

    parser.add_argument("--bottlenecks-n", type=int, default=4)
    parser.add_argument("--fusion-startidx", type=int, default=1)

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help=(
            "Gradient clipping max norm. Default is model-specific: None for "
            "non-BT models, 0.5 for bottleneck_transformer."
        ),
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log training/eval metrics to Weights & Biases.",
    )
    parser.add_argument("--wandb-project", type=str, default="pyhealth-mortality")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Defaults to '{task}_{model}_seed{seed}' if unset.",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated wandb tags, e.g. 'labs,rnn'. Defaults to '{task},{model}' if unset.",
    )

    parser.add_argument(
        "--mamba-state-size",
        "--mamba_state_size",
        dest="mamba_state_size",
        type=int,
        default=16,
        help="SSM state size for EHRMamba and JambaEHR blocks.",
    )
    parser.add_argument(
        "--mamba-conv-kernel",
        "--mamba_conv_kernel",
        "--conv-kernel",
        "--conv_kernel",
        dest="mamba_conv_kernel",
        type=int,
        default=4,
        help="Causal conv kernel size for EHRMamba and JambaEHR blocks.",
    )
    parser.add_argument(
        "--jamba-transformer-layers",
        "--jamba_transformer_layers",
        dest="jamba_transformer_layers",
        type=int,
        default=2,
        help="Number of Transformer (attention) layers in JambaEHR.",
    )
    parser.add_argument(
        "--jamba-mamba-layers",
        "--jamba_mamba_layers",
        dest="jamba_mamba_layers",
        type=int,
        default=6,
        help="Number of Mamba (SSM) layers in JambaEHR. Library default is 6; "
             "pass 2 for a depth-matched comparison against --num-layers 2.",
    )
    parser.add_argument(
        "--no-lab-standardization",
        action="store_true",
        default=False,
        help="Disable train-split lab z-scoring (raw-lab ablation).",
    )

    args = parser.parse_args()
    if args.patients is not None:
        parser.error(
            "--patients is not a supported flag and is not a 5-patient smoke. "
            "Omit it for the full table; use --dev N for a patient-limited run."
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    output_csv_path = run(args)
    print(f"Saved predictions to: {output_csv_path}")