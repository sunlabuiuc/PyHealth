"""End-to-end protocol runner for unified-embedding mortality models on MIMIC-IV.

Trains and evaluates a unified-embedding model (MLP / RNN / Transformer /
BottleneckTransformer / EHRMamba / JambaEHR) on a MIMIC-IV mortality task,
then writes per-sample predictions to CSV and run metrics to JSON.

Tasks
-----
--task labs (default)
    LabsMIMIC4: 10-dim lab vectors only.

--task notes_labs (recommended for multimodal)
    NotesLabsMIMIC4: notes + 10-dim lab vectors.

(The CXR task classes land in a follow-up commit, which will add a
``notes_labs_cxr`` choice here.)

At the end of a run the runner prints and writes ``metrics.json`` with the
per-epoch train time, evaluation/inference time, and frozen-text cache
stats (hits, misses, size). That file is what the cache on/off A/B reads.

Example
-------
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /path/to/mimiciv/2.2 \\
      --task labs \\
      --model transformer \\
      --heads 4 --num-layers 2 \\
      --dev --device cpu \\
      --epochs 1 --batch-size 32 --lr 1e-3 \\
      --output-dir ./output/unified_e2e

    # Notes + labs with the text cache disabled (cache A/B, off arm):
    python examples/mortality_prediction/unified_embedding_e2e_mimic4.py \\
      --ehr-root /data/mimic-iv/2.2 --note-root /data/mimic-iv/note \\
      --task notes_labs --model transformer \\
      --no-cache-frozen-text --max-frozen-text-cache 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import (
    MLP,
    RNN,
    Transformer,
    UnifiedMultimodalEmbeddingModel,
)
from pyhealth.models.bottleneck_transformer import BottleneckTransformer
from pyhealth.models.ehrmamba import EHRMamba
from pyhealth.models.jamba_ehr import JambaEHR
from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4, NotesLabsMIMIC4
from pyhealth.trainer import Trainer, _cpu_seconds
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
        group: Optional[str] = None,
        job_type: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.group = group
        self.job_type = job_type
        self._run = None
        if self.enabled:
            import wandb

            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                config=config,
                group=group,
                job_type=job_type,
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

    if args.task == "notes_labs":
        if not args.note_root:
            raise ValueError("--task notes_labs requires --note-root.")
        note_tables = ["discharge", "radiology"]

    return MIMIC4Dataset(
        ehr_root=args.ehr_root,
        ehr_tables=ehr_tables,
        note_root=args.note_root if note_tables else None,
        note_tables=note_tables,
        cache_dir=args.cache_dir,
        dev=args.dev if args.dev else False,
        num_workers=args.num_workers,
        use_dask_distributed=args.use_dask_distributed,
    )


def _build_task(args: argparse.Namespace):
    if args.task == "notes_labs":
        return NotesLabsMIMIC4(
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


def _window_suffix(observation_window_hours: float) -> str:
    """Directory/name suffix for the observation-window arm.

    An observation-window arm is a different experiment from the full-stay
    run at the same task/model/seed; without the suffix the two share an
    output directory and a W&B run name.
    """
    return f"_w{observation_window_hours:g}" if observation_window_hours else ""


def _select_inference_loader(test_loader, val_loader, train_loader):
    """Name the split predictions come from, warning when it is not test.

    The old ``test_loader or val_loader or train_loader`` chain could report
    training metrics as test with nothing saying so. The split is now named,
    warned about, and recorded.
    """
    if test_loader is not None:
        return test_loader, "test"
    if val_loader is not None:
        warnings.warn(
            "No test split available; reporting predictions from the VALIDATION "
            "split. These are not test metrics.",
            RuntimeWarning,
            stacklevel=2,
        )
        return val_loader, "val"
    warnings.warn(
        "No test or validation split available; reporting predictions from "
        "the TRAINING split. These metrics are held-in and meaningless as a "
        "generalisation estimate.",
        RuntimeWarning,
        stacklevel=2,
    )
    return train_loader, "train"


def _build_unified_embedding(
    args: argparse.Namespace, sample_dataset: Any
) -> UnifiedMultimodalEmbeddingModel:
    return UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
        freeze_text_encoder=args.freeze_encoder,
        cache_frozen_text=args.cache_frozen_text,
        max_frozen_text_cache=args.max_frozen_text_cache,
    )


def _build_model(args: argparse.Namespace, sample_dataset: Any, unified):
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
    eval_split: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    y_true_flat = y_true.reshape(-1).tolist()
    y_prob_flat = y_prob.reshape(-1).tolist()

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "y_true",
                "y_prob",
                "y_pred_threshold_0_5",
                "eval_split",
            ],
        )
        writer.writeheader()
        for idx, prob in enumerate(y_prob_flat):
            writer.writerow(
                {
                    "patient_id": patient_ids[idx],
                    "y_true": int(y_true_flat[idx]),
                    "y_prob": float(prob),
                    "y_pred_threshold_0_5": int(float(prob) >= 0.5),
                    "eval_split": eval_split,
                }
            )


class _EpochTimedLoader:
    """Wrap a training loader to record per-epoch wall and CPU time.

    ``Trainer.train`` returns ``None`` and its epoch loop lives inside the
    Trainer, so the runner cannot time epochs without changing the Trainer.
    The training loop calls ``iter(loader)`` once per epoch (initially, then
    again each time the iterator is exhausted), so each ``__iter__`` call
    marks an epoch boundary. Everything else delegates to the wrapped
    loader, so shuffling and seeding behavior is unchanged.
    """

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self.epoch_boundaries: List[float] = []
        self.cpu_boundaries: List[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        self.epoch_boundaries.append(time.perf_counter())
        self.cpu_boundaries.append(_cpu_seconds())
        return iter(self._loader)

    def epoch_seconds(self) -> List[float]:
        """Per-epoch durations; the last epoch ends when training returns."""
        ends = self.epoch_boundaries[1:] + [time.perf_counter()]
        return [end - start for start, end in zip(self.epoch_boundaries, ends)]

    def epoch_cpu_seconds(self) -> List[Any]:
        """Per-epoch CPU seconds; ``None`` where the clock is unavailable."""
        ends = self.cpu_boundaries[1:] + [_cpu_seconds()]
        return [
            None
            if start is None or end is None
            else max(end - start, 0.0)
            for start, end in zip(self.cpu_boundaries, ends)
        ]


def run(args: argparse.Namespace, sample_dataset: Any = None) -> Path:
    """Run the end-to-end protocol and write predictions + metrics.

    Args:
        args: Parsed CLI flags (see :func:`parse_args`).
        sample_dataset: Optional prebuilt sample dataset (e.g. synthetic in
            tests). When ``None`` (the CLI path), the MIMIC-IV dataset is
            loaded from ``args`` roots and the task is applied to it.
    """
    set_seed(args.seed)

    if sample_dataset is None:
        base_dataset = _build_base_dataset(args)
        task = _build_task(args)
        sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)
        if len(sample_dataset) == 0:
            raise RuntimeError(
                "Task produced zero samples. Check roots/tables or adjust settings."
            )

    train_ds, val_ds, test_ds, split_mode = _split_dataset(
        sample_dataset, seed=args.seed
    )

    unified = _build_unified_embedding(args, sample_dataset)
    model = _build_model(args, sample_dataset, unified)

    loader_kwargs: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": (args.prefetch_factor if args.num_workers > 0 else None),
    }
    timed_train_loader = _EpochTimedLoader(
        get_dataloader(
            train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
        )
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

    # Name the split predictions come from up front: it is recorded in
    # run_config.json alongside split_mode and the split sizes.
    inference_loader, eval_split = _select_inference_loader(
        test_loader, val_loader, timed_train_loader
    )

    # The task MUST be in the name. Without it, two arms of the same comparison
    # at the same seed, for example --task labs and --task notes_labs, resolve
    # to one directory and the second run overwrites the first. Same for a
    # --dev smoke check and an observation-window arm: without their
    # suffixes they would collide with the full run.
    dev_suffix = f"_dev{args.dev}" if args.dev else ""
    window_suffix = _window_suffix(args.observation_window_hours)
    exp_name = (
        f"{args.task}_{args.model}_seed{args.seed}"
        f"{dev_suffix}{window_suffix}"
    )
    output_dir = Path(args.output_dir)

    wandb_logger = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or exp_name,
        tags=args.wandb_tags.split(",") if args.wandb_tags else [args.task, args.model],
        config=vars(args),
        # Group by arm and split by backbone so a many-cell sweep is
        # navigable instead of one flat list of runs.
        group=f"{args.task}{window_suffix}",
        job_type=args.model,
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
            "split_mode": split_mode,
            "eval_split": eval_split,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        },
    )

    epoch_train_seconds: List[float] = []
    epoch_train_cpu_seconds: List[Any] = []
    if args.epochs > 0 and len(train_ds) > 0:
        trainer.train(
            train_dataloader=timed_train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params=optimizer_params,
            weight_decay=args.weight_decay,
            max_grad_norm=effective_max_grad_norm,
            monitor="pr_auc",
            load_best_model_at_last=True,
            patience=args.patience,
            accumulation_steps=args.accumulation_steps,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
        )
        epoch_train_seconds = timed_train_loader.epoch_seconds()
        epoch_train_cpu_seconds = timed_train_loader.epoch_cpu_seconds()

    train_cpu_seconds = (
        round(sum(v for v in epoch_train_cpu_seconds if v is not None), 2)
        if any(v is not None for v in epoch_train_cpu_seconds)
        else None
    )
    train_cpu_util_pct = (
        round(100.0 * train_cpu_seconds / max(sum(epoch_train_seconds), 1e-9), 1)
        if train_cpu_seconds is not None and epoch_train_seconds
        else None
    )

    eval_seconds = 0.0
    test_scores: Dict[str, float] = {}
    if test_loader is not None:
        eval_start = time.perf_counter()
        test_scores = trainer.evaluate(test_loader)
        eval_seconds = time.perf_counter() - eval_start
        wandb_logger.log({f"test_{k}": v for k, v in test_scores.items()})

    inference_start = time.perf_counter()
    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )
    inference_seconds = time.perf_counter() - inference_start

    # metrics.json carries validation only, so without this the test numbers
    # worth reporting live nowhere on disk: only in stdout and W&B, and
    # recoverable afterwards only by re-scoring predictions by hand.
    if test_scores:
        test_path = output_dir / exp_name / "test_metrics.json"
        with open(test_path, "w") as handle:
            json.dump({"eval_split": eval_split, **test_scores}, handle, indent=2)

    output_csv = output_dir / exp_name / f"predictions_{args.model}.csv"
    _write_predictions(output_csv, patient_ids, y_true, y_prob, eval_split)

    metrics = {
        "config": vars(args),
        "split": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds) if test_ds is not None else 0,
        },
        "epoch_train_seconds": epoch_train_seconds,
        "epoch_train_cpu_seconds": epoch_train_cpu_seconds,
        "train_cpu_seconds": train_cpu_seconds,
        "train_cpu_util_pct": train_cpu_util_pct,
        "eval_seconds": eval_seconds,
        "inference_seconds": inference_seconds,
        "frozen_text_cache": unified.frozen_text_cache_stats,
        "test_scores": test_scores,
    }
    metrics_path = output_dir / exp_name / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str))

    print(f"Saved predictions to: {output_csv}")
    print(f"Saved metrics to: {metrics_path}")
    print(
        "epoch_train_seconds="
        f"{[round(t, 3) for t in epoch_train_seconds]} "
        f"train_cpu_seconds={train_cpu_seconds} "
        f"train_cpu_util_pct={train_cpu_util_pct} "
        f"eval_seconds={eval_seconds:.3f} "
        f"inference_seconds={inference_seconds:.3f} "
        f"frozen_text_cache={metrics['frozen_text_cache']}"
    )

    wandb_logger.finish()

    return output_csv


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2E unified embedding on MIMIC-IV mortality tasks."
    )
    parser.add_argument("--ehr-root", type=str, required=True)
    parser.add_argument("--note-root", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./output/unified_e2e")

    parser.add_argument(
        "--task",
        type=str,
        choices=["labs", "notes_labs"],
        default="labs",
        help=(
            "notes_labs: admission-context text + labs (needs --note-root). "
            "labs: 10-dim lab vectors only."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mlp",
            "rnn",
            "transformer",
            "bottleneck_transformer",
            "ehrmamba",
            "jambaehr",
        ],
        default="rnn",
    )

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lr",
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
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP dtype when --use-amp is set. bf16 is more stable (default).",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default 1, no accumulation).",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--no-dask-distributed",
        dest="use_dask_distributed",
        action="store_false",
        default=True,
        help=(
            "Skip the Dask distributed cluster for event caching and use "
            "the local scheduler instead."
        ),
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Copy CPU tensors into page-locked memory in the loaders.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep loader workers alive between epochs (needs --num-workers > 0).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Batches each loader worker prefetches (needs --num-workers > 0).",
    )
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
        "--observation-window-hours",
        type=float,
        default=0,
        help=(
            "Hours from each admission to collect labs/radiology notes. "
            "Default 0 means no cap: full stay through discharge."
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
    parser.add_argument(
        "--cache-frozen-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse frozen text-encoder [CLS] vectors across forwards "
            "(default on). Pass --no-cache-frozen-text for the cache-off arm "
            "of the A/B timing run."
        ),
    )
    parser.add_argument(
        "--max-frozen-text-cache",
        type=int,
        default=1_000_000,
        help="Maximum cached frozen [CLS] vectors; rows past the cap are re-encoded.",
    )
    parser.add_argument("--rnn-type", type=str, default="GRU")
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)

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
        type=int,
        default=16,
        help="SSM state size for EHRMamba and JambaEHR blocks.",
    )
    parser.add_argument(
        "--mamba-conv-kernel",
        type=int,
        default=4,
        help="Causal conv kernel size for EHRMamba and JambaEHR blocks.",
    )
    parser.add_argument(
        "--jamba-transformer-layers",
        type=int,
        default=2,
        help="Number of Transformer (attention) layers in JambaEHR.",
    )
    parser.add_argument(
        "--jamba-mamba-layers",
        type=int,
        default=2,
        help="Number of Mamba (SSM) layers in JambaEHR.",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> Path:
    """CLI entry point (``argv`` injectable for tests)."""
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    output_csv_path = main()
    print(f"Saved predictions to: {output_csv_path}")
