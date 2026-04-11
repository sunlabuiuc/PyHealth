"""Unified multimodal embedding + EHRMamba runner for MIMIC-IV + CXR.

This script runs EHR + notes + X-ray (metadata/negbio) with the
ClinicalNotesICDLabsCXRMIMIC4 task.

Default roots are set to shared PhysioNet mounts:
- ehr_root:  /shared/rsaas/physionet.org/files/mimiciv/2.2
- note_root: /shared/rsaas/physionet.org/files/mimic-note
- cxr_root:  /shared/rsaas/physionet.org/files/MIMIC-CXR

Quick start:
    python examples/mortality_prediction/
    multimodal_embedding_mamba_mimic4_cxr.py \
      --quick-test

Smoke test (single forward + inference, no train):
    python examples/mortality_prediction/
    multimodal_embedding_mamba_mimic4_cxr.py \
      --smoke-forward
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Tuple

import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import EHRMamba, UnifiedMultimodalEmbeddingModel
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesICDLabsCXRMIMIC4
from pyhealth.trainer import Trainer


def _split_dataset(dataset: Any, seed: int) -> Tuple[Any, Any, Any]:
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.8, 0.1, 0.1], seed=seed)
    if len(train_ds) == 0 or len(test_ds) == 0:
        train_ds, val_ds, test_ds = split_by_sample(dataset, [0.8, 0.1, 0.1], seed=seed)
    return train_ds, val_ds, test_ds


def _build_run_output_path(args: argparse.Namespace) -> str:
    device_tag = args.device.replace(":", "")
    lr_tag = f"{args.lr:g}".replace(".", "p")
    dropout_tag = f"{args.dropout:g}".replace(".", "p")
    run_tag = (
        f"cxr{args.cxr_variant}"
        f"_emb{args.embedding_dim}"
        f"_layers{args.num_layers}"
        f"_state{args.state_size}"
        f"_conv{args.conv_kernel}"
        f"_drop{dropout_tag}"
        f"_win{args.observation_window_hours}"
        f"_ep{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{lr_tag}"
        f"_{device_tag}"
        f"_nw{args.num_workers}"
        f"_seed{args.seed}"
        f"_dev{int(args.dev)}"
        f"_quick{int(args.quick_test)}"
        f"_smoke{int(args.smoke_forward)}"
    )
    return os.path.join(
        os.getcwd(), "output", "multimodal_embedding_mamba_mimic4_cxr", run_tag
    )


def run(args: argparse.Namespace) -> Tuple[int, int]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    total_start = time.perf_counter()

    cuda_device_index = None
    if args.device.startswith("cuda"):
        device_index = torch.device(args.device).index
        cuda_device_index = 0 if device_index is None else device_index

    print("Using dataset roots:")
    print(f"  ehr_root:     {args.ehr_root}")
    print(f"  note_root:    {args.note_root}")
    print(f"  cxr_root:     {args.cxr_root}")
    print(f"  cxr_variant:  {args.cxr_variant}")
    print(f"  cache_dir:    {args.cache_dir}")
    print(f"  num_workers:  {args.num_workers}")

    base_dataset = MIMIC4Dataset(
        ehr_root=args.ehr_root,
        note_root=args.note_root,
        cxr_root=args.cxr_root,
        cxr_variant=args.cxr_variant,
        ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        note_tables=["discharge", "radiology"],
        cxr_tables=["metadata", "negbio", "chexpert", "split"],
        cache_dir=args.cache_dir,
        dev=args.dev,
        num_workers=args.num_workers,
    )

    task = ClinicalNotesICDLabsCXRMIMIC4(
        window_hours=args.observation_window_hours,
    )
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "Task produced zero samples. Check roots/tables or use "
            "--quick-test first."
        )

    print(f"Task sample count: {len(sample_dataset)}")
    print("Input processor schemas:")
    for key in sample_dataset.input_schema.keys():
        processor = sample_dataset.input_processors.get(key)
        if processor is None:
            print(f"  - {key}: <no processor>")
            continue
        print(f"  - {key}: {type(processor).__name__}, " f"schema={processor.schema()}")

    train_ds, val_ds, test_ds = _split_dataset(sample_dataset, seed=args.seed)

    unified = UnifiedMultimodalEmbeddingModel(
        processors=sample_dataset.input_processors,
        embedding_dim=args.embedding_dim,
    )
    model = EHRMamba(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        state_size=args.state_size,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
        unified_embedding=unified,
    )
    print(f"EHRMamba unified mode: {model._use_unified}")

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

    print(
        "Split sizes: " f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    debug_batch = next(iter(train_loader))
    print("Batch field diagnostics (train batch 0):")
    for key in sample_dataset.input_schema.keys():
        processor = sample_dataset.input_processors.get(key)
        feature = debug_batch.get(key)
        schema = processor.schema() if processor is not None else ()
        print(f"  - {key}: type={type(feature).__name__}, schema={schema}")

        if isinstance(feature, tuple):
            for i, elem in enumerate(feature):
                shape = getattr(elem, "shape", None)
                print(f"      tuple[{i}] type={type(elem).__name__} " f"shape={shape}")

        if processor is not None and isinstance(feature, tuple):
            for field_name in ("value", "time", "mask"):
                if field_name in schema:
                    idx = schema.index(field_name)
                    if idx < len(feature):
                        selected = feature[idx]
                        shape = getattr(selected, "shape", None)
                        print(
                            f"      schema['{field_name}'] -> tuple[{idx}] "
                            f"type={type(selected).__name__} shape={shape}"
                        )

    trainer = Trainer(
        model=model,
        metrics=["accuracy"],
        device=args.device,
        enable_logging=True,
        output_path=_build_run_output_path(args),
    )

    if not args.smoke_forward and args.epochs > 0 and len(train_ds) > 0:
        if cuda_device_index is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device_index)
            torch.cuda.synchronize(cuda_device_index)

        train_start = time.perf_counter()
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params={"lr": args.lr},
            monitor=None,
            load_best_model_at_last=False,
        )
        if cuda_device_index is not None:
            torch.cuda.synchronize(cuda_device_index)
            peak_train_bytes = torch.cuda.max_memory_allocated(cuda_device_index)
            peak_train_vram_mb = peak_train_bytes / (1024**2)
        else:
            peak_train_vram_mb = None
        train_runtime_sec = time.perf_counter() - train_start
    else:
        peak_train_vram_mb = None
        train_runtime_sec = None

    inference_loader = test_loader or val_loader or train_loader
    y_true, y_prob, _, patient_ids = trainer.inference(
        inference_loader, return_patient_ids=True
    )

    if cuda_device_index is not None:
        torch.cuda.synchronize(cuda_device_index)

    total_runtime_sec = time.perf_counter() - total_start
    print("Benchmark summary:")
    print(f"  total_runtime_sec: {total_runtime_sec:.2f}")
    if train_runtime_sec is None:
        print("  training_runtime_sec: N/A (training skipped)")
        print("  peak_train_vram_mb: N/A (training skipped)")
    else:
        print(f"  training_runtime_sec: {train_runtime_sec:.2f}")
        if peak_train_vram_mb is None:
            print("  peak_train_vram_mb: N/A (non-CUDA device)")
        else:
            print(f"  peak_train_vram_mb: {peak_train_vram_mb:.2f}")

    return len(patient_ids), y_true.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified multimodal embedding + EHRMamba on "
        "MIMIC-IV mortality with CXR."
    )
    parser.add_argument(
        "--ehr-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/mimiciv/2.2",
    )
    parser.add_argument(
        "--note-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/mimic-note",
    )
    parser.add_argument(
        "--cxr-root",
        type=str,
        default="/shared/rsaas/physionet.org/files/MIMIC-CXR",
    )
    parser.add_argument(
        "--cxr-variant",
        type=str,
        default="sunlab",
        choices=["default", "sunlab"],
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/shared/eng/pyhealth",
    )

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--observation-window-hours", type=int, default=24)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--smoke-forward", action="store_true")
    args = parser.parse_args()

    if args.quick_test:
        args.dev = True
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)

    return args


if __name__ == "__main__":
    cli_args = parse_args()
    num_patient_ids, num_rows = run(cli_args)
    print(f"Inference completed (patient_ids={num_patient_ids}, " f"rows={num_rows}).")
