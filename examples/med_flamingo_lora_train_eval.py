"""Train and evaluate Med-Flamingo LoRA adapters for medical VQA tasks.

Example:
    python examples/med_flamingo_lora_train_eval.py \
        --dataset vqa_rad \
        --root /path/to/vqa_rad \
        --llama_path /path/to/llama \
        --output_dir ./runs/med_flamingo_lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from pyhealth.datasets import (
    PathVQADataset,
    VQARADDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import MedFlamingo
from pyhealth.tasks import GenerativeMedicalVQA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", choices=["vqa_rad", "path_vqa"], default="vqa_rad")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--llama_path", type=str, required=True)

    parser.add_argument("--annotation_path", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--clean_split_path", type=str, default=None)
    parser.add_argument("--enable_dedup", action="store_true")
    parser.add_argument(
        "--dedup_method",
        choices=["auto", "faiss", "phash", "none"],
        default="auto",
    )

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")

    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_every_n_steps", type=int, default=0)

    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--metrics",
        type=str,
        default="exact_match",
        help="Comma-separated metrics: exact_match,bertscore_f1",
    )

    parser.add_argument("--quantization", choices=["auto", "8bit", "fp16"], default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf_repo_id", type=str, default="med-flamingo/med-flamingo")
    parser.add_argument("--checkpoint_filename", type=str, default="model.pt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none")

    parser.add_argument("--resume_adapter", type=str, default=None)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="run_summary.json")

    return parser.parse_args()


def build_dataset(args: argparse.Namespace):
    common_kwargs = {
        "root": args.root,
        "annotation_path": args.annotation_path,
        "images_dir": args.images_dir,
        "clean_split_path": args.clean_split_path,
        "enable_dedup": args.enable_dedup,
        "dedup_method": args.dedup_method,
    }
    if args.dataset == "vqa_rad":
        return VQARADDataset(**common_kwargs)
    return PathVQADataset(**common_kwargs)


def filter_samples_by_split(sample_dataset, split: str) -> List[Dict]:
    normalized = split.strip().lower()
    return [
        sample
        for sample in sample_dataset
        if str(sample.get("split", "")).strip().lower() == normalized
    ]


def maybe_build_sample_dataset(samples: List[Dict], task: GenerativeMedicalVQA, name: str):
    if not samples:
        return None
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=name,
        task_name=task.task_name,
    )


def main() -> None:
    args = parse_args()

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]

    base_dataset = build_dataset(args)
    task = GenerativeMedicalVQA()
    sample_dataset = base_dataset.set_task(task=task)

    train_samples = filter_samples_by_split(sample_dataset, args.train_split)
    val_samples = filter_samples_by_split(sample_dataset, args.val_split)
    test_samples = filter_samples_by_split(sample_dataset, args.test_split)

    if not train_samples:
        raise ValueError(f"No training samples found for split: {args.train_split}")

    train_dataset = maybe_build_sample_dataset(train_samples, task, f"{args.dataset}_train")
    val_dataset = maybe_build_sample_dataset(val_samples, task, f"{args.dataset}_val")
    test_dataset = maybe_build_sample_dataset(test_samples, task, f"{args.dataset}_test")

    assert train_dataset is not None
    train_loader = get_dataloader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = (
        get_dataloader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )
    test_loader = (
        get_dataloader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
        if test_dataset is not None
        else None
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MedFlamingo(
        dataset=train_dataset,
        llama_path=args.llama_path,
        hf_repo_id=args.hf_repo_id,
        checkpoint_filename=args.checkpoint_filename,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        num_shots=args.num_shots,
        sampling_strategy="random",
        seed=args.seed,
        device=args.device,
        enable_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
    )

    if args.resume_adapter:
        model.load_lora_adapter(args.resume_adapter, is_trainable=True)

    fit_summary = model.fit_lora(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        eval_every_n_steps=args.eval_every_n_steps,
        metrics=metrics,
        output_dir=str(output_dir),
    )

    eval_result = None
    if not args.skip_eval and test_loader is not None:
        best_adapter = output_dir / "best_adapter"
        if best_adapter.exists():
            model.load_lora_adapter(str(best_adapter), is_trainable=False)
        model.set_support_pool(train_samples)
        eval_result = model.predict_generation(test_loader, metrics=metrics)

    payload = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "fit_summary": fit_summary,
        "eval_result": eval_result,
        "artifacts_dir": str(output_dir.resolve()),
    }

    output_json = output_dir / args.output_json
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved run summary to {output_json}")

    train_dataset.close()
    if val_dataset is not None:
        val_dataset.close()
    if test_dataset is not None:
        test_dataset.close()
    sample_dataset.close()


if __name__ == "__main__":
    main()
