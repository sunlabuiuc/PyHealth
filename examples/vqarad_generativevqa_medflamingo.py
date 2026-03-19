"""Run Med-Flamingo generative VQA with few-shot ablations in PyHealth.

Example:
    python examples/vqarad_generativevqa_medflamingo.py \
        --dataset vqa_rad \
        --root /path/to/vqa_rad \
        --llama_path /path/to/llama \
        --output_json ./medflamingo_vqa_results.json
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
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--llama_path", type=str, required=True, help="Local path to LLaMA")

    parser.add_argument("--annotation_path", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--clean_split_path", type=str, default=None)
    parser.add_argument("--enable_dedup", action="store_true")
    parser.add_argument(
        "--dedup_method",
        choices=["auto", "faiss", "phash", "none"],
        default="auto",
    )

    parser.add_argument("--support_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--shots", type=str, default="0,1,3,5")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--quantization",
        choices=["auto", "8bit", "fp16"],
        default="auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hf_repo_id", type=str, default="med-flamingo/med-flamingo")
    parser.add_argument("--checkpoint_filename", type=str, default="model.pt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Optional LoRA adapter directory produced by MedFlamingo.save_lora_adapter",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="exact_match",
        help="Comma-separated metrics: exact_match,bertscore_f1",
    )
    parser.add_argument("--output_json", type=str, default="medflamingo_vqa_results.json")
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
    samples: List[Dict] = []
    for sample in sample_dataset:
        if str(sample.get("split", "")).strip().lower() == normalized:
            samples.append(sample)
    return samples


def print_results_table(results: Dict[int, Dict[str, float]]) -> None:
    header = f"{'shots':>7} | {'exact_match':>12} | {'bertscore_f1':>12}"
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for shots in sorted(results.keys()):
        exact_match = results[shots].get("exact_match")
        bertscore = results[shots].get("bertscore_f1")
        exact_text = f"{exact_match:.4f}" if exact_match is not None else "n/a"
        bert_text = f"{bertscore:.4f}" if bertscore is not None else "n/a"
        print(f"{shots:>7} | {exact_text:>12} | {bert_text:>12}")
    print(line)


def main() -> None:
    args = parse_args()

    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    shot_values = [int(value.strip()) for value in args.shots.split(",") if value.strip()]

    base_dataset = build_dataset(args)
    task = GenerativeMedicalVQA()
    sample_dataset = base_dataset.set_task(task=task)

    support_samples = filter_samples_by_split(sample_dataset, args.support_split)
    test_samples = filter_samples_by_split(sample_dataset, args.test_split)

    if not support_samples:
        raise ValueError(f"No support samples found for split: {args.support_split}")
    if not test_samples:
        raise ValueError(f"No test samples found for split: {args.test_split}")

    test_dataset = create_sample_dataset(
        samples=test_samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=f"{args.dataset}_test",
        task_name=task.task_name,
    )

    test_dataloader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MedFlamingo(
        dataset=test_dataset,
        llama_path=args.llama_path,
        hf_repo_id=args.hf_repo_id,
        checkpoint_filename=args.checkpoint_filename,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        num_shots=0,
        sampling_strategy="random",
        seed=args.seed,
        device=args.device,
    )
    if args.adapter_dir:
        model.load_lora_adapter(args.adapter_dir, is_trainable=False)
    model.set_support_pool(support_samples)

    metrics_by_shot: Dict[int, Dict[str, float]] = {}
    predictions_by_shot: Dict[int, List[Dict]] = {}

    for shots in shot_values:
        model.num_shots = shots
        model._rng.seed(args.seed)
        result = model.evaluate_generation(test_dataloader, metrics=metrics)
        metrics_by_shot[shots] = result["metrics"]
        predictions_by_shot[shots] = result["predictions"]

    print_results_table(metrics_by_shot)

    output_payload = {
        "dataset": args.dataset,
        "adapter_dir": args.adapter_dir,
        "support_split": args.support_split,
        "test_split": args.test_split,
        "shots": shot_values,
        "metrics_by_shot": {str(key): value for key, value in metrics_by_shot.items()},
        "predictions_by_shot": {
            str(key): value for key, value in predictions_by_shot.items()
        },
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved results to {output_path.resolve()}")

    test_dataset.close()
    sample_dataset.close()


if __name__ == "__main__":
    main()
