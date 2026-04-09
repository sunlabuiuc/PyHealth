"""End-to-end VQA-RAD MedFlamingo pipeline with ablation study.

This script demonstrates the complete PyHealth pipeline for the MedFlamingo
model on the VQA-RAD medical visual question answering dataset:

1. Load the VQA-RAD base dataset
2. Apply ``MedicalVQATask`` via ``set_task()``
3. Split into train / validation / test sets
4. Create dataloaders
5. Train ``MedFlamingo`` with ``Trainer.train()``
6. Evaluate with ``Trainer.evaluate()``
7. Run a compact few-shot generation example
8. **Ablation study** comparing three independent axes:
   - Cross-attention density  (``cross_attn_every_n_layers`` in {1, 2, 4})
   - Perceiver resampler size (``num_resampler_tokens``       in {16, 32, 64})
   - Frozen vs. fine-tunable vision encoder  (``freeze_vision`` in {True, False})

Ablation motivation:
    MedFlamingo's core design choices are (1) how densely to interleave
    cross-attention layers between vision and language, (2) how many latent
    tokens the Perceiver Resampler compresses visual features into, and (3)
    whether the frozen CLIP backbone benefits from end-to-end fine-tuning on
    the downstream VQA task.  The three ablation axes isolate each variable
    while holding the others at the paper's default.

Usage::

    # Baseline only (fast):
    python examples/vqarad_medvqa_medflamingo.py --root /path/to/vqarad

    # With full ablation study (slower; runs 7 training trials):
    python examples/vqarad_medvqa_medflamingo.py --root /path/to/vqarad --ablation

Note:
    The default ``MedFlamingo`` constructor downloads large Hugging Face
    weights (CLIP ViT-L/14, OPT-6.7B) on first run, which requires
    substantial disk space and memory.  For fast local testing without
    downloading weights, replace ``MedFlamingo`` with the
    ``TestableMedFlamingo`` stub from ``tests/core/test_medflamingo.py``.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from pyhealth.datasets import (
    VQARADDataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import MedFlamingo
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def choose_splitter(samples):
    """Prefer patient-level splitting when the sample dataset preserves it."""
    patient_to_index = getattr(samples, "patient_to_index", {})
    if patient_to_index:
        return split_by_patient, "patient"
    return split_by_sample, "sample"


def build_few_shot_text(sample: dict) -> str:
    """Formats one processed sample as a simple in-context example."""
    return f"Q: {sample['question']}\nA: {sample['answer']}"


# ---------------------------------------------------------------------------
# Ablation helpers
# ---------------------------------------------------------------------------


def _run_one_config(
    samples,
    train_ds,
    val_ds,
    test_ds,
    *,
    cross_attn_every_n_layers: int,
    num_resampler_tokens: int,
    freeze_vision: bool,
    batch_size: int,
    epochs: int,
) -> Dict[str, float]:
    """Train and evaluate MedFlamingo for one ablation configuration.

    Args:
        samples: The full :class:`~pyhealth.datasets.SampleDataset` used to
            configure the model (vocabulary size, feature keys, etc.).
        train_ds: Training split.
        val_ds: Validation split.
        test_ds: Test split.
        cross_attn_every_n_layers: How often to insert a gated cross-attention
            dense block.  Smaller values mean denser vision-language interaction.
        num_resampler_tokens: Number of fixed-length visual tokens produced by
            the Perceiver Resampler.
        freeze_vision: Whether to freeze the CLIP vision encoder weights.
        batch_size: DataLoader batch size.
        epochs: Number of training epochs.

    Returns:
        Dict with keys ``val_accuracy``, ``val_loss``, ``test_accuracy``, and
        ``test_loss`` for this configuration.
    """
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    model = MedFlamingo(
        dataset=samples,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        num_resampler_tokens=num_resampler_tokens,
        freeze_vision=freeze_vision,
    )

    trainer = Trainer(model=model, metrics=["accuracy", "f1_macro"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
    )

    val_scores = trainer.evaluate(val_loader)
    test_scores = trainer.evaluate(test_loader)

    return {
        "val_accuracy": val_scores.get("accuracy", float("nan")),
        "val_loss": val_scores.get("loss", float("nan")),
        "test_accuracy": test_scores.get("accuracy", float("nan")),
        "test_loss": test_scores.get("loss", float("nan")),
    }


def _print_results_table(rows: List[dict], title: str) -> None:
    """Print a formatted results table for the ablation study.

    Args:
        rows: List of dicts, each containing ``config`` and four metric keys.
        title: Title printed above the table.
    """
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    header = (
        f"{'Config':<36} {'Val Acc':>8} {'Val Loss':>9}"
        f" {'Test Acc':>9} {'Test Loss':>10}"
    )
    print(header)
    print("-" * 72)
    for row in rows:
        print(
            f"{row['config']:<36}"
            f" {row['val_accuracy']:>8.4f}"
            f" {row['val_loss']:>9.4f}"
            f" {row['test_accuracy']:>9.4f}"
            f" {row['test_loss']:>10.4f}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train MedFlamingo on VQA-RAD with optional ablation study"
    )
    parser.add_argument("--root", required=True, help="Path to the VQA-RAD root")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for processed dataset artifacts",
    )
    parser.add_argument("--dataset-num-workers", type=int, default=1)
    parser.add_argument("--task-num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Run full ablation study across cross_attn_every_n_layers, "
            "num_resampler_tokens, and freeze_vision (runs 7 training trials)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1 – Load dataset
    # ------------------------------------------------------------------
    dataset = VQARADDataset(
        root=args.root,
        cache_dir=args.cache_dir,
        num_workers=args.dataset_num_workers,
    )
    dataset.stats()

    # ------------------------------------------------------------------
    # Step 2 – Apply task
    # ------------------------------------------------------------------
    task_samples = dataset.set_task(num_workers=args.task_num_workers)

    # ------------------------------------------------------------------
    # Step 3 – Split
    # ------------------------------------------------------------------
    splitter, split_name = choose_splitter(task_samples)
    print(f"Using {split_name}-level split")
    train_dataset, val_dataset, test_dataset = splitter(
        task_samples,
        [0.7, 0.1, 0.2],
        seed=42,
    )

    # ------------------------------------------------------------------
    # Steps 4-6 – Baseline training run (default hyperparameters)
    # cross_attn_every_n_layers=4, num_resampler_tokens=64, freeze_vision=True
    # ------------------------------------------------------------------
    print("\n=== Baseline (xattn_every=4, tokens=64, frozen_vision=True) ===")
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MedFlamingo(dataset=task_samples)
    trainer = Trainer(model=model, metrics=["accuracy", "f1_macro"])

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )

    test_metrics = trainer.evaluate(test_loader)
    print("Baseline test metrics:", test_metrics)

    # ------------------------------------------------------------------
    # Step 7 – Few-shot generation example
    # ------------------------------------------------------------------
    query_sample = test_dataset[0]
    context_sample = train_dataset[0]
    generation = model.generate(
        images=[query_sample["image"]],
        prompt=query_sample["question"],
        few_shot_examples=[
            {
                "image": context_sample["image"],
                "text": build_few_shot_text(context_sample),
            }
        ],
        max_new_tokens=args.max_new_tokens,
    )
    print("Few-shot generation:", generation)

    # ------------------------------------------------------------------
    # Step 8 – Ablation study
    #
    # Three independent axes are studied:
    #
    # A) Cross-attention density  (cross_attn_every_n_layers ∈ {1, 2, 4})
    #    More frequent cross-attention inserts more vision-language bridges
    #    into the frozen LLM stack.  The paper uses every 4th layer; denser
    #    insertion trades compute for richer multimodal grounding.
    #
    # B) Perceiver Resampler capacity (num_resampler_tokens ∈ {16, 32, 64})
    #    The resampler maps raw CLIP patch tokens to a fixed-length sequence.
    #    Fewer tokens are cheaper but may lose spatial detail; more tokens
    #    preserve finer-grained visual information.
    #
    # C) Vision encoder fine-tuning (freeze_vision ∈ {True, False})
    #    The original Flamingo/MedFlamingo paper freezes CLIP to preserve its
    #    pretrained representations.  Unfreezing allows CLIP to adapt to
    #    medical imagery but risks overfitting on small datasets.
    #
    # All ablations use a single training epoch for speed; increase --epochs
    # for more reliable comparisons.
    # ------------------------------------------------------------------
    if args.ablation:
        print("\n\n" + "#" * 72)
        print("# ABLATION STUDY")
        print("#" * 72)

        # ---- Ablation A: cross_attn_every_n_layers ----
        xattn_results = []
        for n in [1, 2, 4]:
            print(f"\n--- Ablation A: cross_attn_every_n_layers={n} ---")
            scores = _run_one_config(
                task_samples,
                train_dataset,
                val_dataset,
                test_dataset,
                cross_attn_every_n_layers=n,
                num_resampler_tokens=64,      # default
                freeze_vision=True,           # default
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            xattn_results.append({"config": f"xattn_every={n}", **scores})
        _print_results_table(
            xattn_results,
            "Ablation A: cross_attn_every_n_layers"
            " (tokens=64, frozen_vision=True)",
        )

        # ---- Ablation B: num_resampler_tokens ----
        token_results = []
        for t in [16, 32, 64]:
            print(f"\n--- Ablation B: num_resampler_tokens={t} ---")
            scores = _run_one_config(
                task_samples,
                train_dataset,
                val_dataset,
                test_dataset,
                cross_attn_every_n_layers=4,  # default
                num_resampler_tokens=t,
                freeze_vision=True,           # default
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            token_results.append({"config": f"resampler_tokens={t}", **scores})
        _print_results_table(
            token_results,
            "Ablation B: num_resampler_tokens"
            " (xattn_every=4, frozen_vision=True)",
        )

        # ---- Ablation C: freeze_vision ----
        freeze_results = []
        for fv in [True, False]:
            label = "frozen" if fv else "fine-tuned"
            print(f"\n--- Ablation C: freeze_vision={fv} ({label}) ---")
            scores = _run_one_config(
                task_samples,
                train_dataset,
                val_dataset,
                test_dataset,
                cross_attn_every_n_layers=4,  # default
                num_resampler_tokens=64,      # default
                freeze_vision=fv,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            freeze_results.append({"config": f"vision_{label}", **scores})
        _print_results_table(
            freeze_results,
            "Ablation C: freeze_vision"
            " (xattn_every=4, resampler_tokens=64)",
        )

        print("\nAblation study complete.")

    task_samples.close()
