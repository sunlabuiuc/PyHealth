"""LRP with StageNet on MIMIC-IV for mortality prediction.

Demonstrates Layer-wise Relevance Propagation (LRP) interpretability
using epsilon-rule and alphabeta-rule on MIMIC-IV data.

Usage:
    python lrp_stagenet_mimic4.py
"""

from pathlib import Path

import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    load_processors,
    save_processors,
    split_by_patient,
)
from pyhealth.interpret.methods import LayerwiseRelevancePropagation
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer


def decode_indices_to_tokens(indices_tensor, processor, feature_key):
    """Decode token indices back to original codes using processor vocabulary."""
    if not hasattr(processor, "code_vocab"):
        return None
    reverse_vocab = {idx: token for token, idx in processor.code_vocab.items()}

    def decode(idx):
        return reverse_vocab.get(idx, f"<unknown_{idx}>")

    items = indices_tensor.tolist()
    if indices_tensor.dim() == 1:
        return [decode(i) for i in items]
    elif indices_tensor.dim() == 2:
        return [[decode(i) for i in row] for row in items]
    elif indices_tensor.dim() == 3:
        return [[[decode(i) for i in inner] for inner in row] for row in items]
    return items


def print_lrp_results(attributions, sample_batch, sample_dataset, top_k=10):
    """Print top-k LRP attribution results per feature."""
    processors = sample_dataset.input_processors

    for feature_key, attr in attributions.items():
        if attr.numel() == 0:
            continue

        input_data = sample_batch[feature_key]
        if isinstance(input_data, tuple):
            input_tensor = input_data[1]
        else:
            input_tensor = input_data

        total = attr[0].sum().item()
        flat = attr[0].flatten()
        k = min(top_k, flat.numel())
        top_idx = torch.topk(flat.abs(), k=k).indices

        print(f"\n  {feature_key} (shape={attr.shape}, total_relevance={total:+.6f}):")

        is_continuous = torch.is_floating_point(input_tensor)
        processor = processors.get(feature_key)

        for rank, fidx in enumerate(top_idx.tolist(), 1):
            val = flat[fidx].item()
            if is_continuous and attr[0].dim() == 3:
                dim2 = attr[0].shape[2]
                t, f = fidx // dim2, fidx % dim2
                if input_tensor.dim() == 3 and t < input_tensor.shape[1] and f < input_tensor.shape[2]:
                    actual = input_tensor[0, t, f].item()
                    print(f"    {rank:2d}. T{t} F{f} val={actual:7.2f} -> {val:+.6f}")
                else:
                    print(f"    {rank:2d}. idx={fidx} -> {val:+.6f}")
            elif not is_continuous and processor:
                tokens = decode_indices_to_tokens(input_tensor[0], processor, feature_key)
                if tokens and attr[0].dim() == 3:
                    dim2 = attr[0].shape[2]
                    t, f = fidx // dim2, fidx % dim2
                    if t < len(tokens) and f < len(tokens[t]):
                        print(f"    {rank:2d}. Visit {t} '{tokens[t][f]}' -> {val:+.6f}")
                        continue
                print(f"    {rank:2d}. idx={fidx} -> {val:+.6f}")
            else:
                print(f"    {rank:2d}. idx={fidx} -> {val:+.6f}")


def main():
    # Load MIMIC-IV
    print("Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients", "admissions", "diagnoses_icd",
            "procedures_icd", "labevents",
        ],
        dev=True,
    )
    base_dataset.stats()

    # Processors
    processor_dir = Path("../../output/processors/stagenet_mortality_mimic4_lrp")
    cache_dir = Path("../../mimic4_stagenet_lrp_cache")

    if processor_dir.exists() and any(processor_dir.iterdir()):
        print(f"Loading processors from {processor_dir}")
        input_processors = load_processors(str(processor_dir))
        sample_dataset = base_dataset.set_task(
            MortalityPredictionStageNetMIMIC4(padding=20),
            processors=input_processors,
            cache_dir=str(cache_dir),
        )
    else:
        print("Creating new processors...")
        processor_dir.mkdir(parents=True, exist_ok=True)
        sample_dataset = base_dataset.set_task(
            MortalityPredictionStageNetMIMIC4(padding=20),
            cache_dir=str(cache_dir),
        )
        save_processors(sample_dataset.input_processors, str(processor_dir))

    print(f"Samples: {len(sample_dataset)}")

    # Split
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=1, shuffle=False)

    # Model
    model = StageNet(
        dataset=sample_dataset, embedding_dim=128, chunk_size=128,
        levels=3, dropout=0.3,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(
        model=model, device="cpu",
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )
    trainer.train(
        train_dataloader=train_loader, val_dataloader=val_loader,
        epochs=5, monitor="roc_auc", optimizer_params={"lr": 1e-4},
    )

    # Evaluate
    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # LRP interpretation
    sample_batch = next(iter(test_loader))

    with torch.no_grad():
        output = model(**sample_batch)
        probs = output["y_prob"]
        pred = torch.argmax(probs, dim=-1)
        true_label = sample_batch[model.label_key]
    print(f"\nPrediction: true={int(true_label[0].item())}, "
          f"pred={int(pred[0].item())}, "
          f"P(survived)={probs[0, 0].item():.4f}, P(died)={probs[0, 1].item():.4f}")

    # Epsilon rule
    print("\nLRP Epsilon-Rule (eps=0.01):")
    lrp_eps = LayerwiseRelevancePropagation(
        model, rule="epsilon", epsilon=0.01, use_embeddings=True
    )
    attr_eps = lrp_eps.attribute(**sample_batch)
    print_lrp_results(attr_eps, sample_batch, sample_dataset)

    # AlphaBeta rule
    print("\nLRP AlphaBeta-Rule (alpha=1.0, beta=0.0):")
    lrp_ab = LayerwiseRelevancePropagation(
        model, rule="alphabeta", alpha=1.0, beta=0.0, use_embeddings=True
    )
    attr_ab = lrp_ab.attribute(**sample_batch)
    print_lrp_results(attr_ab, sample_batch, sample_dataset)

    # Conservation comparison
    print("\nRelevance comparison:")
    for key in attr_eps:
        eps_t = attr_eps[key][0].sum().item()
        ab_t = attr_ab[key][0].sum().item()
        print(f"  {key}: epsilon={eps_t:+.6f}, alphabeta={ab_t:+.6f}")

    print(f"\nProcessors saved at: {processor_dir}")


if __name__ == "__main__":
    main()