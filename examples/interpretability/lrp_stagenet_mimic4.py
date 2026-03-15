"""
Example of using Layer-wise Relevance Propagation (LRP) with StageNet on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Loading pre-trained processors (or creating new ones)
3. Training a StageNet model for mortality prediction
4. Using LRP for interpretability with different propagation rules
5. Comparing epsilon-rule vs alphabeta-rule
6. Visualizing top contributing features

Key advantages of LRP:
- Single backward pass (faster than IG which needs multiple forward passes)
- No baseline required (unlike IG or DeepLift)
- Conservation property: relevances sum to model output f(x)
- Different rules available for specific use cases:
  * epsilon-rule: numerically stable, smoother attributions
  * alphabeta-rule: sharper visualizations, highlights strong evidence

Lab Feature Mapping (10-dimensional vector):
    Dimension 0: Sodium
    Dimension 1: Potassium
    Dimension 2: Chloride
    Dimension 3: Bicarbonate
    Dimension 4: Glucose
    Dimension 5: Calcium
    Dimension 6: Magnesium
    Dimension 7: Anion Gap
    Dimension 8: Osmolality
    Dimension 9: Phosphate
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


# ============================================================================
# HELPER FUNCTIONS FOR LRP ATTRIBUTION VISUALIZATION
# ============================================================================


def decode_indices_to_tokens(
    indices_tensor: torch.Tensor, processor, feature_key: str
) -> list:
    """Decode token indices back to original codes using processor vocabulary.

    Args:
        indices_tensor: Tensor of token indices [batch, seq_len, tokens] or
                       [batch, seq_len]
        processor: The feature processor with code_vocab
        feature_key: Name of the feature (for printing)

    Returns:
        List structure matching input tensor dimensions with decoded tokens
    """
    # Create reverse vocabulary mapping: index -> token
    if not hasattr(processor, "code_vocab"):
        return None

    reverse_vocab = {idx: token for token, idx in processor.code_vocab.items()}

    # Convert tensor to list for easier processing
    indices_list = indices_tensor.tolist()

    # Helper to decode a single index
    def decode_idx(idx):
        return reverse_vocab.get(idx, f"<unknown_idx_{idx}>")

    # Handle different dimensionalities
    if indices_tensor.dim() == 1:
        # 1D: [seq_len]
        return [decode_idx(idx) for idx in indices_list]
    elif indices_tensor.dim() == 2:
        # 2D: [batch, seq_len] or [seq_len, tokens]
        return [[decode_idx(idx) for idx in row] for row in indices_list]
    elif indices_tensor.dim() == 3:
        # 3D: [batch, seq_len, tokens]
        return [
            [[decode_idx(idx) for idx in inner] for inner in row]
            for row in indices_list
        ]
    else:
        return indices_list


def _print_attributions(
    indices, flat_attr, attr, input_tensor, is_continuous, processor, feature_key
):
    """Helper to print attribution details."""
    # Define lab category names for dimension mapping
    LAB_CATEGORY_NAMES = [
        "Sodium",
        "Potassium",
        "Chloride",
        "Bicarbonate",
        "Glucose",
        "Calcium",
        "Magnesium",
        "Anion Gap",
        "Osmolality",
        "Phosphate",
    ]

    if attr.dim() == 3:
        dim2 = attr.shape[2]
        for rank, flat_idx in enumerate(indices.tolist(), 1):
            idx1 = flat_idx // dim2
            idx2 = flat_idx % dim2
            attr_val = flat_attr[flat_idx].item()

            if is_continuous:
                # Continuous: show timestep, feature (with lab name), value
                if (
                    input_tensor.dim() == 3
                    and idx1 < input_tensor.shape[1]
                    and idx2 < input_tensor.shape[2]
                ):
                    actual_value = input_tensor[0, idx1, idx2].item()
                    sign = "+" if attr_val >= 0 else ""

                    # Map dimension to lab name if this is labs feature
                    if feature_key == "labs" and idx2 < len(LAB_CATEGORY_NAMES):
                        lab_name = LAB_CATEGORY_NAMES[idx2]
                        print(
                            f"    {rank:2d}. T{idx1:3d} {lab_name:12s} "
                            f"val={actual_value:7.2f} → {sign}{attr_val:.6f}"
                        )
                    else:
                        print(
                            f"    {rank:2d}. T{idx1:3d} F{idx2} "
                            f"val={actual_value:7.2f} → {sign}{attr_val:.6f}"
                        )
            else:
                # Discrete: decode tokens with "Visit" prefix
                decoded_tokens = decode_indices_to_tokens(
                    input_tensor[0], processor, feature_key
                )
                if (
                    decoded_tokens
                    and idx1 < len(decoded_tokens)
                    and idx2 < len(decoded_tokens[idx1])
                ):
                    token = decoded_tokens[idx1][idx2]
                    sign = "+" if attr_val >= 0 else ""
                    print(
                        f"    {rank:2d}. Visit {idx1} Token {idx2:2d} "
                        f"'{token:12s}' → {sign}{attr_val:.6f}"
                    )

    elif attr.dim() == 2:
        for rank, flat_idx in enumerate(indices.tolist(), 1):
            attr_val = flat_attr[flat_idx].item()
            sign = "+" if attr_val >= 0 else ""

            if is_continuous and input_tensor.dim() >= 2:
                if flat_idx < input_tensor.shape[1]:
                    actual_value = input_tensor[0, flat_idx].item()
                    print(
                        f"    {rank:2d}. Idx{flat_idx:3d} "
                        f"val={actual_value:7.2f} → {sign}{attr_val:.6f}"
                    )
            else:
                decoded_tokens = decode_indices_to_tokens(
                    input_tensor[0], processor, feature_key
                )
                if decoded_tokens and flat_idx < len(decoded_tokens):
                    token = decoded_tokens[flat_idx]
                    print(
                        f"    {rank:2d}. Idx{flat_idx:3d} "
                        f"'{token:12s}' → {sign}{attr_val:.6f}"
                    )


def print_lrp_attribution_results(
    attributions, sample_batch, sample_dataset, target_name, top_k=10
):
    """Print LRP attribution results in a clean, organized format.

    Args:
        attributions: Dict of attribution tensors from LRP
        sample_batch: Input batch
        sample_dataset: Dataset with processors
        target_name: Name of the target/label (e.g., 'mortality')
        top_k: Number of top features to display per input
    """
    print(f"\n{'='*70}")
    print(f"LRP ATTRIBUTION RESULTS (Top {top_k} features)")
    print(f"{'='*70}")

    # Get processors for decoding
    processors = sample_dataset.input_processors

    for feature_key, attr in attributions.items():
        # Skip if attribution is empty
        if attr.numel() == 0:
            continue

        # Get corresponding input
        input_data = sample_batch[feature_key]

        # Handle tuple format (time, values) for StageNet
        if isinstance(input_data, tuple):
            input_tensor = input_data[1]  # Get values tensor
        else:
            input_tensor = input_data

        # Get processor for this feature
        processor = processors.get(feature_key)

        # Determine if continuous or discrete
        is_continuous = torch.is_floating_point(input_tensor)

        # Calculate total relevance and statistics
        total_relevance = attr[0].sum().item()
        positive_relevance = attr[0][attr[0] > 0].sum().item()
        negative_relevance = attr[0][attr[0] < 0].sum().item()

        print(f"\n{feature_key.upper()}")
        print(f"  Shape: {attr.shape}")
        print(f"  Total relevance: {total_relevance:+.6f}")
        print(f"  Positive: {positive_relevance:+.6f} | "
              f"Negative: {negative_relevance:+.6f}")

        # Flatten for top-k selection
        flat_attr = attr[0].flatten()

        # Get top-k by absolute value
        k = min(top_k, flat_attr.numel())
        top_indices = torch.topk(flat_attr.abs(), k=k).indices

        print(f"\n  Top {k} features by absolute LRP relevance:")
        _print_attributions(
            top_indices,
            flat_attr,
            attr[0],
            input_tensor,
            is_continuous,
            processor,
            feature_key,
        )

    print(f"\n{'='*70}\n")


def print_model_prediction(model, sample_batch, device="cpu"):
    """Print model's prediction details.

    Args:
        model: Trained StageNet model
        sample_batch: Input batch
        device: Device to run on
    """
    # Move batch to device
    sample_batch_device = {}
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            sample_batch_device[key] = value.to(device)
        elif isinstance(value, tuple):
            sample_batch_device[key] = tuple(v.to(device) for v in value)
        else:
            sample_batch_device[key] = value

    # Get prediction
    with torch.no_grad():
        output = model(**sample_batch_device)
        probs = output["y_prob"]
        preds = torch.argmax(probs, dim=-1)
        label_key = model.label_key
        true_label = sample_batch_device[label_key]

    print("\n" + "=" * 70)
    print("MODEL PREDICTION")
    print("=" * 70)
    print(f"  True label: {int(true_label.cpu()[0].item())}")
    print(f"  Predicted class: {int(preds.cpu()[0].item())}")
    print(f"  Class probabilities:")
    print(f"    Class 0 (Survived): {probs[0, 0].cpu().item():.4f}")
    print(f"    Class 1 (Died): {probs[0, 1].cpu().item():.4f}")
    print("=" * 70)


def compare_lrp_rules(
    model, sample_batch, sample_dataset, device="cpu", top_k=10
):
    """Compare epsilon-rule and alphabeta-rule LRP attributions.

    Args:
        model: Trained StageNet model
        sample_batch: Input batch
        sample_dataset: Dataset with processors
        device: Device to run on
        top_k: Number of top features to display
    """
    # Move batch to device
    sample_batch_device = {}
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            sample_batch_device[key] = value.to(device)
        elif isinstance(value, tuple):
            sample_batch_device[key] = tuple(v.to(device) for v in value)
        else:
            sample_batch_device[key] = value

    print("\n" + "=" * 70)
    print("COMPARING LRP PROPAGATION RULES")
    print("=" * 70)

    # 1. Epsilon rule (default, numerically stable)
    print("\n" + "-" * 70)
    print("1. EPSILON RULE (ε=0.01)")
    print("-" * 70)
    print("Properties:")
    print("  - Numerically stable")
    print("  - Smoother attributions")
    print("  - Good for general interpretation")
    print("  - Prevents division by zero")

    lrp_epsilon = LayerwiseRelevancePropagation(
        model, rule="epsilon", epsilon=0.01, use_embeddings=True
    )
    attributions_epsilon = lrp_epsilon.attribute(**sample_batch_device)

    print_lrp_attribution_results(
        attributions_epsilon, sample_batch_device, sample_dataset, "mortality", top_k
    )

    # 2. Alpha-beta rule (sharper visualizations)
    print("\n" + "-" * 70)
    print("2. ALPHABETA RULE (α=1.0, β=0.0)")
    print("-" * 70)
    print("Properties:")
    print("  - Sharper visualizations")
    print("  - Highlights strong positive evidence")
    print("  - Ignores negative contributions (β=0)")
    print("  - Better for identifying key features")

    lrp_alphabeta = LayerwiseRelevancePropagation(
        model, rule="alphabeta", alpha=1.0, beta=0.0, use_embeddings=True
    )
    attributions_alphabeta = lrp_alphabeta.attribute(**sample_batch_device)

    print_lrp_attribution_results(
        attributions_alphabeta,
        sample_batch_device,
        sample_dataset,
        "mortality",
        top_k,
    )

    # 3. Compare total relevances
    print("\n" + "-" * 70)
    print("RELEVANCE COMPARISON")
    print("-" * 70)

    for feature_key in attributions_epsilon.keys():
        eps_total = attributions_epsilon[feature_key][0].sum().item()
        ab_total = attributions_alphabeta[feature_key][0].sum().item()

        print(f"\n{feature_key}:")
        print(f"  Epsilon-rule total: {eps_total:+.6f}")
        print(f"  AlphaBeta-rule total: {ab_total:+.6f}")
        print(f"  Difference: {abs(eps_total - ab_total):.6f}")


# ============================================================================
# MAIN TRAINING AND INTERPRETATION PIPELINE
# ============================================================================


def main():
    """Main pipeline for StageNet training and LRP interpretation."""

    print("=" * 70)
    print("STAGENET + LRP INTERPRETABILITY EXAMPLE")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Load MIMIC-IV Dataset
    # ========================================================================
    print("\nSTEP 1: Loading MIMIC-IV dataset...")

    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        dev=True,  # Use development mode for faster processing
    )

    base_dataset.stats()  # This prints dataset statistics including number of patients

    # ========================================================================
    # STEP 2: Check for Existing Processors or Create New Ones
    # ========================================================================
    print("\nSTEP 2: Checking for existing processors...")

    processor_dir = Path("../../output/processors/stagenet_mortality_mimic4_lrp")
    cache_dir = Path("../../mimic4_stagenet_lrp_cache")

    if processor_dir.exists() and any(processor_dir.iterdir()):
        print(f"  ✓ Found existing processors at: {processor_dir}")
        print("  Loading processors for consistent encoding...")

        # Load existing processors
        input_processors = load_processors(str(processor_dir))

        # Apply task and create dataset
        sample_dataset = base_dataset.set_task(
            MortalityPredictionStageNetMIMIC4(padding=20),
            processors=input_processors,
            cache_dir=str(cache_dir),
        )
    else:
        print("  ✗ No existing processors found")
        print("  Creating new processors...")
        processor_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset with new processors
        sample_dataset = base_dataset.set_task(
            MortalityPredictionStageNetMIMIC4(padding=20),
            cache_dir=str(cache_dir),
        )

        # Save processors for future use
        save_processors(sample_dataset.input_processors, str(processor_dir))
        print(f"  ✓ Processors saved to: {processor_dir}")

    print(f"  Total samples created: {len(sample_dataset)}")

    # ========================================================================
    # STEP 3: Split Dataset
    # ========================================================================
    print("\nSTEP 3: Splitting dataset...")

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)

    # ========================================================================
    # STEP 4: Initialize StageNet Model
    # ========================================================================
    print("\nSTEP 4: Initializing StageNet model...")

    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # ========================================================================
    # STEP 5: Train the Model
    # ========================================================================
    print("\nSTEP 5: Training model...")

    trainer = Trainer(
        model=model,
        device="cpu",  # Change to "cuda" if available
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=5,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-4},
    )

    # ========================================================================
    # STEP 6: Evaluate on Test Set
    # ========================================================================
    print("\nSTEP 6: Evaluating on test set...")

    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # ========================================================================
    # STEP 7: LRP Interpretation on Sample
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: LAYER-WISE RELEVANCE PROPAGATION (LRP)")
    print("=" * 70)

    # Get a test sample
    sample_batch = next(iter(test_loader))

    # Print model's prediction
    print_model_prediction(model, sample_batch, device="cpu")

    # Compare different LRP rules
    compare_lrp_rules(
        model=model,
        sample_batch=sample_batch,
        sample_dataset=sample_dataset,
        device="cpu",
        top_k=10,
    )

    # ========================================================================
    # STEP 8: Key Insights about LRP
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS ABOUT LRP")
    print("=" * 70)
    print("""
LRP vs Other Attribution Methods:

1. LRP (Layer-wise Relevance Propagation):
   ✓ Single backward pass (fast)
   ✓ No baseline needed
   ✓ Relevances sum to f(x) (conservation property)
   ✓ Multiple rules available (epsilon, alphabeta, etc.)
   - Rule choice affects interpretation

2. Integrated Gradients:
   ✓ Theoretically grounded (axiomatic)
   ✓ Baseline-independent results
   - Requires multiple forward passes (slower)
   - Needs baseline selection
   - Gradients sum to f(x) - f(baseline)

3. DeepLift:
   ✓ Fast (single pass)
   ✓ Captures saturation effects
   - Requires baseline selection
   - Attribution sum: f(x) - f(baseline)

When to use LRP:
- Need fast attributions (single backward pass)
- Conservation property desired (sum to f(x))
- Want to try different propagation rules
- No obvious baseline exists
- Debugging model decisions

When to use other methods:
- IG: Need theoretical guarantees, have good baseline
- DeepLift: ReLU-heavy models, have good baseline
- SHAP: Need game-theoretic explanation
- LIME: Need local linear approximation
    """)

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nProcessors saved at: {processor_dir}")
    print("Rerun this script to use cached processors for faster startup.")


if __name__ == "__main__":
    main()
