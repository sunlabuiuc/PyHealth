"""
Example of using StageNet for mortality prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the MortalityPredictionStageNetMIMIC4 task
3. Creating a SampleDataset with StageNet processors
4. Training a StageNet model
5. Using Integrated Gradients for interpretability

Lab Feature Mapping:
    The MortalityPredictionStageNetMIMIC4 task creates 10-dimensional lab
    vectors where each dimension corresponds to a lab category:

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

    This mapping is used to decode attribution indices back to lab names.
"""

import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer


# ============================================================================
# HELPER FUNCTIONS FOR ATTRIBUTION VISUALIZATION
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


def print_attribution_results(
    attributions, sample_batch, sample_dataset, target_name, top_k=10
):
    """Print attribution results in a clean, organized format.

    Args:
        attributions: Dict of attribution tensors
        sample_batch: Input batch
        sample_dataset: Dataset with processors
        target_name: Name of target class (e.g., "Mortality=1")
        top_k: Number of top elements to display (per direction)
    """
    print(f"\n{'='*70}")
    print(f"Attribution Results for {target_name}")
    print(f"{'='*70}")
    print("NOTE: Attributions can be positive (increases prediction) or")
    print("      negative (decreases prediction).")
    print("      They sum to f(x) - f(baseline).")
    print(f"{'='*70}\n")

    for feature_key in attributions:
        attr = attributions[feature_key]
        print(f"{feature_key}:")
        print(f"  Shape: {attr.shape}")
        print(f"  Mean: {attr.mean().item():.6f}, " f"Std: {attr.std().item():.6f}")
        print(f"  Range: [{attr.min().item():.6f}, " f"{attr.max().item():.6f}]")
        print(f"  Sum: {attr.sum().item():.6f}\n")

        # Get processor and input tensor
        processor = sample_dataset.input_processors.get(feature_key)
        input_tensor = sample_batch[feature_key]
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[1]

        is_continuous = processor is None or not hasattr(processor, "code_vocab")

        # Show top positive and top negative separately
        if attr.numel() > 0:
            flat_attr = attr.flatten()
            top_k_actual = min(top_k, flat_attr.numel())

            # Get top positive attributions
            pos_mask = flat_attr > 0
            if pos_mask.any():
                pos_vals = flat_attr[pos_mask]
                pos_indices_rel = torch.topk(
                    pos_vals, k=min(top_k_actual, len(pos_vals))
                ).indices
                pos_indices = torch.where(pos_mask)[0][pos_indices_rel]

                print(f"  Top {len(pos_indices)} POSITIVE " f"(increases prediction):")
                _print_attributions(
                    pos_indices,
                    flat_attr,
                    attr,
                    input_tensor,
                    is_continuous,
                    processor,
                    feature_key,
                )

            # Get top negative attributions
            neg_mask = flat_attr < 0
            if neg_mask.any():
                neg_vals = flat_attr[neg_mask]
                neg_indices_rel = torch.topk(
                    torch.abs(neg_vals), k=min(top_k_actual, len(neg_vals))
                ).indices
                neg_indices = torch.where(neg_mask)[0][neg_indices_rel]

                print(f"  Top {len(neg_indices)} NEGATIVE " f"(decreases prediction):")
                _print_attributions(
                    neg_indices,
                    flat_attr,
                    attr,
                    input_tensor,
                    is_continuous,
                    processor,
                    feature_key,
                )
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function for StageNet mortality prediction with IG."""
    # STEP 1: Load MIMIC-IV base dataset
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
    )

    # STEP 2: Apply StageNet mortality prediction task
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=4,
        cache_dir="../../mimic4_stagenet_cache",
    )

    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    # Inspect a sample
    sample = sample_dataset.samples[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"ICD Codes: {sample['icd_codes']}")
    print(f"  Labs shape: {len(sample['labs'][0])} timesteps")
    print(f"  Mortality: {sample['mortality']}")

    # STEP 3: Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

    # STEP 4: Initialize StageNet model
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params} parameters")

    # STEP 5: Load pre-trained model checkpoint
    print(f"\n{'='*60}")
    print("LOADING PRE-TRAINED MODEL")
    print(f"{'='*60}")

    checkpoint_path = (
        "/home/johnwu3/projects/PyHealth_Branch_Testing/PyHealth/output/"
        "20251028-191219/best.ckpt"
    )

    print(f"✓ Loading checkpoint: {checkpoint_path}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        device="cpu",
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )

    # Load the checkpoint
    trainer.load_ckpt(checkpoint_path)
    print("✓ Checkpoint loaded successfully")

    # STEP 6: Interpretability with Integrated Gradients
    print(f"\n{'='*60}")
    print("INTEGRATED GRADIENTS INTERPRETABILITY")
    print(f"{'='*60}")

    # Create single-sample dataloader for interpretability
    single_sample_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    print("✓ Integrated Gradients interpreter initialized")

    # Get a single test sample
    sample_batch = next(iter(single_sample_loader))
    print("\nAnalyzing sample:")
    print(f"  Patient ID: {sample_batch['patient_id'][0]}")
    print(f"  True label: {sample_batch['mortality'][0].item()}")

    # Get model prediction for this sample
    with torch.no_grad():
        output = model(**sample_batch)
        predicted_prob = output["y_prob"][0, 0].item()
        predicted_class = int(predicted_prob > 0.5)
    print(f"  Predicted class: {predicted_class}")
    print(f"  Predicted probability: {predicted_prob:.4f}")

    # Compute attributions for both target classes to compare
    print("\nComputing attributions for both target classes...")
    print("(Helps understand what drives predictions in each direction)\n")

    attributions_mortality = ig.attribute(**sample_batch, target_class_idx=1, steps=5)
    attributions_survival = ig.attribute(**sample_batch, target_class_idx=0, steps=5)

    # Display results for mortality prediction (target=1)
    print_attribution_results(
        attributions_mortality,
        sample_batch,
        sample_dataset,
        "Target: Mortality = 1 (Death)",
        top_k=10,
    )

    # Display results for survival prediction (target=0)
    print_attribution_results(
        attributions_survival,
        sample_batch,
        sample_dataset,
        "Target: Mortality = 0 (Survival)",
        top_k=10,
    )

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Comparing Attributions by Target Class")
    print(f"{'='*70}")
    print("Features that increase mortality risk (positive in target=1)")
    print("  vs. features that increase survival (positive in target=0)\n")

    for feature_key in attributions_mortality:
        mort_sum = attributions_mortality[feature_key].sum().item()
        surv_sum = attributions_survival[feature_key].sum().item()
        print(f"{feature_key}:")
        print(f"  Target=1 (mortality) sum: {mort_sum:+.6f}")
        print(f"  Target=0 (survival) sum:  {surv_sum:+.6f}")
        print(f"  Difference (should be ~0): {mort_sum + surv_sum:.6f}\n")

    print("Interpretation Guide:")
    print("  • For target=1: positive = increases death risk")
    print("  •               negative = protective against death")
    print("  • For target=0: positive = increases survival chance")
    print("  •               negative = increases death risk")
    print("  • Attributions from both targets have roughly opposite signs")


if __name__ == "__main__":
    main()
