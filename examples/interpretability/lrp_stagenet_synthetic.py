"""
Standalone example of Layer-wise Relevance Propagation (LRP) with StageNet.

This example demonstrates LRP interpretability on StageNet using synthetic data,
so you can run it without requiring MIMIC-IV or any external datasets.

Key features demonstrated:
1. Creating synthetic patient data with multiple features
2. Training a StageNet model
3. Applying LRP for interpretability
4. Comparing epsilon-rule vs alphabeta-rule
5. Visualizing top contributing features

To run:
    python lrp_stagenet_synthetic.py
"""

import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.interpret.methods import LayerwiseRelevancePropagation
from pyhealth.models import StageNet
from pyhealth.processors import StageNetProcessor, StageNetTensorProcessor


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================


def generate_synthetic_data(
    num_samples: int = 500,
    num_visits_range: Tuple[int, int] = (3, 10),
    num_codes_range: Tuple[int, int] = (5, 20),
    num_lab_tests: int = 5,
    vocab_size: int = 100,
    seed: int = 42,
) -> list:
    """Generate synthetic patient samples for StageNet.

    Args:
        num_samples: Number of patient samples to generate
        num_visits_range: Min/max number of visits per patient
        num_codes_range: Min/max number of diagnosis codes per visit
        num_lab_tests: Number of different lab test types
        vocab_size: Size of diagnosis code vocabulary
        seed: Random seed for reproducibility

    Returns:
        List of sample dictionaries with patient data
    """
    random.seed(seed)
    np.random.seed(seed)

    samples = []

    for i in range(num_samples):
        num_visits = random.randint(*num_visits_range)

        # Generate diagnosis codes for each visit
        diagnoses_list = []
        diagnosis_times = []

        for visit_idx in range(num_visits):
            # Random number of codes per visit
            num_codes = random.randint(*num_codes_range)
            visit_codes = [f"D{random.randint(0, vocab_size-1)}" for _ in range(num_codes)]
            diagnoses_list.append(visit_codes)

            # Time intervals between visits (in hours)
            if visit_idx == 0:
                diagnosis_times.append(0.0)
            else:
                diagnosis_times.append(random.uniform(24, 720))  # 1-30 days

        # Generate lab test results for each visit
        lab_values_list = []
        lab_times = []

        measurement_idx = 0
        for visit_idx in range(num_visits):
            # Random number of lab measurements per visit
            num_measurements = random.randint(3, 10)

            for _ in range(num_measurements):
                # Generate lab values (some may be None/missing)
                lab_vector = []
                for lab_idx in range(num_lab_tests):
                    # Ensure first patient's first measurement has all valid values
                    # This helps the processor detect the feature dimension
                    if (i == 0 and measurement_idx == 0) or random.random() < 0.8:
                        # Normal lab values with some noise
                        value = 100.0 + random.gauss(0, 20)
                        lab_vector.append(value)
                    else:
                        lab_vector.append(None)

                lab_values_list.append(lab_vector)

                # Time within visit (hours from visit start)
                lab_times.append(random.uniform(0, 24))
                measurement_idx += 1

        # Generate binary label (mortality risk)
        # Higher risk if more visits or certain "risky" codes
        risky_codes = sum(
            1 for visit_codes in diagnoses_list for code in visit_codes if int(code[1:]) < 20
        )
        risk_score = (num_visits * 0.1) + (risky_codes * 0.05) + random.gauss(0, 0.1)
        label = 1 if risk_score > 0.5 else 0

        sample = {
            "patient_id": f"P{i:04d}",
            "diagnoses": (diagnosis_times, diagnoses_list),
            "labs": (lab_times, lab_values_list),
            "label": label,
        }

        samples.append(sample)

    return samples


# ============================================================================
# PROCESSOR SETUP
# ============================================================================


def setup_processors(samples: list) -> Tuple[Dict, Dict]:
    """Setup and fit processors for the synthetic data.

    Args:
        samples: List of sample dictionaries

    Returns:
        Tuple of (input_processors, output_processors)
    """
    # Initialize processors
    input_processors = {
        "diagnoses": StageNetProcessor(padding=10),
        "labs": StageNetTensorProcessor(),
    }

    output_processors = {"label": lambda x: x}  # Identity function

    # Fit processors on data
    for key, processor in input_processors.items():
        if hasattr(processor, "fit"):
            processor.fit(samples, key)

    return input_processors, output_processors


# ============================================================================
# LRP VISUALIZATION HELPERS
# ============================================================================


def print_lrp_results(
    attributions: Dict[str, torch.Tensor],
    sample_batch: Dict,
    processors: Dict,
    top_k: int = 10,
):
    """Print LRP attribution results in a readable format.

    Args:
        attributions: Dictionary of attribution tensors from LRP
        sample_batch: The input batch
        processors: Input processors for decoding
        top_k: Number of top features to display
    """
    print("\n" + "=" * 70)
    print(f"TOP {top_k} FEATURES BY LRP RELEVANCE")
    print("=" * 70)

    for feature_key, attr_tensor in attributions.items():
        if attr_tensor is None or attr_tensor.numel() == 0:
            continue

        print(f"\n{feature_key.upper()}")
        print("-" * 70)

        # Get the first sample in batch
        attr = attr_tensor[0].detach().cpu()
        flat_attr = attr.flatten()

        if flat_attr.numel() == 0:
            continue

        # Get top-k absolute relevances
        k = min(top_k, flat_attr.numel())
        top_values, top_indices = torch.topk(flat_attr.abs(), k=k)

        # Get input data
        feature_input = sample_batch[feature_key]
        if isinstance(feature_input, tuple):
            feature_input = feature_input[1]  # Get values from (time, values)

        feature_input = feature_input[0].detach().cpu()  # First sample

        # Print top features
        for rank, (abs_val, flat_idx) in enumerate(zip(top_values, top_indices), 1):
            relevance = flat_attr[flat_idx].item()

            # Convert flat index to multi-dimensional coordinates
            coords = []
            remaining = flat_idx.item()
            for dim_size in reversed(attr.shape):
                coords.append(remaining % dim_size)
                remaining //= dim_size
            coords = tuple(reversed(coords))

            # Decode the feature value
            try:
                if torch.is_floating_point(feature_input):
                    # Continuous feature (labs)
                    if len(coords) <= len(feature_input.shape):
                        actual_value = feature_input[coords].item()
                    else:
                        actual_value = 0.0
                    print(
                        f"  {rank:2d}. position={coords}, "
                        f"value={actual_value:7.2f}, "
                        f"relevance={relevance:+.6f}"
                    )
                else:
                    # Categorical feature (diagnoses)
                    # Handle possibly nested structure
                    if len(coords) == 1 and feature_input.dim() >= 1:
                        token_idx = int(feature_input[coords[0]].item())
                    elif len(coords) >= 2 and feature_input.dim() >= 2:
                        # For 2D or higher tensors
                        selected = feature_input
                        for c in coords[:min(len(coords), feature_input.dim())]:
                            selected = selected[c]
                        if selected.numel() == 1:
                            token_idx = int(selected.item())
                        else:
                            token_idx = 0  # Default for multi-element
                    else:
                        token_idx = 0
                        
                    processor = processors.get(feature_key)

                    # Decode token
                    if processor and hasattr(processor, "code_vocab"):
                        reverse_vocab = {idx: token for token, idx in processor.code_vocab.items()}
                        token = reverse_vocab.get(token_idx, f"<idx_{token_idx}>")
                    else:
                        token = f"<idx_{token_idx}>"

                    if token != "<pad>" and token_idx != 0:
                        print(
                            f"  {rank:2d}. position={coords}, "
                            f"token='{token}', "
                            f"relevance={relevance:+.6f}"
                        )
            except (IndexError, RuntimeError) as e:
                # Skip if indexing fails
                continue


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main pipeline for StageNet training and LRP interpretation."""

    print("=" * 70)
    print("STAGENET + LRP INTERPRETABILITY (SYNTHETIC DATA)")
    print("=" * 70)

    # ========================================================================
    # STEP 1: Generate Synthetic Data
    # ========================================================================
    print("\nSTEP 1: Generating synthetic patient data...")

    samples = generate_synthetic_data(
        num_samples=500,
        num_visits_range=(3, 10),
        num_codes_range=(5, 20),
        num_lab_tests=5,
        vocab_size=100,
        seed=42,
    )

    print(f"  Generated {len(samples)} patient samples")
    print(f"  Example sample keys: {list(samples[0].keys())}")

    # ========================================================================
    # STEP 2: Create SampleDataset (in-memory)
    # ========================================================================
    print("\nSTEP 2: Creating SampleDataset...")

    from pyhealth.datasets.sample_dataset import InMemorySampleDataset

    # Create label processor that returns correct shape for binary classification
    from pyhealth.processors.base_processor import FeatureProcessor
    class LabelProcessor(FeatureProcessor):
        def fit(self, samples, key):
            pass
        def process(self, value):
            # Return shape [1] so batch becomes [batch_size, 1] to match logits
            return torch.tensor([value], dtype=torch.float)
        def size(self):
            return 1

    # Create dataset directly in memory (simpler than disk-based approach)
    dataset = InMemorySampleDataset(
        samples=samples,
        input_schema={"diagnoses": "stagenet", "labs": "stagenet_tensor"},
        output_schema={"label": "binary"},  # Use mode string for output
        output_processors={"label": LabelProcessor()},  # But provide custom processor
    )

    print(f"  Dataset size: {len(dataset)}")

    # Get processors from the dataset
    input_processors = dataset.input_processors

    # Split into train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, len(dataset)))

    train_dataset = dataset.subset(train_indices)
    val_dataset = dataset.subset(val_indices)
    test_dataset = dataset.subset(test_indices)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # ====================================================================
    # STEP 3: Initialize StageNet Model
    # ====================================================================
    print("\nSTEP 3: Initializing StageNet model...")

    # Use default StageNet parameters for better LRP compatibility
    model = StageNet(
        dataset=dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # ====================================================================
    # STEP 4: Train the Model (simplified training loop)
    # ====================================================================
    print("\nSTEP 4: Training model...")

    # Force CPU for compatibility (GPU may not be compatible with PyTorch version)
    device = torch.device("cpu")
    model = model.to(device)
    print(f"  Using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create dataloaders
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)

    # Simple training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else
                   tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
                   if isinstance(v, tuple) else v
                for k, v in batch.items()
            }

            optimizer.zero_grad()

            # Forward pass (model computes loss internally)
            outputs = model(**batch)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            preds = outputs["y_prob"]
            labels = batch["label"].squeeze()
            correct += ((preds > 0.5).long() == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    # ====================================================================
    # STEP 5: LRP Interpretation
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: LAYER-WISE RELEVANCE PROPAGATION (LRP)")
    print("=" * 70)

    model.eval()

    # Get a test sample
    sample_batch = next(iter(test_loader))
    sample_batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else
           tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
           if isinstance(v, tuple) else v
        for k, v in sample_batch.items()
    }

    # Debug: Print model architecture for LRP
    print("\nModel Architecture Info:")
    print(f"  Feature keys: {model.feature_keys}")
    print(f"  Embedding dim: {model.embedding_model.embedding_dim}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'fc' in name:
            print(f"  Final FC layer: {module.in_features} → {module.out_features}")

    # Print model prediction
    with torch.no_grad():
        output = model(**sample_batch)
        pred_prob = torch.sigmoid(output["logit"]).item()
        pred_class = int(pred_prob > 0.5)
        true_label = int(sample_batch["label"].item())

    print(f"\nModel Prediction:")
    print(f"  True label: {true_label}")
    print(f"  Predicted class: {pred_class}")
    print(f"  Prediction probability: {pred_prob:.4f}")

    # ====================================================================
    # Compare LRP Rules
    # ====================================================================
    print("\n" + "=" * 70)
    print("LRP WITH EPSILON-RULE (ε=0.01)")
    print("=" * 70)

    lrp_epsilon = LayerwiseRelevancePropagation(
        model, rule="epsilon", epsilon=0.01, use_embeddings=True
    )
    
    print("Computing LRP attributions...")
    try:
        attributions_epsilon = lrp_epsilon.attribute(**sample_batch)
    except RuntimeError as e:
        print(f"\nERROR in LRP: {e}")
        print("\nTrying with larger epsilon for numerical stability...")
        lrp_epsilon = LayerwiseRelevancePropagation(
            model, rule="epsilon", epsilon=0.1, use_embeddings=True
        )
        try:
            attributions_epsilon = lrp_epsilon.attribute(**sample_batch)
        except RuntimeError as e2:
            print(f"\nERROR with larger epsilon: {e2}")
            print("\nLRP may not fully support this StageNet configuration.")
            print("Please use IntegratedGradients instead (see integrated_gradients_stagenet_synthetic.py)")
            return

    print_lrp_results(
        attributions_epsilon, sample_batch, input_processors, top_k=10
    )

    print("\n" + "=" * 70)
    print("LRP WITH ALPHABETA-RULE (α=1, β=0)")
    print("=" * 70)

    lrp_alphabeta = LayerwiseRelevancePropagation(
        model, rule="alphabeta", alpha=1.0, beta=0.0, use_embeddings=True
    )
    attributions_alphabeta = lrp_alphabeta.attribute(**sample_batch)

    print_lrp_results(
        attributions_alphabeta, sample_batch, input_processors, top_k=10
    )

    # ====================================================================
    # Conservation Property Check
    # ====================================================================
    print("\n" + "=" * 70)
    print("LRP CONSERVATION PROPERTY CHECK")
    print("=" * 70)

    with torch.no_grad():
        model_output = model(**sample_batch)["logit"].squeeze().item()

    print(f"\nModel output f(x): {model_output:.6f}")

    for feature_key in attributions_epsilon.keys():
        eps_total = attributions_epsilon[feature_key][0].sum().item()
        ab_total = attributions_alphabeta[feature_key][0].sum().item()

        print(f"\n{feature_key}:")
        print(f"  Epsilon-rule sum: {eps_total:+.6f}")
        print(f"  AlphaBeta-rule sum: {ab_total:+.6f}")

    # Total relevance
    eps_total_all = sum(
        attributions_epsilon[k][0].sum().item() for k in attributions_epsilon.keys()
    )
    ab_total_all = sum(
        attributions_alphabeta[k][0].sum().item() for k in attributions_alphabeta.keys()
    )

    print(f"\nTotal relevance across all features:")
    print(f"  Epsilon-rule: {eps_total_all:+.6f}")
    print(f"  AlphaBeta-rule: {ab_total_all:+.6f}")
    print(f"  Model output: {model_output:+.6f}")
    print(f"\nConservation error (epsilon): {abs(eps_total_all - model_output):.6f}")
    print(f"Conservation error (alphabeta): {abs(ab_total_all - model_output):.6f}")

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
LRP (Layer-wise Relevance Propagation) Properties:

1. Conservation: Relevances sum to model output f(x)
   - Both rules (epsilon and alphabeta) conserve relevance
   - Small numerical errors are expected due to floating point

2. Rule Comparison:
   - Epsilon-rule: Numerically stable, smoother attributions
   - AlphaBeta-rule: Can produce sharper visualizations
   - Both are valid - choice depends on use case

3. Interpretation:
   - Positive relevance: Feature supports the prediction
   - Negative relevance: Feature contradicts the prediction
   - Magnitude: Strength of contribution

4. Advantages of LRP:
   - Single backward pass (fast)
   - No baseline required
   - Conservation property
   - Multiple rules available

    """)

    print("=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
