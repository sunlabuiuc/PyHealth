"""
Simple example demonstrating Layer-wise Relevance Propagation (LRP) usage.

This example shows how to use LRP for model interpretability on a simple
dataset with synthetic data.

LRP provides direct relevance scores that sum to the model's output,
unlike Integrated Gradients which sums to the difference from a baseline.
"""

import torch

from pyhealth.datasets import SampleDataset, get_dataloader, split_by_patient
from pyhealth.interpret.methods import LayerWiseRelevancePropagation
from pyhealth.models import MLP
from pyhealth.trainer import Trainer


def create_synthetic_dataset():
    """Create a simple synthetic dataset for demonstration."""
    samples = []
    
    # Create synthetic patient data
    for patient_id in range(100):
        for visit_id in range(3):
            sample = {
                "patient_id": f"patient-{patient_id}",
                "visit_id": f"visit-{visit_id}",
                # Discrete features (diagnosis codes)
                "conditions": [f"cond-{i}" for i in range(3)],
                # Continuous features (lab values)
                "labs": [float(i) * 1.5 for i in range(5)],
                # Binary label
                "label": patient_id % 2,
            }
            samples.append(sample)
    
    # Create dataset schema
    input_schema = {
        "conditions": "sequence",  # Discrete medical codes
        "labs": "tensor",  # Continuous values
    }
    output_schema = {"label": "binary"}
    
    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="synthetic_example"
    )
    
    return dataset


def main():
    """Main execution function."""
    print("="*70)
    print("Layer-wise Relevance Propagation (LRP) - Simple Example")
    print("="*70)
    
    # Step 1: Create synthetic dataset
    print("\n[1] Creating synthetic dataset...")
    dataset = create_synthetic_dataset()
    print(f"✓ Total samples: {len(dataset)}")
    print(f"  Input schema: {dataset.input_schema}")
    print(f"  Output schema: {dataset.output_schema}")
    
    # Step 2: Split dataset
    print("\n[2] Splitting dataset...")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        dataset, [0.7, 0.15, 0.15]
    )
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Step 3: Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    
    # Step 4: Initialize MLP model
    print("\n[3] Initializing MLP model...")
    model = MLP(
        dataset=dataset,
        embedding_dim=64,
        hidden_dim=64,
        dropout=0.3
    )
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Step 5: Train model (brief training for demonstration)
    print("\n[4] Training model...")
    trainer = Trainer(
        model=model,
        device="cpu",
        metrics=["pr_auc", "roc_auc", "accuracy"]
    )
    
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,  # Just a few epochs for demonstration
        monitor="roc_auc"
    )
    print("✓ Training complete")
    
    # Step 6: LRP Interpretability
    print("\n" + "="*70)
    print("LAYER-WISE RELEVANCE PROPAGATION (LRP)")
    print("="*70)
    
    # Get a single test sample
    sample_batch = next(iter(test_loader))
    print("\n[5] Analyzing sample:")
    print(f"  Patient ID: {sample_batch['patient_id'][0]}")
    print(f"  True label: {sample_batch['label'][0].item()}")
    
    # Get model prediction
    with torch.no_grad():
        output = model(**sample_batch)
        predicted_prob = output["y_prob"][0, 0].item()
        predicted_class = int(predicted_prob > 0.5)
        logit = output["logit"][0, 0].item()
    
    print(f"  Predicted class: {predicted_class}")
    print(f"  Predicted probability: {predicted_prob:.4f}")
    print(f"  Logit (raw output): {logit:.4f}")
    
    # Step 7: Compute LRP attributions with epsilon-rule
    print("\n[6] Computing LRP attributions (ε-rule)...")
    lrp_epsilon = LayerWiseRelevancePropagation(
        model,
        rule="epsilon",
        epsilon=0.01
    )
    
    attributions_epsilon = lrp_epsilon.attribute(**sample_batch, target_class_idx=1)
    
    print("\n--- LRP Results (ε-rule) ---")
    total_relevance = 0
    for feature_key, relevance in attributions_epsilon.items():
        feature_sum = relevance.sum().item()
        total_relevance += feature_sum
        
        print(f"\n{feature_key}:")
        print(f"  Shape: {relevance.shape}")
        print(f"  Sum of relevances: {feature_sum:.4f}")
        print(f"  Mean: {relevance.mean().item():.4f}")
        print(f"  Range: [{relevance.min().item():.4f}, {relevance.max().item():.4f}]")
        
        # Show top-5 most relevant features
        flat_rel = relevance.flatten()
        if len(flat_rel) > 0:
            top_k = min(5, len(flat_rel))
            top_indices = torch.topk(flat_rel, k=top_k)
            print(f"  Top-{top_k} most relevant indices: {top_indices.indices.tolist()}")
            print(f"  Top-{top_k} relevance values: {[f'{v:.4f}' for v in top_indices.values.tolist()]}")
    
    print(f"\nTotal relevance across all features: {total_relevance:.4f}")
    print(f"Model output (logit): {logit:.4f}")
    print(f"Relevance conservation: {abs(total_relevance - logit) < 0.1}")
    
    # Step 8: Compare with alphabeta-rule
    print("\n[7] Computing LRP attributions (αβ-rule)...")
    lrp_alphabeta = LayerWiseRelevancePropagation(
        model,
        rule="alphabeta",
        alpha=1.0,
        beta=0.0  # Only consider positive contributions
    )
    
    attributions_alphabeta = lrp_alphabeta.attribute(**sample_batch, target_class_idx=1)
    
    print("\n--- LRP Results (αβ-rule) ---")
    total_relevance_ab = 0
    for feature_key, relevance in attributions_alphabeta.items():
        feature_sum = relevance.sum().item()
        total_relevance_ab += feature_sum
        
        print(f"\n{feature_key}:")
        print(f"  Sum of relevances: {feature_sum:.4f}")
        print(f"  Mean: {relevance.mean().item():.4f}")
        print(f"  Range: [{relevance.min().item():.4f}, {relevance.max().item():.4f}]")
    
    print(f"\nTotal relevance across all features: {total_relevance_ab:.4f}")
    print(f"Model output (logit): {logit:.4f}")
    
    # Step 9: Comparison summary
    print("\n" + "="*70)
    print("SUMMARY: ε-rule vs αβ-rule")
    print("="*70)
    print(f"Model output (target class logit): {logit:.4f}")
    print(f"ε-rule total relevance: {total_relevance:.4f}")
    print(f"αβ-rule total relevance: {total_relevance_ab:.4f}")
    print("\nNote: LRP relevances should sum to approximately the model's output.")
    print("The ε-rule is more stable, while αβ-rule can produce sharper attributions.")
    
    print("\n" + "="*70)
    print("Example complete!")
    print("="*70)


if __name__ == "__main__":
    main()
