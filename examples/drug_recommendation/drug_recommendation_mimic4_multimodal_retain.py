"""
Drug Recommendation on MIMIC-IV with MultimodalRETAIN

This example demonstrates how to use the MultimodalRETAIN model with mixed
input modalities for drug recommendation on MIMIC-IV.

The MultimodalRETAIN model can handle:
- Sequential features (visit histories with diagnoses, procedures) → RETAIN processing
  with reverse time attention mechanism
- Non-sequential features (demographics, static measurements) → Direct embedding

This example shows:
1. Loading MIMIC-IV data with mixed feature types
2. Applying a drug recommendation task
3. Training a MultimodalRETAIN model with both sequential and non-sequential inputs
4. Evaluating the model performance
5. Comparing to vanilla RETAIN (sequential only)
"""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import MultimodalRETAIN
from pyhealth.tasks import DrugRecommendationMIMIC4
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load MIMIC-IV base dataset
    print("=" * 60)
    print("STEP 1: Loading MIMIC-IV Dataset")
    print("=" * 60)
    
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,  # Use development mode for faster testing
        num_workers=4,
    )
    base_dataset.stats()

    # STEP 2: Apply drug recommendation task with multimodal features
    print("\n" + "=" * 60)
    print("STEP 2: Setting Drug Recommendation Task")
    print("=" * 60)
    
    # Use the DrugRecommendationMIMIC4 task
    # This task creates visit-level nested sequences from diagnoses/procedures
    # and recommends drugs for the current visit
    task = DrugRecommendationMIMIC4()
    sample_dataset = base_dataset.set_task(
        task,
        num_workers=4,
    )
    
    print(f"\nTotal samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")
    
    # Inspect a sample
    if len(sample_dataset) > 0:
        sample = sample_dataset[0]
        print("\nSample structure:")
        print(f"  Patient ID: {sample['patient_id']}")
        for key in sample_dataset.input_schema.keys():
            if key in sample:
                if isinstance(sample[key], (list, tuple)):
                    if sample[key] and isinstance(sample[key][0], (list, tuple)):
                        print(f"  {key}: {len(sample[key])} visits")
                    else:
                        print(f"  {key}: length {len(sample[key])}")
                else:
                    print(f"  {key}: {type(sample[key])}")
        # Show drugs key from output
        if 'drugs' in sample:
            print(f"  drugs (target): {len(sample['drugs'])} prescriptions")

    # STEP 3: Split dataset
    print("\n" + "=" * 60)
    print("STEP 3: Splitting Dataset")
    print("=" * 60)
    
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    # STEP 4: Initialize MultimodalRETAIN model
    print("\n" + "=" * 60)
    print("STEP 4: Initializing MultimodalRETAIN Model")
    print("=" * 60)
    
    model = MultimodalRETAIN(
        dataset=sample_dataset,
        embedding_dim=128,
        dropout=0.5,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    
    # Print feature classification
    print(f"\nSequential features (RETAIN processing): {model.sequential_features}")
    print(f"Non-sequential features (direct embedding): {model.non_sequential_features}")
    
    # Calculate expected embedding dimensions
    total_dim = len(model.feature_keys) * model.embedding_dim
    print(f"\nPatient representation dimension: {total_dim}")

    # STEP 5: Train the model
    print("\n" + "=" * 60)
    print("STEP 5: Training Model")
    print("=" * 60)
    
    trainer = Trainer(
        model=model,
        device="cuda:0",  # Change to "cpu" if no GPU available
        metrics=["pr_auc_samples", "roc_auc_samples", "jaccard_samples", "f1_samples"],
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=10,
        monitor="jaccard_samples",
        optimizer_params={"lr": 1e-3},
    )

    # STEP 6: Evaluate on test set
    print("\n" + "=" * 60)
    print("STEP 6: Evaluating on Test Set")
    print("=" * 60)
    
    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # STEP 7: Demonstrate model predictions
    print("\n" + "=" * 60)
    print("STEP 7: Sample Predictions")
    print("=" * 60)
    
    import torch
    
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        output = model(**sample_batch)

    print(f"\nBatch size: {output['y_prob'].shape[0]}")
    print(f"Output shape: {output['y_prob'].shape}")
    print(f"(batch_size, num_drug_types)")
    
    # Show first patient predictions
    print(f"\nFirst patient top-5 drug recommendations:")
    first_patient_probs = output['y_prob'][0]
    top5_drugs = torch.topk(first_patient_probs, k=min(5, len(first_patient_probs)))
    for i, (drug_idx, prob) in enumerate(zip(top5_drugs.indices, top5_drugs.values)):
        print(f"  {i+1}. Drug index {drug_idx.item()}: probability {prob.item():.4f}")
    
    # Show ground truth for first patient
    print(f"\nFirst patient ground truth drugs:")
    first_patient_true = output['y_true'][0]
    true_drug_indices = torch.where(first_patient_true > 0)[0]
    print(f"  Number of prescribed drugs: {len(true_drug_indices)}")
    if len(true_drug_indices) > 0:
        print(f"  Drug indices: {true_drug_indices.tolist()[:10]}...")

    # STEP 8: Compare with vanilla RETAIN (if applicable)
    print("\n" + "=" * 60)
    print("STEP 8: Model Architecture Comparison")
    print("=" * 60)
    
    print("\nMultimodalRETAIN vs. Vanilla RETAIN:")
    print("  Vanilla RETAIN:")
    print("    - Only handles sequential (visit-level) features")
    print("    - Processes all features through reverse time attention")
    print("  ")
    print("  MultimodalRETAIN:")
    print("    - Handles both sequential and non-sequential features")
    print(f"    - Sequential features ({len(model.sequential_features)}): "
          f"{model.sequential_features}")
    print(f"    - Non-sequential features ({len(model.non_sequential_features)}): "
          f"{model.non_sequential_features}")
    print("    - More flexible for heterogeneous EHR data")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: MultimodalRETAIN Training Complete")
    print("=" * 60)
    print(f"Model: MultimodalRETAIN")
    print(f"Dataset: MIMIC-IV")
    print(f"Task: Drug Recommendation")
    print(f"Sequential features: {len(model.sequential_features)}")
    print(f"Non-sequential features: {len(model.non_sequential_features)}")
    print(f"Best validation Jaccard: {results.get('jaccard_samples', 0):.4f}")
    print("\nRETAIN advantages:")
    print("  - Reverse time attention for interpretability")
    print("  - Visit-level attention weights (alpha)")
    print("  - Variable-level attention weights (beta)")
    print("  - Multimodal extension allows richer feature sets")
    print("=" * 60)

