"""
Mortality Prediction on MIMIC-IV with MultimodalRNN

This example demonstrates how to use the MultimodalRNN model with mixed
input modalities for in-hospital mortality prediction on MIMIC-IV.

The MultimodalRNN model can handle:
- Sequential features (diagnoses, procedures, lab timeseries) → RNN processing
- Non-sequential features (demographics, static measurements) → Direct embedding

This example shows:
1. Loading MIMIC-IV data with mixed feature types
2. Applying a mortality prediction task
3. Training a MultimodalRNN model with both sequential and non-sequential inputs
4. Evaluating the model performance
"""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import MultimodalRNN
from pyhealth.tasks import InHospitalMortalityMIMIC4
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load MIMIC-IV base dataset
    print("=" * 60)
    print("STEP 1: Loading MIMIC-IV Dataset")
    print("=" * 60)
    
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        dev=True,  # Use development mode for faster testing
        num_workers=4,
    )
    base_dataset.stats()

    # STEP 2: Apply mortality prediction task with multimodal features
    print("\n" + "=" * 60)
    print("STEP 2: Setting Mortality Prediction Task")
    print("=" * 60)
    
    # Use the InHospitalMortalityMIMIC4 task
    # This task will create sequential features from diagnoses, procedures, and labs
    task = InHospitalMortalityMIMIC4()
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
                    print(f"  {key}: length {len(sample[key])}")
                else:
                    print(f"  {key}: {type(sample[key])}")
        print(f"  Mortality: {sample.get('mortality', 'N/A')}")

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

    # STEP 4: Initialize MultimodalRNN model
    print("\n" + "=" * 60)
    print("STEP 4: Initializing MultimodalRNN Model")
    print("=" * 60)
    
    model = MultimodalRNN(
        dataset=sample_dataset,
        embedding_dim=128,
        hidden_dim=128,
        rnn_type="GRU",
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params:,} parameters")
    
    # Print feature classification
    print(f"\nSequential features (RNN processing): {model.sequential_features}")
    print(f"Non-sequential features (direct embedding): {model.non_sequential_features}")
    
    # Calculate expected embedding dimensions
    seq_dim = len(model.sequential_features) * model.hidden_dim
    non_seq_dim = len(model.non_sequential_features) * model.embedding_dim
    total_dim = seq_dim + non_seq_dim
    print(f"\nPatient representation dimension:")
    print(f"  Sequential contribution: {seq_dim}")
    print(f"  Non-sequential contribution: {non_seq_dim}")
    print(f"  Total: {total_dim}")

    # STEP 5: Train the model
    print("\n" + "=" * 60)
    print("STEP 5: Training Model")
    print("=" * 60)
    
    trainer = Trainer(
        model=model,
        device="cuda:0",  # Change to "cpu" if no GPU available
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=10,
        monitor="roc_auc",
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
    print(f"First 10 predicted probabilities:")
    for i, (prob, true_label) in enumerate(
        zip(output['y_prob'][:10], output['y_true'][:10])
    ):
        print(f"  Sample {i+1}: prob={prob.item():.4f}, true={int(true_label.item())}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: MultimodalRNN Training Complete")
    print("=" * 60)
    print(f"Model: MultimodalRNN")
    print(f"Dataset: MIMIC-IV")
    print(f"Task: In-Hospital Mortality Prediction")
    print(f"Sequential features: {len(model.sequential_features)}")
    print(f"Non-sequential features: {len(model.non_sequential_features)}")
    print(f"Best validation ROC-AUC: {max(results.get('roc_auc', 0), 0):.4f}")
    print("=" * 60)

