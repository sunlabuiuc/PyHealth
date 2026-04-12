"""
Demo script for WatchSleepNet on the DREAMT dataset.

This example demonstrates how to:
1. Load the DREAMT dataset
2. Access patient data and wearable signal files
3. Set the sleep staging task using SleepStagingDREAMT
4. Split by patient and create DataLoaders
5. Train and evaluate the WatchSleepNet model

Dataset: DREAMT (Dataset for Real-time sleep stage EstimAtion using
Multisensor wearable Technology) 

Note: Update the `root` path below to point to your local DREAMT download.
"""

import torch
import torch.nn as nn

from pyhealth.datasets import DREAMTDataset, get_dataloader, split_by_patient
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingDREAMT

if __name__ == "__main__":

    # Initialize DREAMT dataset
    DREAMT_ROOT = "/Users/cpenquit/Desktop/dreamt/2.1.0" # Update this path to your local DREAMT download
    dataset = DREAMTDataset(root=DREAMT_ROOT)

    print("=" * 70)
    print("Loading DREAMT Dataset")
    print("=" * 70)
    dataset.stats()
    print()

    # Get list of unique patient IDs
    patient_ids = dataset.unique_patient_ids
    print(f"Total number of patients: {len(patient_ids)}")
    print(f"Sample patient IDs: {patient_ids[:5]}")
    print()

    # Access individual patient data
    print("=" * 70)
    print("Patient Data Example")
    print("=" * 70)
    first_patient_id = patient_ids[0]
    patient = dataset.get_patient(first_patient_id)

    print(f"Patient ID: {patient.patient_id}")
    print(f"Number of events: {len(patient.data_source)}")
    print()

    # Access clinical attributes
    event = patient.get_events(event_type="dreamt_sleep")[0]
    print("Clinical attributes:")
    print(f"  Age: {event.age}")
    print(f"  Gender: {event.gender}")
    print(f"  BMI: {event.bmi}")
    print(f"  AHI: {event.ahi}")
    print(f"  64Hz data file: {event.file_64hz}")
    print(f"  100Hz data file: {event.file_100hz}")
    print()

    # Set sleep staging task
    print("=" * 70)
    print("Setting Sleep Staging Task")
    print("=" * 70)

    task = SleepStagingDREAMT()
    sample_dataset = dataset.set_task(task=task)

    print(f"Generated {len(sample_dataset)} samples")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")
    print()

    # split by patient
    print("=" * 70)
    print("Splitting Data by Patient")
    print("=" * 70)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.6, 0.2, 0.2]
    )

    # create data loaders
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val:   {len(val_dataset)} samples")
    print(f"Test:  {len(test_dataset)} samples")
    print()

    # Define and train WatchSleepNet

    print("=" * 70)
    print("Training WatchSleepNet")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = WatchSleepNet(dataset=sample_dataset).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch_signals = batch["signal"].to(device)
            batch_labels = batch["label"].to(device)

            optimizer.zero_grad()
            output = model(batch_signals)

            # Use prediction from the last epoch in the sequence to match label
            last_epoch_output = output[:, -1, :]
            loss = criterion(last_epoch_output, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_signals.size(0)
            preds = last_epoch_output.argmax(dim=-1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_labels.size(0)

        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_signals = batch["signal"].to(device)
                batch_labels = batch["label"].to(device)
                output = model(batch_signals)
                preds = output[:, -1, :].argmax(dim=-1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1:2d}/{num_epochs}  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

    print()

    # Evaluate on test set

    print("=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)

    stage_names = {0: "Wake", 1: "Light Sleep", 2: "Deep Sleep"}

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in test_loader:
            batch_signals = batch["signal"].to(device)
            batch_labels = batch["label"].to(device)
            output = model(batch_signals)
            all_preds.append(output[:, -1, :].argmax(dim=-1).cpu())
            all_true.append(batch_labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()

    test_acc = (all_preds == all_true).mean()
    print(f"Test Accuracy: {test_acc:.4f}")
    print()

    # Per-class accuracy
    for cls_id, cls_name in stage_names.items():
        mask = all_true == cls_id
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == cls_id).mean()
            print(f"  {cls_name}: {cls_acc:.4f} ({mask.sum()} samples)")

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
