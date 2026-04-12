"""
Demo script for WatchSleepNet on the SHHS dataset.

This example demonstrates how to:
1. Load the SHHS dataset
2. Access patient data and wearable signal files
3. Set the sleep staging task using SleepStagingSHHS
4. Split by patient and create DataLoaders
5. Train and evaluate the WatchSleepNet model

Dataset: SHHS (Sleep Heart Health Study)

Note: Update the `root` path below to point to your local SHHS download.
"""

from pyhealth.trainer import Trainer
from pyhealth.datasets import SHHSDataset, get_dataloader, split_by_patient
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingSHHS

_EPOCHS = 10
_DECAY_WEIGHT = 1e-5

if __name__ == "__main__":

    # Initialize SHHS dataset
    SHHS_ROOT = "/path/to/shhs"  # Update this path to your local SHHS download
    dataset = SHHSDataset(root=SHHS_ROOT)

    print("=" * 70)
    print("Loading SHHS Dataset")
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
    event = patient.get_events(event_type="shhs_sleep")[0]
    print("Clinical attributes:")
    print(f"  Visit: {event.visitnumber}")
    print(f"  Age: {event.age}")
    print(f"  Sex: {event.sex}")
    print(f"  BMI: {event.bmi}")
    print(f"  AHI: {event.ahi}")
    print(f"  Signal file: {event.signal_file}")
    print(f"  Annotation file: {event.annotation_file}")
    print(f"  ECG sample rate: {event.ecg_sample_rate} Hz")
    print()

    # Set sleep staging task
    print("=" * 70)
    print("Setting Sleep Staging Task")
    print("=" * 70)

    task = SleepStagingSHHS()
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
    print("Train and Test WatchSleepNet")
    print("=" * 70)

    model = WatchSleepNet(
        dataset=sample_dataset,
        lstm_hidden_size=128 # default hidden size for WatchSleepNet as used in the original paper
        # lstm_hidden_size=64 # ablation with smaller hidden size as there are only 3 classes in this task (Wake, NREM, REM)
        # lstm_hidden_size=256 # ablation with larger hidden size to test if it improves performance on this task
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    trainer = Trainer(
        model=model,
        metrics=["cohen_kappa", "f1_macro", "f1_weighted", "accuracy"],
        exp_name="watchsleepnet_sleep_staging"
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=_EPOCHS,
        monitor="cohen_kappa",
        monitor_criterion="max",
        weight_decay=_DECAY_WEIGHT
    )

    scores = trainer.evaluate(test_loader)
    print()
    print("=" * 70)
    print("Results on Test Set")
    print("=" * 70)
    print(
        f"Cohen's Kappa: {scores['cohen_kappa']:.4f}\n"
        f"F1 Macro: {scores['f1_macro']:.4f}\n"
        f"F1 Weighted: {scores['f1_weighted']:.4f}\n"
        f"Accuracy: {scores['accuracy']:.4f}\n"
        f"Loss: {scores['loss']:.4f}"
    )

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
