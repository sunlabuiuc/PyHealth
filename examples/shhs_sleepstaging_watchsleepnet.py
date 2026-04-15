"""
Demo script for WatchSleepNet on the SHHS dataset.

This example demonstrates how to:
1. Load the SHHS dataset
2. Access patient data and wearable signal files
3. Set the sleep staging task using SleepStagingSHHSIBI
4. Split by patient and create DataLoaders
5. Train and evaluate the WatchSleepNet model

Dataset: SHHS (Sleep Heart Health Study)

Note: Update the `SHHS_ROOT` path below to point to your local SHHS download.

Ablation Studies: 
    can be run as described in the comments at the bottom
    of this file. By default, the original WatchSleepNet configuration is
    used (sequence length of 20 and LSTM hidden size of 128).
"""

import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import neurokit2 as nk
import pandas as pd

from pyhealth.trainer import Trainer
from pyhealth.datasets import SHHSECGDataset, get_dataloader, split_by_patient
from pyhealth.models import WatchSleepNet
from pyhealth.tasks import SleepStagingSHHSIBI

_EPOCHS = 10
_DECAY_WEIGHT = 1e-5

# Set to None by default to use synthetic mock data instead of real SHHS data
# Update this path to your local SHHS download if you want to run on real data
SHHS_ROOT = None


def main(seq_len: int = 20, lstm_hidden_size: int = 128) -> None:

    if seq_len != 20:
        print("=" * 70)
        print(f"Using sequence length of {seq_len} for ablation study")
        print("=" * 70)
        print()
    if lstm_hidden_size != 128:
        print("=" * 70)
        print(f"Using LSTM hidden size of {lstm_hidden_size} for ablation study")
        print("=" * 70)
        print()

    if SHHS_ROOT is not None:
        dataset = SHHSECGDataset(root=SHHS_ROOT)
    else:
        print("=" * 70)
        print("SHHS root not set, using synthetic sample data")
        print("=" * 70)
        _tmpdir = _build_sample_dataset_root(num_patients=3)
        dataset = SHHSECGDataset(root=_tmpdir.name)

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

    task = SleepStagingSHHSIBI(seq_len=seq_len)
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
        lstm_hidden_size=lstm_hidden_size
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    trainer = Trainer(
        model=model,
        metrics=[
            "cohen_kappa",
            "f1_macro",
            "f1_weighted",
            "accuracy",
            "roc_auc_macro_ovr"
        ],
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
        f"Accuracy: {scores['accuracy']:.4f}\n"
        f"F1 Macro: {scores['f1_macro']:.4f}\n"
        f"F1 Weighted: {scores['f1_weighted']:.4f}\n"
        f"AUROC: {scores['roc_auc_macro_ovr']:.4f}\n"
        f"Cohen's Kappa: {scores['cohen_kappa']:.4f}\n"
        f"Loss: {scores['loss']:.4f}"
    )

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


def _write_edf(path: str, signal: np.ndarray, sample_rate: int,
               ch_name: str = "ECG") -> None:
    """Write a minimal single-channel EDF file without external dependencies.

    Args:
        path: Destination file path for the EDF file.
        signal: 1-D signal array to write.
        sample_rate: Sampling frequency of the signal in Hz.
        ch_name: Channel label to embed in the EDF header. Default is "ECG".
    """
    record_duration = 1  # seconds per data record
    samples_per_record = sample_rate * record_duration
    n_records = len(signal) // samples_per_record
    signal = signal[: n_records * samples_per_record]

    phys_min = float(signal.min())
    phys_max = float(signal.max())
    if phys_min == phys_max:
        phys_min -= 1.0
        phys_max += 1.0
    dig_min, dig_max = -32768, 32767

    gain = (phys_max - phys_min) / (dig_max - dig_min)
    offset = phys_max / gain - dig_max
    digital = np.clip(
        np.round(signal / gain - offset), dig_min, dig_max
    ).astype(np.int16)

    ns = 1
    header_bytes = (ns + 1) * 256
    now = datetime.now()

    with open(path, "wb") as f:
        # Fixed 256-byte header
        f.write(b"0       ")
        f.write(f"{'X X X X':<80}".encode("ascii"))
        f.write(f"{'Startdate':<80}".encode("ascii"))
        f.write(now.strftime("%d.%m.%y").encode("ascii"))
        f.write(now.strftime("%H.%M.%S").encode("ascii"))
        f.write(f"{header_bytes:<8}".encode("ascii"))
        f.write(b" " * 44)
        f.write(f"{n_records:<8}".encode("ascii"))
        f.write(f"{record_duration:<8}".encode("ascii"))
        f.write(f"{ns:<4}".encode("ascii"))

        # Signal header (256 bytes per signal)
        f.write(f"{ch_name:<16}".encode("ascii"))
        f.write(b" " * 80)   # transducer type
        f.write(b" " * 8)    # physical dimension
        f.write(f"{phys_min:<8.4g}".encode("ascii"))
        f.write(f"{phys_max:<8.4g}".encode("ascii"))
        f.write(f"{dig_min:<8}".encode("ascii"))
        f.write(f"{dig_max:<8}".encode("ascii"))
        f.write(b" " * 80)   # prefiltering
        f.write(f"{samples_per_record:<8}".encode("ascii"))
        f.write(b" " * 32)   # reserved

        # Data records
        for i in range(n_records):
            start = i * samples_per_record
            f.write(digital[start: start + samples_per_record].tobytes())


def _write_profusion_xml(path: str, n_epochs: int) -> None:
    """Write a minimal Profusion-format XML sleep annotation file.

    Cycles through sleep stages (Wake, N1, N2, N3, REM) to produce a
    mix of labels across the recording.

    Args:
        path: Destination file path for the XML file.
        n_epochs: Number of 30-second epochs to annotate.
    """
    stage_cycle = [0, 1, 2, 3, 2, 1, 5, 5, 2, 3]
    root = ET.Element("PSGAnnotation")
    sleep_stages = ET.SubElement(root, "SleepStages")
    for i in range(n_epochs):
        ET.SubElement(sleep_stages, "SleepStage").text = str(
            stage_cycle[i % len(stage_cycle)]
        )
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")


def _build_sample_dataset_root(num_patients: int = 3) -> tempfile.TemporaryDirectory:
    """Build a temporary SHHS-like directory with synthetic EDF and XML files.

    Creates one shhs1 visit per patient with a simulated ECG signal and
    Profusion XML annotations, then writes shhs-metadata.csv so that
    SHHSDataset can load the directory without calling prepare_metadata.

    Args:
        num_patients: Number of synthetic patients to generate. Default is 3.

    Returns:
        A TemporaryDirectory object whose .name attribute is the root path.
        The caller must retain a reference to prevent early cleanup.
    """
    tmpdir = tempfile.TemporaryDirectory()
    root = Path(tmpdir.name)

    edf_dir = root / "edfs"
    xml_dir = root / "annotations"
    edf_dir.mkdir()
    xml_dir.mkdir()

    records = []
    for i in range(num_patients):
        patient_id = f"2000{i + 1:02d}"
        edf_path = edf_dir / f"shhs1-{patient_id}.edf"
        xml_path = xml_dir / f"shhs1-{patient_id}-profusion.xml"

        num_epochs = 40
        duration = num_epochs * 30  # seconds
        sample_rate = 125  # shhs1 = 125 Hz ECG, shhs2 = 256 Hz ECG
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=sample_rate,
                              noise=0.05)
        _write_edf(str(edf_path), np.array(ecg, dtype=np.float32),
                   sample_rate)
        _write_profusion_xml(str(xml_path), num_epochs)

        records.append({
            "patient_id": patient_id,
            "visitnumber": 1,
            "signal_file": str(edf_path),
            "annotation_file": str(xml_path),
            "ecg_sample_rate": sample_rate,
            "age": 50 + i,
            "sex": "Male" if i % 2 == 0 else "Female",
            "bmi": 25.0 + i,
            "ahi": 10.0 + i,
        })

    pd.DataFrame(records).to_csv(root / "shhs-metadata.csv", index=False)
    return tmpdir


if __name__ == "__main__":
    main()

    """
    Dataset Ablation: Sequence Length

    The original WatchSleepNet paper used a sequence length of 20
    (with 30-second epochs making the context window 5 minutes).

    Testing shorter (10) and longer (30) sequence lengths to see if it improves
    performance on this task. With 30-second epochs, a sequence length of
    10 corresponds to a 5-minute context window, while a sequence length of
    20 corresponds to a 10-minute context window.
    """
    # main(seq_len=10)
    # main(seq_len=20) # original WatchSleepNet configuration
    # main(seq_len=30)

    """
    Model Ablation: LSTM Hidden Size

    The original WatchSleepNet paper used an LSTM hidden size of 128.

    Testing shorter (64) and longer (256) hidden sizes to see if it improves
    performance on this task. A smaller hidden size may be sufficient given
    the small number of classes (Wake, NREM, REM), while a larger hidden size
    may allow the model to capture more complex temporal dependencies in the data.
    """
    # main(lstm_hidden_size=64)
    # main(lstm_hidden_size=128)  # original WatchSleepNet configuration
    # main(lstm_hidden_size=256)

    """
    Pipeline Ablation: Sequence Length + LSTM Hidden Size

    The original WatchSleepNet paper used an LSTM hidden size
    of 128 and sequence length of 20.

    Testing best-performing sequence length and hidden size combinations from
    the above ablations to see if there is an interaction effect between these
    two hyperparameters in comparison to the original WatchSleepNet configuration.
    """
    # main(seq_len=20, lstm_hidden_size=128) # original WatchSleepNet configuration
    # main(seq_len=10, lstm_hidden_size=64)
