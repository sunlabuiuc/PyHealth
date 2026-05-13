from pyhealth.datasets import Wav2SleepDataset
from pyhealth.datasets import get_dataloader, split_by_sample
from pyhealth.tasks import Wav2SleepStaging
from pyhealth.models import Wav2Sleep
from pyhealth.trainer import Trainer

import os
import tempfile
import numpy as np
import mne
import xml.etree.ElementTree as ET


def create_mock_data(tmp_dir: str, n_patients: int = 8) -> str:
    """Generate synthetic PSG data matching the Wav2SleepDataset structure.

    Writes fake EDF + XML annotation pairs so the example runs without
    requiring access to restricted datasets (SHHS, MESA, etc.).
    Uses the SHHS dataset structure:
        shhs/polysomnography/edfs/shhs1/{patient_id}.edf
        shhs/polysomnography/annotations-events-profusion/shhs1/{patient_id}-profusion.xml
    """
    sfreq = 128
    duration_s = 30 * 10  # 10 epochs of 30s
    n_samples = int(sfreq * duration_s)
    ch_names = ["ECG", "THOR RES", "ABDO RES"]
    ch_types = ["ecg", "bio", "bio"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    edf_dir = os.path.join(tmp_dir, "shhs", "polysomnography", "edfs", "shhs1")
    label_dir = os.path.join(
        tmp_dir, "shhs", "polysomnography", "annotations-events-profusion", "shhs1"
    )
    os.makedirs(edf_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for i in range(n_patients):
        patient_id = f"shhs1-{i:06d}"

        # Fake EDF
        data = np.random.randn(len(ch_names), n_samples).astype(np.float32)
        raw = mne.io.RawArray(data, info, verbose=False)
        edf_path = os.path.join(edf_dir, f"{patient_id}.edf")
        mne.export.export_raw(edf_path, raw, fmt="edf", verbose=False)

        # Fake XML annotation (10 epochs, random stages 0-5, no -1 unscored)
        root_el = ET.Element("CMPStudyConfig")
        staging_el = ET.SubElement(root_el, "SleepStages")
        for stage in np.random.choice([0, 1, 2, 3, 5], size=10):
            el = ET.SubElement(staging_el, "SleepStage")
            el.text = str(stage)
        xml_path = os.path.join(label_dir, f"{patient_id}-profusion.xml")
        ET.ElementTree(root_el).write(xml_path)

    return tmp_dir


def example(root: str = None):
    """Demonstrate the full wav2sleep pipeline integrated with PyHealth.

    Runs Dataset → Task → Model → Trainer using either real PSG data from
    sleepdata.org or synthetic mock data if no root is provided.

    Args:
        root: Path to a directory containing PSG data in the Wav2SleepDataset
              structure. If None, synthetic mock data is generated automatically.
              To obtain real data, complete a Data Use Agreement at sleepdata.org.
    """

    if root is None:
        tmp_dir = tempfile.mkdtemp()
        root = create_mock_data(tmp_dir)
        print(f"No --root provided. Running on synthetic mock data at {tmp_dir}")

    dataset = Wav2SleepDataset(root)
    task = Wav2SleepStaging()

    samples = dataset.set_task(task)
    train_dataset, val_dataset, test_dataset = split_by_sample(
        samples, [0.5, 0.25, 0.25]
    )
    train_loader = get_dataloader(train_dataset, batch_size=2, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=2, shuffle=False)

    wav2sleep = Wav2Sleep(samples)

    trainer = Trainer(model=wav2sleep)
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=1)
    trainer.evaluate(test_loader)


def demo_ablation(root: str = None):
    """Demonstrate the wav2sleep stochastic signal masking ablation.

    Replicates the ablation study from Section 4.3 of the wav2sleep paper,
    which investigates model robustness when signals are randomly dropped
    during training. This is controlled via custom per-signal masking
    probabilities passed to Wav2Sleep.

    Args:
        root: Path to a directory containing PSG data in the Wav2SleepDataset
              structure. If None, synthetic mock data is generated automatically.
              To obtain real data, complete a Data Use Agreement at sleepdata.org.
    """

    if root is None:
        tmp_dir = tempfile.mkdtemp()
        root = create_mock_data(tmp_dir)
        print(f"No --root provided. Running on synthetic mock data at {tmp_dir}")

    dataset = Wav2SleepDataset(root)
    task = Wav2SleepStaging()

    samples = dataset.set_task(task)
    train_dataset, val_dataset, test_dataset = split_by_sample(
        samples, [0.5, 0.25, 0.25]
    )
    train_loader = get_dataloader(train_dataset, batch_size=2, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=2, shuffle=False)

    # ABLATION Create a custom set of masking probabilities
    mask_probabilities = {"ECG": 0.5, "PPG": 0.5, "THX": 0.5, "ABD": 0.5}

    wav2sleep = Wav2Sleep(samples, stochastic_mask_probabilities=mask_probabilities)

    trainer = Trainer(model=wav2sleep)
    trainer.train(train_dataloader=train_loader, val_dataloader=val_loader, epochs=1)
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    # if you have Data Access from sleepdata.org, you can download and use the datasets
    root = "../../full_sample_PSG"
    # otherwise, rely on mock data
    root = root if os.path.isdir(root) else None
    example(root)
    demo_ablation(root)
