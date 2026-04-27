"""
Author(s): Bronze Frazer
NetID(s):  bfrazer2
Paper:     wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals
Link:      https://arxiv.org/abs/2411.04644
Desc:      PyHealth Task for sleep stage classification using wav2sleep
"""

import logging
from typing import Any, Dict

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

import mne
import numpy as np
import pandas as pd
import scipy
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class Wav2SleepStaging(BaseTask):
    """Multi-class sleep stage classification from heterogenous biosignal data

    This task prepares biosignal and annotation data from multiple
    polysomnography datasets.

    Attributes:
        task_name: The name of the task. Set to "Wav2SleepStaging"
        input_schema: Schema for the task input
        output_schema: Schema for the task output
        XML_STAGE_MAP: A label map for annotaion-events-profusion files to expected labels
        WSC_STAGE_MAP: A label map for WSC dataset annotations to expected labels
        CHANNEL_MAPS: A map to extract the expected signal names as recorded in the EDF for a dataset

    Examples:
        >>> from pyhealth.datasets import Wav2SleepDataset
        >>> from pyhealth.tasks import Wav2SleepStaging
        >>> wav2sleep_dataset = Wav2SleepDataset(root = "path/to/root")
        >>> task = Wav2SleepStaging()
        >>> samples = wav2sleep_dataset.set_task(task)
    """

    task_name: str = "Wav2SleepStaging"
    input_schema: Dict[str, str] = {
        "ECG": "tensor",
        "PPG": "tensor",
        "THX": "tensor",
        "ABD": "tensor",
        "availability_mask": "tensor",
    }
    output_schema: Dict[str, str] = {"stages": "tensor"}

    # 0=Wake, 1=Light(N1+N2), 2=Deep(N3), 3=REM, -1=Unscored
    XML_STAGE_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 9: -1}
    WSC_STAGE_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 2, 5: 3, 7: -1}

    # Mappings for the different names of biosignal across the datasets
    CHANNEL_MAPS: Dict[str, Dict[str, Any]] = {
        "shhs": {"ECG": "ECG", "PPG": None, "THX": "THOR RES", "ABD": "ABDO RES"},
        "mesa": {"ECG": "EKG", "PPG": "Pleth", "THX": "Thor", "ABD": "Abdo"},
        "cfs": {
            "ECG": "ECG1",
            "PPG": "PlethWV",
            "THX": "THOR EFFORT",
            "ABD": "ABDO EFFORT",
        },
        "chat": {"ECG": "ECG1", "PPG": "PlethNellcor", "THX": "Chest", "ABD": "ABD"},
        "mros": {"ECG": "ECG L", "PPG": None, "THX": "Thoracic", "ABD": "Abdominal"},
        "ccshs": {
            "ECG": "ECG1",
            "PPG": "PlethWV",
            "THX": "THOR EFFORT",
            "ABD": "ABDO EFFORT",
        },
        "wsc": {"ECG": "ECG", "PPG": None, "THX": "thorax", "ABD": "abdomen"},
    }

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, patient: Patient) -> list[Dict[str, Any]]:
        """Process patient polysomnography biosignals for sleep stage prediction

        Args:
            patient: A patient object containing biosignals (ECG, PPG, THX, ABD),
                     a mask to indicate which signals are available,
                     and sleep stage annotations

        Returns:
            list[Dict[str, Any]]: A list of samples for a patient, where each sample
                is a dict containing the patient's id, their preprocessed biosignals (ECG, PPG, THX, ABD),
                an availability mask for the signals, and ground-truth sleep stages
        """

        samples = []
        for event in patient.get_events():
            signals, availability_mask = self.load_signals(
                event.edf_path, event.source_dataset
            )
            stages = self.load_stages(event.label_path, event.source_dataset)
            sample = {
                "patient_id": patient.patient_id,
                **signals,
                "availability_mask": availability_mask,
                "stages": stages,
            }
            samples.append(sample)

        return samples

    def load_signals(
        self, edf_path: str, dataset: str
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """Extract raw biosignals (EDF, PPG, THX, ABD) from and EDF file

        Args:
            edf_path: Path to the EDF file
            dataset: name of the source dataset (e.g. SHHS, CFS, ...)

        Returns:
            tuple[Dict[str, np.ndarray], np.ndarray]:
                A dictionary of raw biosignals, and a boolean array indicating
                which signals were not found in a recording
        """
        signals = {"ECG": None, "PPG": None, "THX": None, "ABD": None}

        channel_map = self.CHANNEL_MAPS[dataset]
        src_names = [src for src in channel_map.values() if src is not None]

        raw = mne.io.read_raw_edf(
            edf_path, include=src_names, preload=True, verbose=False
        )

        src_to_canonical = {
            src: sig for sig, src in channel_map.items() if src is not None
        }

        data = raw.get_data()

        signals.update(
            {
                src_to_canonical[ch]: data[i]
                for i, ch in enumerate(raw.ch_names)
                if ch in src_to_canonical
            }
        )

        original_sample_rate = raw.info["sfreq"]
        availability_mask = []
        for signal_type, signal in signals.items():
            mask_value = signal is None
            availability_mask.append(mask_value)
            preprocessed_signal = self.preprocess_signal(
                signal_type, signal, original_sample_rate
            )
            signals[signal_type] = preprocessed_signal

        return signals, availability_mask

    def load_stages(self, annotation_path: str, dataset: str) -> np.ndarray:
        """Parse ground-truth sleep stage labels from an annotation file

        The categories are:
            0  = Wake
            1  = Light(N1+N2)
            2  = Deep(N3)
            3  = REM
            -1 = Unscored

        Args:
            annotation_path: path to a sleep stage annotation file
            dataset: name of the source dataset (e.g. SHHS, CFS, ...)

        Returns:
            np.ndarray: An integer array of ground truth sleep stage cateogies
        """

        def load_stages_xml(xml_path: str) -> np.ndarray:
            """Parse the XML annotation format"""
            root = ET.parse(xml_path).getroot()
            stages = [self.XML_STAGE_MAP[int(e.text)] for e in root.iter("SleepStage")]
            return np.array(stages, dtype=np.int8)

        def load_stages_wsc(stg_path: str) -> np.ndarray:
            """Parse the TXT annotation format (only used for the WSC dataset)"""
            df = pd.read_csv(stg_path, sep="\t")
            return (
                df["User-Defined Stage"].map(self.WSC_STAGE_MAP).to_numpy(dtype=np.int8)
            )

        STAGE_LOADERS = {
            "shhs": load_stages_xml,
            "mesa": load_stages_xml,
            "cfs": load_stages_xml,
            "chat": load_stages_xml,
            "mros": load_stages_xml,
            "ccshs": load_stages_xml,
            "wsc": load_stages_wsc,
        }

        stages = STAGE_LOADERS[dataset](annotation_path)
        fixed_epoch_stages = self._pad_or_truncate(
            stages, is_label=True, target_length=1200
        )  # T=1200 epochs
        return fixed_epoch_stages

    def preprocess_signal(
        self, signal_type: str, signal: np.ndarray, original_sample_rate: float
    ) -> np.ndarray:
        """Pre-processing for raw biosignal data

        The steps are outlined in section 4.1. of wav2sleep
            Pad or truncate to 10 hours (T = 1200)
            Resample each biosignal to target frequency
                (~34Hz for ECG & PPG, ~8Hz for THX & ABD)
            Apply unit normalisation

        Args:
            signal_type: name of the biosignal to be preprocessed
            signal: the raw biosignal
            original_sample_rate: the sample rate (in Hertz) of the EDF recording

        Returns:
            np.ndarray: A preprocessed biosignal
        """
        T = 1200  # target number of epochs
        seconds_per_epoch = 30  # epochs are 30 seconds each

        # k = total number of measurements to retain per epoch
        if signal_type in ["ECG", "PPG"]:
            k = 1024
        elif signal_type in ["ABD", "THX"]:
            k = 256

        target_raw_samples = T * seconds_per_epoch * original_sample_rate
        target_output_samples = T * k

        # return a zeros array of target_output_samples if a signal is None
        # (signal_type is not present in the dataset)
        if signal is None:
            return np.zeros(target_output_samples, dtype=np.float32)

        # Step 1: Pad or truncate to exactly 10 hours
        signal = self._pad_or_truncate(
            signal, is_label=False, target_length=int(target_raw_samples)
        )
        # Step 2: Resample to target frequency
        signal = scipy.signal.resample(signal, target_output_samples)
        # Step 3: Unit normalization
        signal = (signal - signal.mean()) / signal.std()

        return signal

    def _pad_or_truncate(
        self, array: np.ndarray, is_label: bool, target_length: int
    ) -> np.ndarray:
        """Pad or truncate an array to a desired size

        Args:
            array: an array to be padded/truncated
            is_label: A flag indicating if `array` should be treated as a list of labels
            target_length: desired length of the array

        Returns:
            np.ndarray: A padded or truncated array
        """
        if len(array) >= target_length:
            return array[:target_length]
        pad_length = target_length - len(array)
        # zero-pad signal arrays, but pad label arrays with -1 (unscored)
        constant_values = -1 if is_label else 0
        return np.pad(
            array, (0, pad_length), mode="constant", constant_values=constant_values
        )


if __name__ == "__main__":
    from pyhealth.datasets import Wav2SleepDataset
    from pyhealth.tasks import Wav2SleepStaging

    wav2sleep_dataset = Wav2SleepDataset(root="../../../full_sample_PSG/")
    task = Wav2SleepStaging()
    sample_dataset = wav2sleep_dataset.set_task(task)

    # test the DataLoader and discover shape of inputs to model
    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(sample_dataset, batch_size=7, shuffle=False)
    data_batch = next(iter(train_loader))
    print(data_batch)
