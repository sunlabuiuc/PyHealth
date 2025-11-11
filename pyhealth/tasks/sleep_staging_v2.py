import os
from dataclasses import dataclass, field
from typing import Any, Dict

import mne

from pyhealth.tasks import BaseTask


@dataclass(frozen=True)
class SleepStagingSleepEDF(BaseTask):
    task_name: str = "SleepStaging"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"signal": "tensor"})
    output_schema: Dict[str, str] = field(
        default_factory=lambda: {"label": "multiclass"}
    )

    def __call__(self, patient: Any, epoch_seconds: int = 30) -> list[dict[str, Any]]:
        """Processes a single patient for the sleep staging task on Sleep EDF.

        Sleep staging aims at predicting the sleep stages (Awake, REM, N1, N2, N3, N4) based on
        the multichannel EEG signals. The task is defined as a multi-class classification.

        Args:
            patient: a list of (load_from_path, signal_file, label_file, save_to_path) tuples, where PSG is the signal files and the labels are
            in label file
            epoch_seconds: how long will each epoch be (in seconds)

        Returns:
            samples: a list of samples, each sample is a dict with patient_id, record_id,
                and epoch_path (the path to the saved epoch {"X": signal, "Y": label} as key.

        Note that we define the task as a multi-class classification task.

        Examples:
            >>> from pyhealth.datasets import SleepEDFDataset
            >>> sleepedf = SleepEDFDataset(
            ...         root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
            ...     )
            >>> from pyhealth.tasks import sleep_staging_sleepedf_fn
            >>> sleepstage_ds = sleepedf.set_task(sleep_staging_sleepedf_fn)
            >>> sleepstage_ds.samples[0]
            {
                'record_id': 'SC4001-0',
                'patient_id': 'SC4001',
                'epoch_path': '/home/chaoqiy2/.cache/pyhealth/datasets/70d6dbb28bd81bab27ae2f271b2cbb0f/SC4001-0.pkl',
                'label': 'W'
            }
        """

        SAMPLE_RATE = 100

        pid = patient.patient_id
        events = patient.get_events(event_type="recordings")

        samples = []
        for event in events:
            data = mne.io.read_raw_edf(event.signal_file)
            X = data.get_data()
            ann = mne.read_annotations(event.label_file)
            labels = []
            for dur, des in zip(ann.duration, ann.description):
                """
                all possible des:
                    - 'Sleep stage W'
                    - 'Sleep stage 1'
                    - 'Sleep stage 2'
                    - 'Sleep stage 3'
                    - 'Sleep stage 4'
                    - 'Sleep stage R'
                    - 'Sleep stage ?'
                    - 'Movement time'
                """
                for _ in range(int(dur) // 30):
                    labels.append(des)

            sample_length = SAMPLE_RATE * epoch_seconds

            for slice_index in range(min(X.shape[1] // sample_length, len(labels))):
                # ingore the no label epoch
                if labels[slice_index] not in [
                    "Sleep stage W",
                    "Sleep stage 1",
                    "Sleep stage 2",
                    "Sleep stage 3",
                    "Sleep stage 4",
                    "Sleep stage R",
                ]:
                    continue

                epoch_signal = X[
                    :, slice_index * sample_length : (slice_index + 1) * sample_length
                ]
                epoch_label = labels[slice_index][-1]  # "W", "1", "2", "3", "4", "R"
                samples.append(
                    {
                        "patient_id": pid,
                        "night": event.night,
                        "patient_age": event.age,
                        "patient_sex": event.sex,
                        "signal": epoch_signal,
                        "label": epoch_label,
                    }
                )

        return samples
