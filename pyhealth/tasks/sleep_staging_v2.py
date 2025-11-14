from typing import Any, Dict

import mne

from pyhealth.tasks import BaseTask


class SleepStagingSleepEDF(BaseTask):
    """Multi-class classification task for sleep staging on Sleep EDF dataset.

    This task predicts sleep stages (Awake, REM, N1, N2, N3, N4) based on
    multichannel EEG signals. The task is defined as a multi-class classification.

    Attributes:
        task_name (str): The name of the task, set to "SleepStaging".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains:
            - "signal": "tensor"
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "label": "multiclass"
    """

    task_name: str = "SleepStaging"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(self, chunk_duration: float = 30.0):
        """Initializes the SleepStagingSleepEDF task.

        Args:
            chunk_duration (float): Duration of each EEG signal chunk in seconds.
                Default is 30.0 seconds.
        """
        self.chunk_duration = chunk_duration
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Processes a single patient for the sleep staging task on Sleep EDF.

        Sleep staging aims at predicting the sleep stages (Awake, REM, N1, N2, N3, N4) based on
        the multichannel EEG signals. The task is defined as a multi-class classification.

        Args:
            patient: A patient object containing SleepEDF data.

        Returns:
            samples: a list of samples, each sample is a dict with patient_id, night,
                patient_age, patient_sex, signal, and label.

        Note that we define the task as a multi-class classification task.

        Examples:
            >>> from pyhealth.datasets import SleepEDFDataset
            >>> sleepedf = SleepEDFDataset(
            ...         root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
            ...     )
            >>>
            >>> sleepstage_ds = sleepedf.set_task()
            >>> sleepstage_ds.samples[0]
            {
                'patient_id': 'SC4001',
                'night': 0,
                'patient_age': 58,
                'patient_sex': 'M',
                'signal': array(...),
                'label': 0
            }
        """

        pid = patient.patient_id
        events = patient.get_events()

        samples = []
        for event in events:
            data = mne.io.read_raw_edf(
                event.signal_file,
                stim_channel="Event marker",
                infer_types=True,
                preload=True,
                verbose="error",
            )
            ann = mne.read_annotations(event.label_file)
            data.set_annotations(ann, emit_warning=False)
            event_id = {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 4,
                "Sleep stage R": 5,
            }

            ann_events, _ = mne.events_from_annotations(
                data, event_id=event_id, chunk_duration=self.chunk_duration
            )

            epochs_train = mne.Epochs(
                data,
                ann_events,
                event_id,
                tmin=0.0,
                tmax=self.chunk_duration - 1.0 / data.info["sfreq"],
                baseline=None,
                preload=True,
            )

            signals = epochs_train.get_data()
            labels = epochs_train.events[:, 2]

            for epoch in range(labels.shape[0]):
                signal = signals[epoch, ...]
                label = labels[epoch, ...]

                samples.append(
                    {
                        "patient_id": pid,
                        "night": event.night,
                        "patient_age": event.age,
                        "patient_sex": event.sex,
                        "signal": signal,
                        "label": int(label),
                    }
                )

        return samples
