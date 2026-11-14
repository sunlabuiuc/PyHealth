from typing import Any, Dict

import mne

from pyhealth.tasks import BaseTask


class SleepStagingSleepEDF(BaseTask):
    """Multi-class classification task for sleep staging on Sleep EDF dataset.

    Works with :class:`~pyhealth.datasets.SleepEDFDataset`.

    This task predicts the Rechtschaffen and Kales stage of each
    ``chunk_duration`` epoch: W=0, 1=1, 2=2, 3=3, 4=4, R=5. Epochs scored
    "Sleep stage ?" or "Movement time" are not used. A night that lacks some
    stages (many Sleep-EDF nights have no stage 4) keeps the same label codes.

    ``signal`` holds every channel of the PSG file, as read by MNE: for the
    cassette subset, the two EEG channels (Fpz-Cz, Pz-Oz), horizontal EOG,
    oro-nasal airflow, submental EMG, rectal temperature and the event marker,
    all at 100 Hz, in volts. Select channels downstream if you need EEG only.
    The full recording is used, so long wake periods before and after sleep
    are kept (about 75% of the epochs are W for SC4001).

    Attributes:
        task_name (str): The name of the task, set to "SleepStaging".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains:
            - "signal": "tensor"
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "label": "multiclass"

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> from pyhealth.tasks.sleep_staging_v2 import SleepStagingSleepEDF
        >>> dataset = SleepEDFDataset(
        ...     root="/path/to/sleep-edf-database-expanded-1.0.0",
        ...     subset="cassette",
        ... )
        >>> samples = dataset.set_task(SleepStagingSleepEDF(chunk_duration=30.0))
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

        Each night (event) of the patient is cut into ``chunk_duration``
        epochs labeled with the scored stage; see the class docstring for the
        label codes and the channels in ``signal``.

        Args:
            patient: A patient object containing SleepEDF data.

        Returns:
            samples: a list of samples, each sample is a dict with patient_id,
                night, patient_age, patient_sex, signal (a ``numpy.ndarray`` of
                shape ``(n_channels, chunk_duration * sfreq)``), and label.

        Note that we define the task as a multi-class classification task.

        Examples:
            >>> from pyhealth.datasets import SleepEDFDataset
            >>> sleepedf = SleepEDFDataset(
            ...     root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0",
            ...     )
            >>>
            >>> sleepstage_ds = sleepedf.set_task()
            >>> sleepstage_ds.samples[0]
            {
                'patient_id': '0',
                'night': 1,
                'patient_age': 33,
                'patient_sex': 1,
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

            if not set(ann.description) & set(event_id):
                # No scored sleep stage in this night (only "Sleep stage ?" or
                # "Movement time"): there is nothing to epoch, and
                # events_from_annotations would raise.
                continue
            ann_events, _ = mne.events_from_annotations(
                data, event_id=event_id, chunk_duration=self.chunk_duration
            )

            # A night does not always contain every stage: many Sleep-EDF
            # hypnograms have no stage 4 (or no stage 3). mne.Epochs raises by
            # default when an event_id key has no event, so ignore missing
            # keys. The label codes stay fixed because event_id is unchanged.
            epochs_train = mne.Epochs(
                data,
                ann_events,
                event_id,
                tmin=0.0,
                tmax=self.chunk_duration - 1.0 / data.info["sfreq"],
                baseline=None,
                preload=True,
                on_missing="ignore",
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
