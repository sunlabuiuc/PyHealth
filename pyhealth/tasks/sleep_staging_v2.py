from typing import Any, ClassVar

import mne
import numpy as np
import torch

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
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}

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


class SleepStagingSequenceSleepEDF(BaseTask):
    """Sequence-to-sequence sleep staging on Sleep EDF, AASM stages.

    Works with :class:`~pyhealth.datasets.SleepEDFDataset`.

    Each sample is ``seq_len`` consecutive 30 s epochs of one EEG channel,
    labeled with the AASM stage of every epoch: W=0, N1=1, N2=2, N3=3, REM=4.
    The Rechtschaffen and Kales stages 3 and 4 are merged into N3. This is the
    input of sequence models such as DeepSleepNet and SalientSleepNet, and of
    SleepKD (Liang et al., IJCAI 2023), which use 20-epoch sequences of
    Fpz-Cz at 100 Hz.

    Epochs scored "Sleep stage ?" or "Movement time" are not used, and a
    window never spans one: windows are cut from runs of consecutive scored
    epochs, and the end of a run shorter than a window is dropped. Wake
    before the first and after the last sleep epoch is trimmed to
    ``wake_edge_minutes``, as in DeepSleepNet's preprocessing.

    ``signal`` is in microvolts, the unit DeepSleepNet was trained on (MNE
    reads volts). EEG is about 1e-5 V, so in volts the first convolution's
    output variance falls far below BatchNorm's ``eps``.

    Attributes:
        task_name (str): The name of the task, "SleepStagingSequence".
        input_schema (dict[str, str]): ``{"signal": "tensor"}``, a float array
            of shape ``(seq_len, 1, 30 * sfreq)``.
        output_schema (dict[str, Any]): ``{"label": "tensor"}`` (``torch.long``),
            the ``(seq_len,)`` stage of each epoch. The labels are a tensor rather than a
            ``timeseries_multiclass`` label because sequence models such as
            DeepSleepNet are bidirectional: each epoch's prediction uses the
            later epochs of the window too.

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> from pyhealth.tasks import SleepStagingSequenceSleepEDF
        >>> dataset = SleepEDFDataset(
        ...     root="/path/to/sleep-edf-database-expanded-1.0.0",
        ...     subset="cassette",
        ... )
        >>> samples = dataset.set_task(SleepStagingSequenceSleepEDF(seq_len=20))
    """

    task_name: str = "SleepStagingSequence"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": ("tensor", {"dtype": torch.long})
    }

    EPOCH_SECONDS: float = 30.0
    STAGES: ClassVar[dict[str, int]] = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }

    def __init__(
        self,
        seq_len: int = 20,
        stride: int | None = None,
        channel: str = "EEG Fpz-Cz",
        wake_edge_minutes: float | None = 30.0,
    ):
        """Initializes the SleepStagingSequenceSleepEDF task.

        Args:
            seq_len (int): Number of 30 s epochs in a sample. Default is 20.
            stride (int | None): Number of epochs between the starts of two
                windows. Default is ``seq_len`` (non-overlapping windows).
            channel (str): The EEG channel to use. Default is "EEG Fpz-Cz";
                the cassette subset also has "EEG Pz-Oz".
            wake_edge_minutes (float | None): Wake kept before the first and
                after the last sleep epoch of a night, in minutes. ``None``
                keeps the whole recording. Default is 30.
        """
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}.")
        stride = seq_len if stride is None else stride
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}.")
        self.seq_len = seq_len
        self.stride = stride
        self.channel = channel
        self.wake_edge_minutes = wake_edge_minutes
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Cuts each night of the patient into ``seq_len``-epoch samples.

        Args:
            patient: A patient object containing SleepEDF data.

        Returns:
            samples: a list of samples, each a dict with patient_id, night,
                start (the onset of the window in the recording, in
                seconds), signal (a ``numpy.ndarray`` of shape
                ``(seq_len, 1, 30 * sfreq)``) and label (a ``numpy.ndarray``
                of ``seq_len`` stage codes).
        """
        samples = []
        for event in patient.get_events():
            data = mne.io.read_raw_edf(
                event.signal_file,
                include=[self.channel],
                preload=True,
                verbose="error",
            )
            ann = mne.read_annotations(event.label_file)
            data.set_annotations(ann, emit_warning=False)
            if not set(ann.description) & set(self.STAGES):
                continue
            # Stages 3 and 4 share code 3, so the event codes are the AASM codes.
            events, _ = mne.events_from_annotations(
                data,
                event_id=self.STAGES,
                chunk_duration=self.EPOCH_SECONDS,
                verbose="error",
            )
            epoch_len = round(self.EPOCH_SECONDS * data.info["sfreq"])
            starts = events[:, 0] - data.first_samp
            labels = events[:, 2]
            # An epoch must lie fully inside the recording; the last
            # annotation often runs past its end.
            keep = starts + epoch_len <= data.n_times
            starts, labels = starts[keep], labels[keep]

            if self.wake_edge_minutes is not None:
                asleep = np.flatnonzero(labels != 0)
                if asleep.size == 0:
                    continue
                edge = round(self.wake_edge_minutes * 60 * data.info["sfreq"])
                lo = starts[asleep[0]] - edge
                hi = starts[asleep[-1]] + edge
                keep = (starts >= lo) & (starts <= hi)
                starts, labels = starts[keep], labels[keep]

            signal = data.get_data(picks=[self.channel])[0] * 1e6  # V -> uV
            # Runs of consecutive epochs: a gap means an unscored epoch.
            breaks = np.flatnonzero(np.diff(starts) != epoch_len) + 1
            for run in np.split(np.arange(len(starts)), breaks):
                for i in range(0, len(run) - self.seq_len + 1, self.stride):
                    idx = run[i : i + self.seq_len]
                    window = np.stack([signal[s : s + epoch_len] for s in starts[idx]])[
                        :, None, :
                    ].astype(np.float32)
                    samples.append(
                        {
                            "patient_id": patient.patient_id,
                            "night": event.night,
                            "start": float(starts[idx[0]] / data.info["sfreq"]),
                            "signal": window,
                            "label": labels[idx].astype(np.int64),
                        }
                    )
        return samples
