"""Sleep Staging Task for Cleveland Family Study (CFS) Dataset."""

from typing import Any, Dict, List, Optional

import mne
import numpy as np

from pyhealth.tasks import BaseTask

__all__ = ["SleepStagingCFS"]


class SleepStagingCFS(BaseTask):
    """Multi-class classification task for sleep staging on CFS dataset.

    This task predicts sleep stages (Awake, N1, N2, N3, REM) based on
    multichannel polysomnography signals (EEG, EOG, EMG, etc.).

    The task extracts 30-second epochs and assigns each epoch a sleep stage label
    based on AASM scoring rules as implemented in the CFS annotations.

    Attributes:
        task_name (str): The name of the task, set to "SleepStagingCFS".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format. Contains:
            - "signal": "tensor"
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "label": "multiclass" (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

    Notes:
        - Sleep stages are mapped as: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        - Each 30-second epoch is extracted as a single sample
        - Epochs with missing or invalid annotations are skipped
        - Signals are expected to be in EDF format with standard AASM channels

    Examples:
        >>> from pyhealth.datasets import CFSDataset
        >>> from pyhealth.tasks import SleepStagingCFS
        >>> # Load CFS PSG dataset
        >>> dataset = CFSDataset(root="/path/to/cfs")
        >>> # Create sleep staging task
        >>> task = SleepStagingCFS(chunk_duration=30.0, preload=True)
        >>> # Generate samples
        >>> samples = dataset.set_task(task)
        >>> print(f"Total samples: {len(samples)}")
        >>> print(samples[0].keys())
    """

    task_name: str = "SleepStagingCFS"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        chunk_duration: float = 30.0,
        preload: bool = True,
        selected_signals: Optional[List[str]] = None,
    ) -> None:
        """Initialize the SleepStagingCFS task.

        Args:
            chunk_duration (float): Duration of each signal chunk in seconds.
                Default is 30.0 seconds (standard AASM epoch duration).
            preload (bool): Whether to preload EDF data into memory.
                Default is True.
            selected_signals (Optional[List[str]]): List of signal names to extract.
                If None, automatically selects available channels from standard
                channels: EEG, EOG-L, EOG-R, EMG-Chin, ECG.
        """
        self.chunk_duration = chunk_duration
        self.preload = preload

        # Standard AASM signal channel names in different format variations
        self.standard_signals = {
            "eeg": [
                "EEG", "EEG 1", "EEG 2", "EEG Fpz-Cz", "EEG Pz-Oz",
                "EEG C3-A2", "EEG C4-A1"
            ],
            "eog_left": ["EOG-L", "EOG L", "EOG-Left", "EOG Left"],
            "eog_right": ["EOG-R", "EOG R", "EOG-Right", "EOG Right"],
            "emg": [
                "EMG", "EMG-Chin", "EMG Chin", "Chin EMG", "Submental EMG",
                "Chin", "Chin EMG"
            ],
            "ecg": ["ECG", "ECG 1", "ECG 2", "ECG-2"],
        }

        self.selected_signals = selected_signals
        super().__init__()

    def _get_signal_channels(self, raw_data: Any) -> List[str]:
        """Get available signal channels from EDF file.

        Args:
            raw_data: MNE RawEDF object.

        Returns:
            List of available signal channel names to use.
        """
        available_channels = raw_data.ch_names

        if self.selected_signals:
            # Use user-specified channels
            selected = [
                ch for ch in self.selected_signals
                if ch in available_channels
            ]
            if not selected:
                raise ValueError(
                    f"No selected signals found in EDF file. "
                    f"Available: {available_channels}"
                )
            return selected

        # Auto-select channels from standard sets
        selected = []
        for signal_type, variants in self.standard_signals.items():
            for variant in variants:
                if variant in available_channels:
                    selected.append(variant)
                    break  # Use first match for this type

        if not selected:
            raise ValueError(
                f"No standard sleep staging signals found in EDF file. "
                f"Available channels: {available_channels}"
            )

        return selected

    def _map_sleep_stage(self, stage_str: str) -> Optional[int]:
        """Map sleep stage annotation string to numeric label.

        Args:
            stage_str: Sleep stage label from annotation file.

        Returns:
            Numeric sleep stage (0=W, 1=N1, 2=N2, 3=N3, 4=R), or None if invalid.
        """
        # Normalize the stage string
        stage_str = stage_str.strip().upper()

        # Map common sleep stage notations
        stage_mapping = {
            "WAKE": 0, "W": 0, "0": 0,
            "SLEEP STAGE 1": 1, "N1": 1, "STAGE 1": 1, "1": 1,
            "SLEEP STAGE 2": 2, "N2": 2, "STAGE 2": 2, "2": 2,
            "SLEEP STAGE 3": 3, "N3": 3, "STAGE 3": 3, "3": 3,
            "SLEEP STAGE 4": 3, "STAGE 4": 3, "4": 3,  # AASM maps Stage 4 to N3
            "REM": 4, "R": 4, "SLEEP STAGE R": 4,
        }

        return stage_mapping.get(stage_str, None)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient for the sleep staging task on CFS.

        Sleep staging predicts sleep stages (Wake, N1, N2, N3, REM) based on
        polysomnography signals. Each 30-second epoch is extracted as a sample.

        Args:
            patient: A patient object containing CFS PSG data.

        Returns:
            samples: A list of samples, each sample is a dict with:
                - patient_id: CFS patient ID (nsrrid)
                - study_id: Study identifier (nsrrid-night)
                - patient_age: Age at time of study (from demographics)
                - patient_sex: Gender (from demographics)
                - signal: (n_channels, n_samples) signal array
                - label: Sleep stage (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

        Note:
            This task implements the multi-class classification for sleep stages
            according to AASM sleep staging criteria as annotated in the CFS dataset.

        Examples:
            >>> from pyhealth.datasets import CFSDataset
            >>> from pyhealth.tasks import SleepStagingCFS
            >>> dataset = CFSDataset(root="/srv/local/data/CFS/")
            >>> task = SleepStagingCFS()
            >>> dataset.set_task(task)
            >>> sample = dataset.samples[0]
            >>> print(sample.keys())
            dict_keys(['patient_id', 'study_id', 'patient_age', 'patient_sex',
                      'signal', 'label'])
            >>> print(sample['signal'].shape)  # (n_channels, 9000) for 30s @ 100Hz
        """
        pid = patient.patient_id
        events = patient.get_events()

        samples = []

        for event in events:
            try:
                # Read EDF file
                raw_data = mne.io.read_raw_edf(
                    event.signal_file,
                    infer_types=True,
                    preload=self.preload,
                    verbose="error",
                )
            except Exception as e:
                print(f"Error reading EDF file {event.signal_file}: {e}")
                continue

            # Get selected signal channels
            try:
                signal_channels = self._get_signal_channels(raw_data)
            except ValueError as e:
                print(f"Error selecting signal channels from {event.signal_file}: {e}")
                continue

            # Get only the selected channels
            raw_data.pick_channels(signal_channels)

            # Read annotations (prefer NSRR version, fall back to Profusion)
            annotation_file = None
            if hasattr(event, "label_file_nsrr") and event.label_file_nsrr:
                annotation_file = event.label_file_nsrr
            elif hasattr(event, "label_file") and event.label_file:
                annotation_file = event.label_file

            if not annotation_file:
                print(f"No annotation file found for {event.study_id}")
                continue

            try:
                annotations = mne.read_annotations(annotation_file)
                raw_data.set_annotations(annotations, emit_warning=False)
            except Exception as e:
                print(f"Error reading annotations from {annotation_file}: {e}")
                continue

            # Create events from annotations
            sfreq = raw_data.info["sfreq"]

            # Extract signal data
            signal_data = raw_data.get_data()  # (n_channels, n_samples)

            # Process annotations into 30-second epochs
            for annotation in annotations:
                # Parse annotation
                onset = annotation["onset"]  # Start time in seconds
                duration = annotation["duration"]  # Duration in seconds
                description = annotation["description"]  # Stage label

                # Map sleep stage
                stage = self._map_sleep_stage(description)
                if stage is None:
                    # Skip invalid/unknown stages
                    continue

                # Only use standard AASM 30-second epochs
                if abs(duration - self.chunk_duration) > 0.1:
                    # Allow small tolerance for rounding
                    continue

                # Extract epoch signal
                start_sample = int(onset * sfreq)
                end_sample = int((onset + self.chunk_duration) * sfreq)

                # Ensure we don't go past the end of the data
                if end_sample > signal_data.shape[1]:
                    continue

                epoch_signal = signal_data[:, start_sample:end_sample]

                # Get patient demographics if available
                age = getattr(event, "age", None)
                sex = getattr(event, "sex", None)

                # Create sample
                sample = {
                    "patient_id": pid,
                    "study_id": event.study_id if hasattr(event, "study_id") else pid,
                    "patient_age": age,
                    "patient_sex": sex,
                    "signal": epoch_signal,
                    "label": int(stage),
                }
                samples.append(sample)

        return samples


__all__ = ["SleepStagingCFS"]
