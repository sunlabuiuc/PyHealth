"""Sleep Staging Task for Cleveland Family Study (CFS) Dataset.

Dataset link:
    Dataset must be requested from the NSRR at https://sleepdata.org/datasets/cfs

Dataset paper: (please cite if you use this dataset)
    Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D,
    Redline S. The National Sleep Research Resource: towards a sleep data commons.
    J Am Med Inform Assoc. 2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064.
    PMID: 29860441; PMCID: PMC6188513.

    Redline S, Tishler PV, Tosteson TD, Williamson J, Kump K, Browner I, Ferrette V,
    Krejci P. The familial aggregation of obstructive sleep apnea. Am J Respir Crit 
    Care Med. 1995 Mar;151(3 Pt 1):682-7.
    doi: 10.1164/ajrccm/151.3_Pt_1.682. PMID: 7881656.

Please include the following text in the Acknowledgements:
    The Cleveland Family Study (CFS) was supported by grants from the National Institutes
    of Health (HL46380, M01 RR00080-39, T32-HL07567, RO1-46380).
    The National Sleep ResearchResource was supported by the National Heart, Lung, and
    Blood Institute (R24 HL114473, 75N92019R002).

Wav2Sleep paper link:
    https://doi.org/10.48550/arXiv.2411.04644

Wav2Sleep paper citation:
    Carter, J. F.; and Tarassenko, L. 2024. wav2sleep: A unified multi-modal approach
    to sleep stage classification from physiological signals. arXiv preprint arXiv:2411.04644.

Authors:
    Austin Jarrett (ajj7@illinois.edu)
    Justin Cheok (jcheok2@illinois.edu)
    Jimmy Scray (escray2@illinois.edu)
"""
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import mne
import numpy as np
from scipy import signal as scipy_signal

from pyhealth.tasks import BaseTask

__all__ = ["SleepStagingCFS"]


class SleepStagingCFS(BaseTask):
    """Multi-class classification task for sleep staging on CFS dataset.

    This task predicts sleep stages (Awake, N1, N2, N3, REM) based on
    multichannel polysomnography signals (EEG, EOG, EMG, etc.).

    The task extracts 30-second epochs and assigns each epoch a sleep stage label
    based on AASM scoring rules as implemented in the CFS annotations.

    Outputs samples in wav2sleep format with individual modality channels
    (eeg, eog_left, eog_right, emg, ecg) and optionally ppg.
    Missing PPG signals are zero-padded.

    Attributes:
        task_name (str): The name of the task, set to "SleepStagingCFS".
        input_schema (Dict[str, str]): The input schema specifying the required
            input format (used internally for validation).
        output_schema (Dict[str, str]): The output schema specifying the output
            format. Contains:
            - "label": "multiclass" (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
            - Multi-modal signal keys: "eeg", "eog_left", "eog_right", "emg", "ecg"
            - Optional: "ppg" (if include_ppg=True)

    Notes:
        - Sleep stages are mapped as: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        - Each 30-second epoch is extracted as a single sample
        - Epochs with missing or invalid annotations are skipped
        - Signals are expected to be in EDF format with standard AASM channels
        - Each modality is resampled to target length for consistency

    Examples:
        >>> from pyhealth.datasets import CFSDataset
        >>> from pyhealth.tasks import SleepStagingCFS
        >>> # Load CFS PSG dataset and generate samples
        >>> dataset = CFSDataset(root="/path/to/cfs")
        >>> task = SleepStagingCFS(chunk_duration=30.0, preload=False)
        >>> # Samples are in wav2sleep format (ready for multi-modal training)
        >>> samples = dataset.set_task(task)
        >>> print(f"Total samples: {len(samples)}")
        >>> print(samples[0].keys())
        # Output: dict_keys(['patient_id', 'study_id', 'patient_age', 'patient_sex',
        #                     'eeg', 'eog_left', 'eog_right', 'emg', 'ecg', 'label'])
    """

    task_name: str = "SleepStagingCFS"
    # Schema for wav2sleep format with multi-modal channels (updated dynamically in __init__)
    input_schema: Dict[str, str] = {
        "eeg": "tensor",
        "eog_left": "tensor",
        "eog_right": "tensor",
        "emg": "tensor",
        "ecg": "tensor",
    }
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        chunk_duration: float = 30.0,
        preload: bool = False,
        selected_signals: Optional[List[str]] = None,
        eeg_samples: int = 256,
        eog_samples: int = 256,
        emg_samples: int = 128,
        ecg_samples: int = 256,
        include_ppg: bool = False,
        ppg_samples: int = 256,
    ) -> None:
        """Initialize the SleepStagingCFS task.

        Args:
            chunk_duration (float): Duration of each signal chunk in seconds.
                Default is 30.0 seconds (standard AASM epoch duration).
            preload (bool): Whether to preload EDF data into memory.
                Default is False (recommended for large datasets).
            selected_signals (Optional[List[str]]): List of signal names to extract.
                If None, automatically selects available channels from standard
                channels: EEG, EOG-L, EOG-R, EMG-Chin, ECG, and optionally PPG.
            eeg_samples (int): Target sample count for EEG after resampling.
                Default is 256 (~8 Hz for 30s epoch).
            eog_samples (int): Target sample count for EOG after resampling.
                Default is 256 (~8 Hz for 30s epoch).
            emg_samples (int): Target sample count for EMG after resampling.
                Default is 128 (~4 Hz for 30s epoch).
            ecg_samples (int): Target sample count for ECG after resampling.
                Default is 256 (~8 Hz for 30s epoch).
            include_ppg (bool): Whether to extract PPG (plethysmograph) signal.
                Note: PPG is available in only ~43% of CFS files. Missing PPG signals
                are zero-padded. Default is False.
            ppg_samples (int): Target sample count for PPG after resampling.
                Default is 256 (~8 Hz for 30s epoch).
        """
        self.chunk_duration = chunk_duration
        self.preload = preload
        self.include_ppg = include_ppg
        
        # Target sample counts for each modality (after resampling)
        self.eeg_samples = eeg_samples
        self.eog_samples = eog_samples
        self.emg_samples = emg_samples
        self.ecg_samples = ecg_samples
        self.ppg_samples = ppg_samples

        # Standard AASM signal channel names in different format variations
        self.standard_signals = {
            "eeg": [
                "EEG", "EEG 1", "EEG 2", "EEG Fpz-Cz", "EEG Pz-Oz",
                "EEG C3-A2", "EEG C4-A1", "C3", "C4", "Cz"
            ],
            "eog_left": [
                "EOG-L", "EOG L", "EOG-Left", "EOG Left",
                "LOC", "E1", "LEOG", "Left Eye"
            ],
            "eog_right": [
                "EOG-R", "EOG R", "EOG-Right", "EOG Right",
                "ROC", "E2", "REOG", "Right Eye"
            ],
            "emg": [
                "EMG", "EMG-Chin", "EMG Chin", "Chin EMG", "Submental EMG",
                "Chin", "Chin EMG", "EMG1", "EMG2", "EMG3"
            ],
            "ecg": [
                "ECG", "ECG 1", "ECG 2", "ECG-2", "ECG1", "ECG2", "EKG"
            ],
            "ppg": [
                "PlethWV", "Pleth", "PPG", "Plethysmograph"
            ],
        }
        
        # Update input schema to include PPG if enabled
        if self.include_ppg:
            self.input_schema = {
                "eeg": "tensor",
                "eog_left": "tensor",
                "eog_right": "tensor",
                "emg": "tensor",
                "ecg": "tensor",
                "ppg": "tensor",
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

    def _parse_nsrr_xml_annotations(self, xml_file: str) -> List[Dict[str, Any]]:
        """Parse NSRR XML annotation file to extract sleep stage annotations.
        
        Args:
            xml_file: Path to the NSRR XML annotation file.
            
        Returns:
            List of annotation dicts with keys: 'onset', 'duration', 'description'
        """
        annotations = []
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find all ScoredEvent elements
            for event in root.findall('.//ScoredEvent'):
                # Get EventConcept which contains the label
                concept_elem = event.find('EventConcept')
                if concept_elem is None or not concept_elem.text:
                    continue
                    
                # EventConcept often has format "Label|Code", split to get label
                concept = concept_elem.text.strip()
                # Handle both "Wake|0" and just "Wake" format
                label = concept.split('|')[0].strip()
                
                # Get Start time (onset in seconds)
                start_elem = event.find('Start')
                if start_elem is None or not start_elem.text:
                    continue
                    
                try:
                    onset = float(start_elem.text)
                except (ValueError, TypeError):
                    continue
                
                # Get Duration
                duration_elem = event.find('Duration')
                if duration_elem is None or not duration_elem.text:
                    continue
                    
                try:
                    duration = float(duration_elem.text)
                except (ValueError, TypeError):
                    continue
                
                # Only include sleep stage annotations (ignoring artifacts, etc.)
                # Sleep stages typically include: Wake, N1, N2, N3, REM, etc.
                sleep_stage_keywords = [
                    'wake', 'n1', 'n2', 'n3', 'rem', 'sleep stage',
                    'stage 1', 'stage 2', 'stage 3', 'stage 4'
                ]
                
                if any(keyword in label.lower() for keyword in sleep_stage_keywords):
                    annotations.append({
                        'onset': onset,
                        'duration': duration,
                        'description': label
                    })
        except Exception as e:
            print(f"Error parsing XML file {xml_file}: {e}")
            
        return annotations

    def _read_annotations_from_xml(self, xml_file: str) -> mne.Annotations:
        """Read NSRR XML annotations and convert to MNE Annotations object.
        
        Args:
            xml_file: Path to the NSRR XML annotation file.
            
        Returns:
            MNE Annotations object or None if parsing fails.
        """
        try:
            events = self._parse_nsrr_xml_annotations(xml_file)
            
            if not events:
                return None
                
            # Convert to MNE Annotations format
            onsets = [e['onset'] for e in events]
            durations = [e['duration'] for e in events]
            descriptions = [e['description'] for e in events]
            
            return mne.Annotations(onsets, durations, descriptions)
        except Exception as e:
            print(f"Error creating MNE annotations from {xml_file}: {e}")
            return None

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
        # Includes CFS format ("Stage 1 sleep"), AASM format ("N1"), and others
        stage_mapping = {
            "WAKE": 0, "W": 0, "0": 0,
            # AASM N1 (Light Sleep, Stage 1)
            "N1": 1, "STAGE 1": 1, "SLEEP STAGE 1": 1, "STAGE 1 SLEEP": 1, "1": 1,
            # AASM N2 (Light Sleep, Stage 2)
            "N2": 2, "STAGE 2": 2, "SLEEP STAGE 2": 2, "STAGE 2 SLEEP": 2, "2": 2,
            # AASM N3 (Deep Sleep, Stage 3)
            "N3": 3, "STAGE 3": 3, "SLEEP STAGE 3": 3, "STAGE 3 SLEEP": 3, "3": 3,
            # Stage 4 maps to N3 per AASM
            "SLEEP STAGE 4": 3, "STAGE 4": 3, "STAGE 4 SLEEP": 3, "4": 3,
            # REM Sleep
            "REM": 4, "REM SLEEP": 4, "SLEEP STAGE REM": 4, "R": 4, "SLEEP STAGE R": 4,
        }

        return stage_mapping.get(stage_str, None)

    def _resample_signal(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Resample a 1D signal to a target length using scipy.

        Args:
            signal: Input signal shape (n_samples,).
            target_length: Target length for resampled signal.

        Returns:
            Resampled signal of shape (target_length,) as float32.
        """
        if signal.shape[0] == target_length:
            return signal.astype(np.float32)

        # Use scipy's resample for high-quality resampling
        resampled = scipy_signal.resample(signal, target_length)
        return resampled.astype(np.float32)

    def _extract_and_resample_modalities(
        self, signal_data: np.ndarray, channel_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract individual modalities from multi-channel signal and resample.

        Extracts EEG, EOG-L, EOG-R, EMG, and ECG channels and resamples each
        to the target length for that modality.

        Args:
            signal_data: Multi-channel signal array shape (n_channels, n_samples).
            channel_names: List of channel names corresponding to rows in signal_data.

        Returns:
            Dict with keys 'eeg', 'eog_left', 'eog_right', 'emg', 'ecg'
            containing resampled signals as float32 numpy arrays.
        """
        modalities = {}

        # Track which modality type each channel belongs to
        channel_to_type = {}
        for i, ch_name in enumerate(channel_names):
            for modality_type, variants in self.standard_signals.items():
                if any(variant in ch_name for variant in variants):
                    channel_to_type[i] = modality_type
                    break

        # Extract and resample each modality
        modality_targets = {
            "eeg": self.eeg_samples,
            "eog_left": self.eog_samples,
            "eog_right": self.eog_samples,
            "emg": self.emg_samples,
            "ecg": self.ecg_samples,
        }
        if self.include_ppg:
            modality_targets["ppg"] = self.ppg_samples

        for modality_type, target_length in modality_targets.items():
            # Find channels for this modality
            modality_channels = [
                i for i, m_type in channel_to_type.items()
                if m_type == modality_type
            ]

            if not modality_channels:
                # If modality not found, create zeros of target length
                modalities[modality_type] = np.zeros(target_length, dtype=np.float32)
            else:
                # Average multiple channels of same modality, or use single channel
                if len(modality_channels) > 1:
                    avg_signal = np.mean(
                        signal_data[modality_channels, :], axis=0
                    )
                else:
                    avg_signal = signal_data[modality_channels[0], :]

                # Resample to target length
                modalities[modality_type] = self._resample_signal(
                    avg_signal, target_length
                )

        return modalities

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
                - eeg: EEG signal resampled to target length (eeg_samples,)
                - eog_left: Left EOG signal resampled to target length (eog_samples,)
                - eog_right: Right EOG signal resampled to target length (eog_samples,)
                - emg: EMG signal resampled to target length (emg_samples,)
                - ecg: ECG signal resampled to target length (ecg_samples,)
                - ppg: PPG signal resampled to target length (ppg_samples,)
                  [only if include_ppg=True; zero-padded if missing in source data]
                - label: Sleep stage (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

        Note:
            This task implements the multi-class classification for sleep stages
            according to AASM sleep staging criteria as annotated in the CFS dataset.
            Outputs are in wav2sleep format with individual modality channels.

        Examples:
            >>> from pyhealth.datasets import CFSDataset
            >>> from pyhealth.tasks import SleepStagingCFS
            >>> dataset = CFSDataset(root="/path/to/cfs")
            >>> task = SleepStagingCFS()
            >>> dataset.set_task(task)
            >>> sample = dataset.samples[0]
            >>> print(sample.keys())
            dict_keys(['patient_id', 'study_id', 'patient_age', 'patient_sex',
                      'eeg', 'eog_left', 'eog_right', 'emg', 'ecg', 'label'])
            >>> print(sample['eeg'].shape)  # (256,) – resampled EEG
        """
        pid = patient.patient_id
        
        # Get demographics data (should be one demographics event per patient)
        demographics_events = patient.get_events(event_type="demographics")
        patient_age = None
        patient_sex = None
        
        if demographics_events:
            demo_event = demographics_events[0]
            patient_age = getattr(demo_event, "age", None)
            # In the CFS dataset, 'male' is the gender field (1=male, 0=female)
            patient_sex = getattr(demo_event, "male", None)
        
        # Only get polysomnography events (not demographics)
        events = patient.get_events(event_type="polysomnography")

        samples = []

        for event in events:
            # Verify signal_file exists
            if not hasattr(event, "signal_file") or not event.signal_file:
                print(f"Event has no signal_file attribute, skipping")
                continue

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
                # Try to read NSRR XML annotations
                if annotation_file.endswith('.xml'):
                    annotations = self._read_annotations_from_xml(annotation_file)
                    if annotations is None:
                        print(f"Failed to parse XML annotations from {annotation_file}")
                        continue
                else:
                    # Try MNE's native read_annotations
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

                # Extract and resample individual modalities
                modalities = self._extract_and_resample_modalities(
                    epoch_signal, raw_data.ch_names
                )

                # Create sample in wav2sleep format
                sample = {
                    "patient_id": pid,
                    "study_id": event.study_id if hasattr(event, "study_id") else pid,
                    "patient_age": patient_age,
                    "patient_sex": patient_sex,
                    "eeg": modalities["eeg"],
                    "eog_left": modalities["eog_left"],
                    "eog_right": modalities["eog_right"],
                    "emg": modalities["emg"],
                    "ecg": modalities["ecg"],
                    "label": int(stage),
                }
                
                # Include PPG if enabled (zero-padded if missing)
                if self.include_ppg:
                    sample["ppg"] = modalities["ppg"]
                
                samples.append(sample)

        return samples


__all__ = ["SleepStagingCFS"]
