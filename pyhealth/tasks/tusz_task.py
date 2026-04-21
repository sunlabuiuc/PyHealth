"""
PyHealth task for extracting features with STFT and Frequency Bands using the Temple University Hospital (TUH) EEG Seizure Corpus (TUSZ) dataset V2.0.5.

Dataset link:
    https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml

Dataset paper:
    Vinit Shah, Eva von Weltin, Silvia Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” arXiv preprint arXiv:1801.08085, 2018. Available: https://arxiv.org/abs/1801.08085

Dataset paper link:
    https://arxiv.org/abs/1801.08085

Author:
    Fernando Kenji Sakabe (fks@illinois.edu), 
    Jesica Hirsch (jesicah2@illinois.edu), 
    Jung-Jung Hsieh (jhsieh8@illinois.edu)
"""
from typing import Any, Dict, List
from pyhealth.tasks import BaseTask
from pyedflib import highlevel
from .tusz_utility_class import TUSZHelper, TUSZSignalHeader, LOG_INFO, LOG_WARN

class TUSZTask(BaseTask):
    """Multi-class/Binary Classification for seizure detection on TUSZ.

    For each EDF recording, this task:
      1) reads the EDF
      2) skips certain conditions
      3) gets labels
      4) gets patient status
      5) resamples signals
      6) transforms labels with resampled signals
      7) segments signals
      8) converts labels to binary targets and bytes
      9) creates bipolar signals

    Each returned sample contains:
      - "signal": torch.FloatTensor, shape (16, n_samples)
      - "label": int (0 = non-seizure, others = seizure)
            full classification
      - "label_bitgt_1": int (0 = non-seizure, 1 = seizure)
            binary classification
      - "label_bitgt_2": int (0 = non-seizure, 1 = seizure)
            another kind of multi-class classfication
      - "label_name": str (0_* = non-seizure, others = seizure)

    Examples:
        >>> from pyhealth.datasets import TUSZDataset
        >>> from pyhealth.tasks import TUSZTask
        >>> dataset = TUSZDataset(root = 'tuh_eeg_v2.0.5', subset = 'train')
        >>> task = TUSZTask(
        ...     sample_rate = SAMPLE_RATE,
        ...     feature_sample_rate = FEATURE_SAMPLE_RATE,
        ... )
        >>> samples = dataset.set_task(task)
        >>> sample = samples[0]
        >>> print(sample['signal'].shape)
    """

    task_name: str = "tusz_task"
    input_schema: Dict[str, str] = { "signal": "tensor" }
    output_schema: Dict[str, str] = {
        "label": "tensor",
        "label_bitgt_1": "tensor",
        "label_bitgt_2": "tensor",
        "label_name": "text",
    }

    def __init__(
        self,
        sample_rate: int            = 200,
        feature_sample_rate: int    = 50,
        label_type: str             = 'csv',
        eeg_type: str               = 'bipolar', # bipolar, uni_bipolar
        min_binary_slicelength: int = 30,
        min_binary_edge_seiz: int   = 3,
    ) -> None:
        super().__init__()
        self.helper = TUSZHelper(
            sample_rate            = sample_rate,
            feature_sample_rate    = feature_sample_rate,
            label_type             = label_type,
            eeg_type               = eeg_type,
            min_binary_slicelength = min_binary_slicelength,
            min_binary_edge_seiz   = min_binary_edge_seiz,
        )

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Preprocesses one patient with nine steps.
        
        Details of the implementations are in TUSZHelper.
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []

        events = patient.get_events()
        for event in events:

            # 1. read edf file
            file = event.signal_file
            signals, signal_headers, _ = highlevel.read_edf(file)
            file_name = ".".join(file.split(".")[:-1])
            data_file_name = file_name.split("/")[-1]
            signal_headers = TUSZSignalHeader(signal_headers)

            # 2. skip certain conditions
            self.helper.log(LOG_INFO, data_file_name, "checking skip conditions...")
            if self.helper.skip_file(file_name, signal_headers):
                self.helper.log(LOG_WARN, data_file_name, f"** skipping {file_name} ***")
                continue

            # 3. get labels
            self.helper.log(LOG_INFO, data_file_name, "processing labels...")
            y_sampled = self.helper.process_label(file_name)

            # 4. get patient status
            self.helper.log(LOG_INFO, data_file_name, "checking patient status...")
            is_seiz_patient = self.helper.is_seizure_patient(file)

            # 5. resample signals
            self.helper.log(LOG_INFO, data_file_name, "resampling...")
            signal_final_list_raw = self.helper.resample(data_file_name, signals, signal_headers)
            if not signal_final_list_raw:
                self.helper.log(LOG_WARN, data_file_name, f"** skipping {file_name} ***")
                continue

            # 6. transform labels with resampled signals
            self.helper.log(LOG_INFO, data_file_name, "transforming labels...")
            y_sampled = self.helper.transform_labels_with_resampled_signals(
                signal_final_list_raw, y_sampled
            )

            # 7. segment signals
            self.helper.log(LOG_INFO, data_file_name, "segmenting signals...")
            sliced_raws, sliced_labels, label_names = self.helper.segment_signals(
                y_sampled, signal_final_list_raw, is_seiz_patient
            )
            if not sliced_raws:
                continue

            # 8. convert labels to binary targets and bytes
            byte_labels, label_bitgt_1, label_bitgt_2 = self.helper.convert_labels(sliced_labels)

            # 9. create bipolar signals and construct sample
            for (
                raw, label, label_bitgt_1, label_bitgt_2, label_name
            ) in zip(
                sliced_raws, byte_labels, label_bitgt_1, label_bitgt_2, label_names
            ):
                sample = {
                    "patient_id": pid,
                    "signal": self.helper.create_bipolar_signals(raw),
                    "label": label,
                    "label_bitgt_1": label_bitgt_1,
                    "label_bitgt_2": label_bitgt_2,
                    "label_name": label_name
                }

                samples.append(sample)

            self.helper.log(LOG_INFO, data_file_name, "** completed successfullly **")

        return samples
