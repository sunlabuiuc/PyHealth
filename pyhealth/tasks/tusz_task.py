# import logging
from typing import Any, Dict, List
from pyhealth.tasks import BaseTask
from pyedflib import highlevel
from .tusz_utility_class import TUSZHelper, TUSZSignalHeader, LOG_INFO, LOG_WARN

class TUSZTask(BaseTask):
    """Multi-class classification task for seizure detection on TUSZ.

    For each EDF recording, this task:
      1) reads the EDF
      2) applies bandpass, notch, and resamples
      3) loads paired event annotations from the .rec file
      4) constructs 5-second windows per event
      5) returns one sample per event

    Each returned sample contains:
      - "signal": torch.FloatTensor, shape (16, n_samples)
      - "label": int (0 = non-seizure, 1 = seizure)
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
        sample_rate            = 200,
        feature_sample_rate    = 50,
        label_type             = 'csv',
        eeg_type               = 'bipolar', # bipolar, uni_bipolar
        min_binary_slicelength = 30,
        min_binary_edge_seiz   = 3,
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
        """Generates samples for one patient."""
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []

        events = patient.get_events()
        for event in events:

            # 0. read edf file
            file = event.signal_file
            signals, signal_headers, _ = highlevel.read_edf(file)
            file_name = ".".join(file.split(".")[:-1])
            data_file_name = file_name.split("/")[-1]
            signal_headers = TUSZSignalHeader(signal_headers)

            # 1. skip certain conditions
            self.helper.log(LOG_INFO, data_file_name, "checking skip conditions...")
            if self.helper.skip_file(file_name, signal_headers):
                self.helper.log(LOG_WARN, data_file_name, f"** skipping {file_name} ***")
                continue

            # 2. get labels and patient status
            self.helper.log(LOG_INFO, data_file_name, "processing labels...")
            y_sampled = self.helper.process_label(file_name)
            self.helper.log(LOG_INFO, data_file_name, "checking patient status...")
            is_seiz_patient = self.helper.is_seizure_patient(file)

            # 3. resample signals
            self.helper.log(LOG_INFO, data_file_name, "resampling...")
            signal_final_list_raw = self.helper.resample(data_file_name, signals, signal_headers)
            if not signal_final_list_raw:
                self.helper.log(LOG_WARN, data_file_name, f"** skipping {file_name} ***")
                continue

            # 4. transform labels with resampled signals
            self.helper.log(LOG_INFO, data_file_name, "transforming labels...")
            y_sampled = self.helper.transform_labels_with_resampled_signals(
                signal_final_list_raw, y_sampled
            )

            # 5. segment signals
            self.helper.log(LOG_INFO, data_file_name, "segmenting signals...")
            sliced_raws, sliced_labels, label_names = self.helper.segment_signals(
                y_sampled, signal_final_list_raw, is_seiz_patient
            )
            if not sliced_raws:
                continue

            # 6. convert labels to binary targets and bytes
            byte_labels, label_bitgt_1, label_bitgt_2 = self.helper.convert_labels(sliced_labels)

            # 7. create bipolar signals and construct sample
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
