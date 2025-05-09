from typing import Any, Dict, List

from .base_task import BaseTask

class EEGIsSeizure(BaseTask):
    """Task for predicting seizures from EEG signals from the TUSZ corpus.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema for the task.
        output_schema (Dict[str, str]): The output schema for the task.
    """
    task_name: str = "EEGIsSeizure"
    input_schema: Dict[str, str] = {"signal": "signal"}
    output_schema: Dict[str, str] = {"seizure": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the seizure prediction task.

        Args:
            patient (Any): A Patient object containing patient data.

        Returns:
            List[Dict[str, Any]]: A list of samples, each sample is a dict with eeg signal 
            data and is seizure as keys.
        """
        events = patient.get_events(event_type="eeg_tusz")

        sample_size = 4

        samples = []

        for event in events:

            # Clears out some signials that are too short.
            if float(event.stop_time) < sample_size:
                continue

            start_time = event.start_time
            stop_time = event.stop_time
            edf_path = event.edf_file
            is_seizure = 1 if event.label == "seiz" else 0
            start_segment = int(float(start_time) / sample_size)
            stop_segment = int(float(stop_time) / sample_size)
            num_segments = int(
                (stop_segment - start_segment) / sample_size
            )

            if num_segments > 0:
                # Append the segments to the samples list
                for i in range(num_segments):
                    segment_idx = start_segment + i * sample_size
                    # Create a new sample for each segment
                    samples.append(
                        {
                            "signal": (segment_idx, edf_path),
                            "seizure": is_seizure,
                        }
                    )
            else:
                samples.append(
                        {
                            "signal": (0, edf_path),
                            "seizure": is_seizure,
                        }
                    )

        return samples