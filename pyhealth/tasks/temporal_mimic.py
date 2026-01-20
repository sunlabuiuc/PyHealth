from typing import Any, Dict, List

from .base_task import BaseTask

import os
import ast

class TemporalMIMICMultilabelClassification(BaseTask):
    """
    A PyHealth task class for multilabel classification in the Temporal MIMIC dataset.
    """

    task_name: str = "TemporalMIMICMultilabelClassification"
    input_schema: Dict[str, str] = {
        "radnotes": "text",
        "image_path": "image"
    }
    output_schema: Dict[str, str] = {
        "labels": "multilabel"
    }

    """
        Args:
            image_root (str): The root directory where your MIMIC-CXR-JPG dataset is stored.
    """
    def __init__(self, image_root: str = "", **kwargs):
        super().__init__(**kwargs)
        self.image_root = image_root

    def __call__(self, patient: Any) -> List[Dict]:
        """
        Generates multi-label classification samples for a single patient.

        Args:
            patient (Patient): A patient object containing one or more 'temporal_mimic' events (visits).

        Returns:
            List[Dict]: A list of dictionaries, one per visit image, each containing:
                - "radnotes": Full radiology report text.
                - "image_path": Path to the corresponding chest X-ray image.
                - "labels": A list of positive findings (strings from labels).
        """
        events = patient.get_events(event_type="temporal_mimic")
        samples = []

        for event in events:
            raw_folders = getattr(event, "img_folders", "[]")
            raw_filenames = getattr(event, "img_filenames", "[]")

            try:
                folders = ast.literal_eval(raw_folders)
                filenames = ast.literal_eval(raw_filenames)
            except Exception:
                continue

            for folder, filename in zip(folders, filenames):
                folder = str(folder).strip()
                filename = str(filename).strip()

                if not folder or not filename:
                    continue

                image_path = os.path.normpath(
                    os.path.join(self.image_root, folder, filename)
                )

                if not os.path.isfile(image_path):
                    continue

                from pyhealth.datasets.temporal_mimic import TemporalMIMICDataset
                labels = [
                    label for label in TemporalMIMICDataset.classes
                    if event[label] != None and float(event[label]) == 1.0
                ]

                samples.append({
                    "radnotes": getattr(event, "radnotes", ""),
                    "image_path": image_path,
                    "labels": labels,
                })

        return samples
