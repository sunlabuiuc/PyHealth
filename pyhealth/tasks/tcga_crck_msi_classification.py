"""Patient-level MSI classification task for TCGA-CRCk."""

from __future__ import annotations

from typing import Dict, List, Optional

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask


class TCGACRCkMSIClassification(BaseTask):
    """Creates one bag-of-tiles sample per patient for MSI classification.

    This task groups all tile events for a single patient into one bag and produces
    a single binary MSI classification sample. It also infers a patient-level
    train/test split from the underlying tile-level event metadata.
    """

    # change the task_name so cache is rebuilt
    task_name: str = "TCGACRCkMSIClassificationPatientLevel"

    input_schema: Dict[str, object] = {
        "tile_bag": (
            "time_image",
            {"image_size": 224, "mode": "RGB", "max_images": 1000},
        )
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, max_tiles: Optional[int] = 1000) -> None:
        """Initializes the TCGA-CRCk MSI classification task.

        Args:
            max_tiles: Maximum number of tile images to include in each patient bag.
                If `None`, all available tiles are used.
        """
        self.max_tiles = max_tiles
        processor_kwargs = {"image_size": 224, "mode": "RGB"}
        if max_tiles is not None:
            processor_kwargs["max_images"] = max_tiles
        self.input_schema = {"tile_bag": ("time_image", processor_kwargs)}

    @staticmethod
    def _normalize_split(value: object) -> str:
        """Normalizes split labels into a canonical string form.

        Args:
            value: Raw split value from event metadata.

        Returns:
            Normalized split name such as "train" or "test". Unrecognized values
            are returned as lowercase stripped strings.
        """
        text = str(value).strip().lower()
        if text in {"train", "training", "tr"}:
            return "train"
        if text in {"test", "testing", "te"}:
            return "test"
        return text

    def __call__(self, patient: Patient) -> List[Dict]:
        """Builds one patient-level MSI classification sample.

        The method collects all TCGA-CRCk events for a patient, verifies that the
        labels are consistent across tiles, infers a patient-level data split, and
        returns a single bag-of-tiles sample.

        Args:
            patient: Patient object containing TCGA-CRCk tile events.

        Returns:
            A list containing one sample dictionary for the patient, or an empty
            list if the patient has no matching events.

        Raises:
            ValueError: If tile labels for the patient are inconsistent.
            ValueError: If a valid train/test split cannot be inferred.
        """
        events: List[Event] = patient.get_events(event_type="tcga_crck")
        if not events:
            return []

        sorted_events = sorted(
            events,
            key=lambda e: (str(e["slide_id"]), int(e["tile_index"])),
        )

        labels = {int(event["label"]) for event in sorted_events}
        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent labels for patient {patient.patient_id}: {sorted(labels)}"
            )
        label = next(iter(labels))

        splits = {self._normalize_split(event["data_split"]) for event in sorted_events}
        if "test" in splits:
            data_split = "test"
        elif "train" in splits:
            data_split = "train"
        else:
            raise ValueError(
                f"Could not infer split for patient {patient.patient_id}: {sorted(splits)}"
            )

        tile_paths = [str(event["tile_path"]) for event in sorted_events]
        tile_times = [float(i) for i in range(len(sorted_events))]

        return [
            {
                "patient_id": str(patient.patient_id),
                "visit_id": str(patient.patient_id),
                "tile_bag": (tile_paths, tile_times),
                "label": label,
                "data_split": data_split,
            }
        ]