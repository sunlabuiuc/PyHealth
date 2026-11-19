"""Skin-lesion diagnosis task for PAD-UFES-20."""

from __future__ import annotations

from typing import Any

from .base_task import BaseTask

PAD6_CLASSES: tuple[str, ...] = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
"""The six official PAD-UFES-20 diagnoses, in label-index order."""

CLIMB5_CLASSES: tuple[str, ...] = ("AKIEC", "BCC", "MEL", "NV", "OTHER")
"""The five classes CLIMB groups PAD-UFES-20 into, in label-index order."""

CLIMB5_MAP: dict[str, str] = {
    "ACK": "AKIEC",
    "SCC": "AKIEC",
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NV",
    "SEK": "OTHER",
}
"""CLIMB grouping: ACK and SCC become AKIEC, NEV becomes NV, SEK becomes OTHER."""

_SCHEMES: dict[str, tuple[str, ...]] = {
    "pad6": PAD6_CLASSES,
    "climb5": CLIMB5_CLASSES,
}


class PADUFES20Classification(BaseTask):
    """Multiclass skin-lesion diagnosis from PAD-UFES-20 smartphone photos.

    Works with :class:`~pyhealth.datasets.PADUFES20Dataset`.

    One sample per image. The image is loaded as RGB (most PAD-UFES-20 PNGs
    are RGBA, which would otherwise give 4-channel tensors) and resized to a
    square ``image_size``. The class list is fixed, so every split has the
    same label indices even when a rare class (MEL has 52 images) is absent.

    Label schemes:
        - ``"pad6"`` (default): the six official diagnoses
          ``ACK, BCC, MEL, NEV, SCC, SEK``.
        - ``"climb5"``: the grouping used by CLIMB,
          ``AKIEC`` (ACK + SCC), ``BCC``, ``MEL``, ``NV`` (NEV) and
          ``OTHER`` (SEK).

    Args:
        label_scheme: ``"pad6"`` or ``"climb5"``.
        image_size: Side length of the resized square image.

    Raises:
        ValueError: If ``label_scheme`` is unknown.

    Examples:
        >>> from pyhealth.tasks import PADUFES20Classification
        >>> task = PADUFES20Classification(label_scheme="climb5")
        >>> task.output_schema["label"][1]["labels"]
        ['AKIEC', 'BCC', 'MEL', 'NV', 'OTHER']
        >>> from pyhealth.datasets import PADUFES20Dataset
        >>> dataset = PADUFES20Dataset(root="/data/pad_ufes_20")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "PADUFES20Classification"

    def __init__(self, label_scheme: str = "pad6", image_size: int = 224) -> None:
        if label_scheme not in _SCHEMES:
            raise ValueError(
                f"label_scheme must be one of {sorted(_SCHEMES)}, got {label_scheme!r}."
            )
        super().__init__()
        self.label_scheme = label_scheme
        self.image_size = int(image_size)
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }
        self.output_schema = {
            "label": ("multiclass", {"labels": list(_SCHEMES[label_scheme])})
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per image of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``pad_ufes_20`` events.

        Returns:
            Samples with ``patient_id``, ``lesion_id``, ``image_id``,
            ``image`` (the PNG path) and ``label`` (a class name).
        """
        samples = []
        for event in patient.get_events(event_type="pad_ufes_20"):
            diagnosis = str(event.diagnostic).strip().upper()
            if diagnosis not in CLIMB5_MAP:
                continue
            label = diagnosis if self.label_scheme == "pad6" else CLIMB5_MAP[diagnosis]
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "lesion_id": str(event.lesion_id),
                    "image_id": event.image_id,
                    "image": event.path,
                    "label": label,
                }
            )
        return samples
