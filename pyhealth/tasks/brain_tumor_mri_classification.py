"""Four-class brain tumour classification on the Bhuvaji brain MRI dataset."""

from __future__ import annotations

from typing import Any, ClassVar

from .base_task import BaseTask


class BrainTumorMRIClassification(BaseTask):
    """Classify a brain MRI slice as no tumour, glioma, meningioma or pituitary.

    Works with :class:`~pyhealth.datasets.BrainTumorMRIDataset`.

    One sample per image. The image is loaded in grayscale (``mode="L"``)
    and resized to ``image_size`` by the ``image`` processor. The label is
    the class folder name. The four classes are fixed in the order of
    :data:`~pyhealth.datasets.brain_tumor_mri.BRAIN_TUMOR_MRI_CLASSES`
    (index 0 ``no_tumor``, 1 ``glioma_tumor``, 2 ``meningioma_tumor``,
    3 ``pituitary_tumor``), so a subset keeps the same indices. Each sample
    also carries ``split`` and ``fold`` for
    :func:`~pyhealth.datasets.split_by_strat_fold`.

    Args:
        image_size: Output edge length in pixels. Defaults to 224.

    Examples:
        >>> from pyhealth.datasets import BrainTumorMRIDataset, split_by_strat_fold
        >>> from pyhealth.tasks import BrainTumorMRIClassification
        >>> task = BrainTumorMRIClassification(image_size=128)
        >>> task.output_schema["label"][1]["labels"][0]
        'no_tumor'
        >>> dataset = BrainTumorMRIDataset(root="/data/btm")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> train, val, test = split_by_strat_fold(
        ...     samples, fold_field="fold", check_patient_disjoint=True
        ... )  # doctest: +SKIP
    """

    task_name: str = "BrainTumorMRIClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"mode": "L", "image_size": 224})
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "label": (
            "multiclass",
            {
                "labels": [
                    "no_tumor",
                    "glioma_tumor",
                    "meningioma_tumor",
                    "pituitary_tumor",
                ]
            },
        )
    }

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.input_schema = {
            "image": ("image", {"mode": "L", "image_size": self.image_size})
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per image event.

        Args:
            patient: A PyHealth ``Patient`` with ``images`` events (one
                group of identical images).

        Returns:
            Sample dicts with ``image``, ``label``, ``split`` and ``fold``.
        """
        return [
            {
                "patient_id": patient.patient_id,
                "record_id": event.record_id,
                "image": event.image_path,
                "label": event.label,
                "split": event.split,
                "fold": int(float(event.fold)),
            }
            for event in patient.get_events(event_type="images")
        ]
