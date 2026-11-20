"""Molecular subtype classification task for the CMMD mammography dataset."""

from typing import Any, ClassVar

from .base_task import BaseTask

#: Molecular subtypes of the CMMD clinical sheet, in label-index order.
CMMD_SUBTYPES = ("Luminal A", "Luminal B", "HER2-enriched", "triple negative")


class CMMDSubtypeClassification(BaseTask):
    """Classify the molecular subtype of a malignant CMMD mammogram.

    Works with :class:`~pyhealth.datasets.CMMDDataset`.

    The CMMD D2 cohort (765 sheet rows: 753 malignant, 12 benign) gives
    the molecular subtype of 749 of its malignant rows (Luminal A, Luminal B, HER2-enriched, triple negative; Cui et al.,
    2021). One sample per mammogram (one breast, one CC or MLO view) whose
    own breast has a clinical row with a subtype, matched on the DICOM
    laterality, so 4 classes with the labels of :data:`CMMD_SUBTYPES`.
    Images without a subtype (all of D1, the 12 D2 benign rows, the 4 D2
    malignant rows with no subtype, and unlabelled contralateral breasts)
    are skipped. The classes are imbalanced: the
    sheet has 152 / 376 / 135 / 86 rows. Split by ``patient_id``.

    The image is read lazily with the ``dicom`` processor at 512 x 512,
    single channel (the same settings as
    :class:`~pyhealth.tasks.CMMDMalignancyClassification`).

    Examples:
        >>> from pyhealth.datasets import CMMDDataset
        >>> from pyhealth.tasks import CMMDSubtypeClassification
        >>> CMMDSubtypeClassification().output_schema["subtype"][0]
        'multiclass'
        >>> dataset = CMMDDataset(root="/path/to/cmmd")  # doctest: +SKIP
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     CMMDSubtypeClassification()
        ... )
    """

    task_name: str = "CMMDSubtypeClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": (
            "dicom",
            {
                "lazy": True,
                "image_size": 512,
                "mode": "L",
                "normalize": True,
                "mean": [0.5],
                "std": [0.5],
            },
        )
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "subtype": ("multiclass", {"labels": list(CMMD_SUBTYPES)})
    }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per mammogram of a breast with a subtype.

        Args:
            patient: A PyHealth ``Patient`` with ``images`` events.

        Returns:
            Sample dicts with ``image``, ``subtype``, ``laterality``,
            ``view`` and ``cohort``.
        """
        samples = []
        for event in patient.get_events(event_type="images"):
            subtype = str(event.subtype or "").strip()
            if subtype not in CMMD_SUBTYPES:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": event.image_path,
                    "subtype": subtype,
                    "laterality": event.laterality,
                    "view": event.view,
                    "cohort": event.cohort,
                }
            )
        return samples
