"""Malignancy classification task for the CMMD mammography dataset."""

from typing import Any, ClassVar

from .base_task import BaseTask


class CMMDMalignancyClassification(BaseTask):
    """Classify CMMD mammograms as malignant (1) or benign (0).

    Works with :class:`~pyhealth.datasets.CMMDDataset`.

    One sample per mammogram (one breast, one CC or MLO view). The label is
    the biopsy-confirmed ``classification`` of that breast's clinical row,
    matched on the DICOM laterality. Images with ``has_label == 0`` are
    skipped: a breast without a clinical row (usually the healthy
    contralateral breast of a one-sided case), or a row whose
    ``classification`` is neither Benign nor Malignant. CLIMB instead gives
    unmatched breasts the patient's label. Split by ``patient_id``.

    CBIS-DDSM has a similar benign/malignant task
    (:class:`~pyhealth.tasks.CBISDDSMPathologyClassification`), but that task
    reads CBIS-DDSM's abnormality events, so CMMD needs its own task over its
    per-breast image events.

    The image is read lazily with the ``dicom`` processor at 512 x 512,
    single channel. The processor resizes to a square, so the 2294 x 1914
    images are slightly squashed rather than padded.

    Examples:
        >>> from pyhealth.datasets import CMMDDataset
        >>> from pyhealth.tasks import CMMDMalignancyClassification
        >>> CMMDMalignancyClassification().output_schema
        {'malignant': 'binary'}
        >>> dataset = CMMDDataset(root="/path/to/cmmd")  # doctest: +SKIP
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     CMMDMalignancyClassification()
        ... )
    """

    task_name: str = "CMMDMalignancyClassification"
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
    output_schema: ClassVar[dict[str, Any]] = {"malignant": "binary"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per labelled mammogram of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``images`` events.

        Returns:
            Sample dicts with ``image``, ``malignant``, ``laterality``,
            ``view`` and ``cohort``.
        """
        samples = []
        for event in patient.get_events(event_type="images"):
            if int(float(event.has_label)) != 1:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": event.image_path,
                    "malignant": int(float(event.label)),
                    "laterality": event.laterality,
                    "view": event.view,
                    "cohort": event.cohort,
                }
            )
        return samples
