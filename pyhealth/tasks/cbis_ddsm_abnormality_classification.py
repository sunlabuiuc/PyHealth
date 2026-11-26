"""Mass vs calcification classification task for CBIS-DDSM."""

from typing import Any, ClassVar

from .base_task import BaseTask
from .cbis_ddsm_classification import _IMAGE_SCHEMA, _check_split, _events_by_image


class CBISDDSMAbnormalityClassification(BaseTask):
    """Image-level abnormality type (mass vs calcification) on CBIS-DDSM.

    Works with :class:`~pyhealth.datasets.CBISDDSMDataset`.

    One sample per full-field mammogram (patient, side, view) and official
    split. The label is 1 when the image's lesions come from the mass CSVs
    and 0 when they come from the calcification CSVs, following the
    dataset's own mass / calcification division (Lee et al., 2017). Images
    with lesions of both types in the same official split (42 in the full
    collection: 38 train, 4 test) are skipped, since they have no single
    type. Each sample keeps the official ``split``.

    Uses the same lazy 512 x 512 single-channel ``dicom`` input as
    :class:`~pyhealth.tasks.CBISDDSMPathologyClassification` (square
    resize, no padding).

    Args:
        split: Keep only images from this official split (``"train"`` or
            ``"test"``). Default None keeps both.

    Examples:
        >>> from pyhealth.datasets import CBISDDSMDataset
        >>> from pyhealth.tasks import CBISDDSMAbnormalityClassification
        >>> CBISDDSMAbnormalityClassification().output_schema
        {'mass': 'binary'}
        >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     CBISDDSMAbnormalityClassification()
        ... )
    """

    task_name: str = "CBISDDSMAbnormalityClassification"
    input_schema: ClassVar[dict[str, Any]] = {"image": _IMAGE_SCHEMA}
    output_schema: ClassVar[dict[str, Any]] = {"mass": "binary"}

    def __init__(self, split: str | None = None) -> None:
        self.split = _check_split(split)
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per single-type mammogram of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``lesions`` events.

        Returns:
            Sample dicts with ``image``, ``mass`` (1 mass, 0 calcification)
            and ``split``.
        """
        samples = []
        for (image_id, split), events in _events_by_image(patient, self.split).items():
            kinds = {str(e.abnormality_type) for e in events}
            if len(kinds) != 1:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": f"{image_id}_{split}",
                    "split": split,
                    "image": events[0].image_path,
                    "mass": int(kinds == {"mass"}),
                }
            )
        return samples
