"""Breast density (BI-RADS A-D) classification task for CBIS-DDSM."""

from typing import Any, ClassVar

from .base_task import BaseTask
from .cbis_ddsm_classification import _IMAGE_SCHEMA, _check_split, _events_by_image


class CBISDDSMDensityClassification(BaseTask):
    """Image-level breast density (1-4) classification on CBIS-DDSM.

    Works with :class:`~pyhealth.datasets.CBISDDSMDataset`.

    One sample per full-field mammogram (patient, side, view) and official
    split, labelled with the radiologist's ACR BI-RADS breast density from
    the case-description CSVs: 1 (almost entirely fatty) to 4 (extremely
    dense), class indices 0 to 3. All lesions of an image carry the same
    density in the official CSVs. Two test-set rows have density 0, which
    is not a BI-RADS category, so images without a density of 1-4 are
    skipped. The classes are imbalanced (train images: 354 / 932 / 734 /
    400 for densities 1-4).

    Uses the same lazy 512 x 512 single-channel ``dicom`` input as
    :class:`~pyhealth.tasks.CBISDDSMPathologyClassification` (square
    resize, no padding).

    Args:
        split: Keep only images from this official split (``"train"`` or
            ``"test"``). Default None keeps both.

    Examples:
        >>> from pyhealth.datasets import CBISDDSMDataset
        >>> from pyhealth.tasks import CBISDDSMDensityClassification
        >>> CBISDDSMDensityClassification().output_schema["density"]
        ('multiclass', {'labels': [1, 2, 3, 4]})
        >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     CBISDDSMDensityClassification()
        ... )
    """

    task_name: str = "CBISDDSMDensityClassification"
    input_schema: ClassVar[dict[str, Any]] = {"image": _IMAGE_SCHEMA}
    output_schema: ClassVar[dict[str, Any]] = {
        "density": ("multiclass", {"labels": [1, 2, 3, 4]})
    }

    def __init__(self, split: str | None = None) -> None:
        self.split = _check_split(split)
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per mammogram with a density of 1-4.

        Args:
            patient: A PyHealth ``Patient`` with ``lesions`` events.

        Returns:
            Sample dicts with ``image``, ``density`` (1-4) and ``split``.
        """
        samples = []
        for (image_id, split), events in _events_by_image(patient, self.split).items():
            densities = set()
            for event in events:
                try:
                    densities.add(int(float(event.breast_density)))
                except (TypeError, ValueError):
                    continue
            densities &= {1, 2, 3, 4}
            if not densities:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": f"{image_id}_{split}",
                    "split": split,
                    "image": events[0].image_path,
                    "density": max(densities),
                }
            )
        return samples
