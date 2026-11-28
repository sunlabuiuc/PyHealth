"""Polyp detection on colonoscopy frames (:class:`~pyhealth.datasets.PolypDataset`)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from .base_task import BaseTask


class PolypDetection(BaseTask):
    """Find every polyp in a colonoscopy frame, as a bounding box.

    One sample per image: ``image`` is the image path, read by the model,
    and ``boxes`` the ground-truth polyps as COCO ``[x, y, width, height]``
    boxes in pixels (a list, one entry per polyp). Samples also carry
    ``subset`` (the test set), ``width`` and ``height``. Both fields use the
    ``"raw"`` processor, so a batch holds Python lists; score predictions
    with :func:`~pyhealth.metrics.detection_metrics_fn`.

    Examples:
        >>> from pyhealth.datasets import PolypDataset
        >>> dataset = PolypDataset(root="/data/DATA/POLYP")  # doctest: +SKIP
        >>> samples = dataset.set_task(PolypDetection())  # doctest: +SKIP
        >>> samples[0]["boxes"]  # doctest: +SKIP
        [[250.0, 304.0, 129.0, 116.0]]
    """

    task_name: str = "PolypDetection"
    input_schema: ClassVar[dict[str, str]] = {"image": "raw"}
    output_schema: ClassVar[dict[str, str]] = {"boxes": "raw"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = []
        for event in patient.get_events(event_type="polyp"):
            samples.append({
                "patient_id": patient.patient_id,
                "image": event.path,
                "boxes": json.loads(event.boxes),
                "subset": event.subset,
                "width": int(float(event.width)),
                "height": int(float(event.height)),
            })
        return samples
