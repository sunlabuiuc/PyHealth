pyhealth.tasks.PolypDetection
=============================

Polyp detection on :class:`~pyhealth.datasets.PolypDataset`: one sample per
colonoscopy frame, with the image path and the ground-truth polyps as COCO
``[x, y, width, height]`` boxes. Score detections with
:func:`~pyhealth.metrics.detection_metrics_fn`.

.. autoclass:: pyhealth.tasks.PolypDetection
    :members:
    :undoc-members:
    :show-inheritance:
