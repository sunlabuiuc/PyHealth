pyhealth.metrics.detection
==========================

COCO average precision for bounding-box detection. The values match
``pycocotools`` ``COCOeval`` (``iouType="bbox"``, area range ``"all"``)
without depending on it. Boxes are COCO ``[x, y, width, height]``.

.. autofunction:: pyhealth.metrics.detection.detection_metrics_fn

.. autofunction:: pyhealth.metrics.detection.box_iou
