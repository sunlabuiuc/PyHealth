pyhealth.metrics.segmentation
===================================

Dice and IoU metrics for 2D/3D segmentation. The functions accept one mask or
a list of per-case masks, matching the per-sample outputs of
:class:`pyhealth.trainer.Trainer`. The core Dice/IoU computations are backed by
MONAI's ``compute_dice`` and ``compute_iou`` from the optional ``imaging``
extra.

.. autofunction:: pyhealth.metrics.segmentation.dice_score

.. autofunction:: pyhealth.metrics.segmentation.iou_score

.. autofunction:: pyhealth.metrics.segmentation.segmentation_metrics_fn

.. autofunction:: pyhealth.metrics.segmentation.case_level_segmentation_metrics

.. autofunction:: pyhealth.metrics.segmentation.per_class_dice_score

.. autofunction:: pyhealth.metrics.segmentation.per_class_segmentation_metrics

.. autofunction:: pyhealth.metrics.segmentation.remap_class_mask
