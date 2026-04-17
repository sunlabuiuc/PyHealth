pyhealth.models.RetinaUNet
===================================

Retina U-Net model for medical image object detection and segmentation.

.. automodule:: pyhealth.models.retina_unet
   :members:
   :undoc-members:
   :show-inheritance:

RetinaUNet: Multi-Task Detection + Segmentation
-----------------------------------------------

RetinaUNet combines a Feature Pyramid Network (FPN) backbone with Retina-style
detection heads and a U-Net-style segmentation head for medical imaging tasks.
The implementation supports both 2D and 3D inputs.

Key Features
^^^^^^^^^^^^

- **Dual-task learning**: Joint object detection and segmentation in one model
- **2D and 3D support**: Works for image slices and volumetric data via ``dim``
- **Anchor-based detection**: Multi-scale anchors over FPN levels ``P2`` to ``P5``
- **Built-in post-processing**: Delta decoding, clipping, and class-wise NMS
- **PyHealth integration**: ``RetinaUNet`` wrapper returns standard loss and output keys

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from pyhealth.models.retina_unet import RetinaUNetCore

    # 2D core model for direct tensor-based experimentation
    model = RetinaUNetCore(in_channels=1, num_classes=2, dim=2)

    # Input: (batch, channels, height, width)
    x = torch.randn(2, 1, 128, 128)
    outputs = model(x)

    detections = outputs["detections"]
    class_logits = outputs["class_logits"]
    bbox_deltas = outputs["bbox_deltas"]
    seg_logits = outputs["segmentation"]
    anchors = outputs["anchors"]

Reference
^^^^^^^^^

If you use Retina U-Net, please cite:

Jaeger, P. F., Isensee, F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H.
Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision
for Medical Object Detection. arXiv:1811.08661.