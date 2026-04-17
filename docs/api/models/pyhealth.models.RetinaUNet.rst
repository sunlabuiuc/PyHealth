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
    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import RetinaUNet

    # Build a minimal sample dataset
    images = torch.randn(2, 1, 128, 128)
    masks = torch.zeros(2, 1, 128, 128)
    masks[0, 0, 20:40, 20:40] = 1

    samples = [
        {
            "patient_id": f"p_{i}",
            "visit_id": f"v_{i}",
            "images": images[i],
            "gt_seg_masks": masks[i],
            "gt_boxes_list": [torch.tensor([20, 20, 40, 40])],
            "gt_classes_list": [torch.tensor([1])],
        }
        for i in range(2)
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"images": "tensor"},
        output_schema={
            "gt_seg_masks": "tensor",
            "gt_boxes_list": "raw",
            "gt_classes_list": "raw",
        },
        dataset_name="Demo",
        task_name="ObjectDetection",
    )

    model = RetinaUNet(dataset=dataset, num_classes=2, dim=2)
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # Training: returns all losses and raw predictions
    outputs = model(**batch)
    loss = outputs["loss"]            # scalar combined loss
    class_loss = outputs["class_loss"]
    bbox_loss = outputs["bbox_loss"]
    seg_loss = outputs["seg_loss"]

    # Inference: pass only images — labels are optional
    with torch.no_grad():
        outputs = model(images=batch["images"])

    det_bboxes = outputs["det_bboxes"]       # post-NMS boxes per image
    class_logits = outputs["class_logits"]   # (B, N_anchors, num_classes)
    bbox_deltas = outputs["bbox_deltas"]     # (B, N_anchors, 4)
    seg_logits = outputs["seg_logits"]       # (B, 2, H, W)
    anchors = outputs["anchors"]             # (N_anchors, 4)

Reference
^^^^^^^^^

If you use Retina U-Net, please cite:

Jaeger, P. F., Isensee, F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H.
Retina U-Net: Embarrassingly Simple Exploitation of Segmentation Supervision
for Medical Object Detection. arXiv:1811.08661.