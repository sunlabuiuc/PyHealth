pyhealth.interpret.methods.gradcam
==================================

Overview
--------

Grad-CAM provides class-conditional heatmaps for CNN-based image
classification models in PyHealth. It uses gradients from a target
convolutional layer to highlight which image regions contributed most to the
selected prediction.

This method is intended for:

- CNN image classification models
- chest X-ray and other medical imaging workflows built on PyHealth image tasks
- models that return either ``logit`` or ``y_prob``

Usage Notes
-----------

1. **CNN model**: Grad-CAM requires a 4D convolutional activation map from
   the target layer.
2. **Target layer**: You can pass either an ``nn.Module`` directly or a dotted
   string path such as ``"model.layer4.1.conv2"``.
3. **Class selection**: If ``class_index`` is omitted, Grad-CAM uses the
   predicted class. For single-output binary models, it attributes to that
   scalar output.
4. **Gradients required**: Do not call ``attribute()`` inside
   ``torch.no_grad()``.
5. **Return shape**: ``attribute()`` returns ``{input_key: cam}`` where the CAM
   tensor has shape ``[B, H, W]``.

Quick Start
-----------

.. code-block:: python

    from pyhealth.interpret.methods import GradCAM
    from pyhealth.interpret.utils import visualize_image_attr

    gradcam = GradCAM(
        model,
        target_layer=model.model.layer4[-1].conv2,
        input_key="image",
    )
    cams = gradcam.attribute(**batch)
    image, heatmap, overlay = visualize_image_attr(
        image=batch["image"][0],
        attribution=cams["image"][0],
    )

For a complete script example, see:
``examples/cxr/gradcam_cxr_tutorial.py``

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.gradcam.GradCAM
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
