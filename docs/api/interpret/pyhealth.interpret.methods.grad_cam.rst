pyhealth.interpret.methods.grad_cam
===================================

Overview
--------

Grad-CAM (Gradient-weighted Class Activation Mapping) explains a prediction of
a convolutional network with a heatmap over the input. It takes the feature
maps of a convolutional layer, weights each channel by the gradient of the
target class score averaged over its positions, sums them and keeps the
positive part. Regions that raise the class score light up.

The implementation follows Selvaraju et al. (2017), "Grad-CAM: Visual
Explanations from Deep Networks via Gradient-based Localization"
(https://arxiv.org/abs/1610.02391), and matches the reference implementation
`pytorch-grad-cam <https://github.com/jacobgil/pytorch-grad-cam>`_.

Grad-CAM needs convolutional feature maps. For attention models such as
:class:`~pyhealth.models.Transformer` or a ViT, use
:class:`~pyhealth.interpret.methods.CheferRelevance` or
:class:`~pyhealth.interpret.methods.AttentionRollout` instead.

Supported Models
----------------

- :class:`~pyhealth.models.TorchvisionModel` with a ResNet or DenseNet backbone (images). Pass ``target_layers="model.layer4"`` for a ResNet (its last residual stage, as in the paper) or ``target_layers="model.features"`` for a DenseNet; the default, the last convolution, only sees the final layer's new channels.
- :class:`~pyhealth.models.SparcNet` and other single-input 1D CNNs (signals, ``[batch, channels, time]``).
- :class:`~pyhealth.models.CNN`, for every feature at once: sequences get one score per code, time series one score per time step, images one score per pixel.
- Any other model, by passing ``target_layers`` explicitly.

Key Features
------------

- **Class-specific**: explains the predicted class by default, or any class with ``target_class_idx``
- **Input-shaped output**: maps are resized to the input and repeated over its other axes, so attributions have the shape of each input; ``upsample=False`` returns the raw maps
- **Safe to call anywhere**: gradients come from ``torch.autograd.grad``, so it works inside ``torch.no_grad()`` and leaves the model's ``.grad`` fields untouched; hooks are always removed
- **Frozen backbones**: works even when the layers before the target don't require gradients

Usage Notes
-----------

1. **Resolution**: a deep layer can be much coarser than the input (7x7 for a ResNet on a 224x224 image, a few time steps for SparcNet on a signal), and the resized map is equally coarse. Choose an earlier layer for finer maps.
2. **Default layer**: for models other than ``CNN``, the default is the last registered convolution. If that convolution isn't used in the forward pass, ``attribute`` raises; pass ``target_layers`` instead.
3. **Mismatched layouts**: if a layer's map can't be lined up with the input (for example a spectrogram model fed a raw signal, such as ``ContraWR``), or the input tensor can't be found, ``attribute`` raises ``ValueError``; use ``upsample=False`` to get the raw maps.

Quick Start
-----------

.. code-block:: python

    from pyhealth.interpret.methods import GradCAM
    from pyhealth.interpret.utils import visualize_image_attr
    from pyhealth.datasets import get_dataloader

    # Assume a trained TorchvisionModel("resnet18") `model` and a `test_dataset`
    grad_cam = GradCAM(model, target_layers="model.layer4")

    batch = next(iter(get_dataloader(test_dataset, batch_size=1, shuffle=False)))
    cam = grad_cam.attribute(**batch)["image"]  # same shape as batch["image"]

    _, _, overlay = visualize_image_attr(
        image=batch["image"][0], attribution=cam[0, 0]
    )

See ``examples/cxr/covid19cxr_grad_cam.py`` for an end-to-end example on chest
X-rays, with one heatmap per class.

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.GradCAM
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
