pyhealth.models.UNet
===================================

U-Net models for 2D and 3D segmentation, their Dice/CE loss, plus
sliding-window inference and case-level evaluation helpers. The Dice term is
backed by MONAI's ``DiceLoss`` and sliding-window inference uses MONAI's
``sliding_window_inference``; both are imported lazily from the optional
``imaging`` extra.

.. autoclass:: pyhealth.models.UNet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.UNet2D
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.UNet3D
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.sliding_window_inference

.. autofunction:: pyhealth.models.evaluate_segmentation

.. autoclass:: pyhealth.models.DiceLoss
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.SegmentationLoss
    :members:
    :undoc-members:
    :show-inheritance:
