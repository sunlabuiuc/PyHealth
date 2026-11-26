pyhealth.models.UNet3Plus
=========================

UNet 3+ (Huang et al., ICASSP 2020, doi: 10.1109/ICASSP40776.2020.9053405)
for one-dimensional segmentation. Park et al. (CHIL 2025,
https://proceedings.mlr.press/v287/park25a.html) found it the best of seven
networks for P / QRS / T delineation on LUDB.

.. autoclass:: pyhealth.models.UNet3Plus
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.segmentation_iou

.. autofunction:: pyhealth.models.unet3plus.dice_loss
