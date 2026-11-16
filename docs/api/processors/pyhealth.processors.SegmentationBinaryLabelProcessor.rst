pyhealth.processors.SegmentationBinaryLabelProcessor
====================================================

Binary segmentation label kind (``"segmentation_binary"``). The mask itself
is read by an image processor; this processor tells models and the
``Trainer`` how to activate, score and gather variable-shaped masks. It is
supplied by :attr:`~pyhealth.models.UNet.label_processors` and is not
available as a string schema or through ``get_metrics_fn``.

.. autoclass:: pyhealth.processors.SegmentationBinaryLabelProcessor
    :members:
    :undoc-members:
    :show-inheritance:
