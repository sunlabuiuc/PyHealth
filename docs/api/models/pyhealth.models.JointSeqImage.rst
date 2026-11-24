pyhealth.models.JointSeqImage
=============================

A joint sequence and image model for irregular medical time series. A
bidirectional recurrent generator, trained against a discriminator, imputes
the missing values for the sequence features; the series are drawn as grid
images (line graphs, spectra, Gramian angular, Markov transition or
recurrence fields) and encoded by a pretrained Swin Transformer; and a
margin contrastive loss and a clustering loss align the two in a joint
representation (`Chen et al., AAAI 2025 <https://arxiv.org/abs/2502.06134>`_).

See ``examples/joint_seq_image_physionet2012_mortality.py`` for the paper's
PhysioNet 2012 mortality experiment.

.. autoclass:: pyhealth.models.JointSeqImage
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.joint_seq_image.DecayImputer
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.joint_seq_image.MaskDiscriminator
    :members:
    :show-inheritance:

.. autofunction:: pyhealth.models.joint_seq_image.time_lags

.. autofunction:: pyhealth.models.joint_seq_image.reverse_padded

.. autofunction:: pyhealth.models.joint_seq_image.nt_xent

.. autofunction:: pyhealth.models.joint_seq_image.margin_contrastive

.. autofunction:: pyhealth.models.joint_seq_image.focal_loss

Time-series images
------------------

.. automodule:: pyhealth.models.time_series_images
    :members:
