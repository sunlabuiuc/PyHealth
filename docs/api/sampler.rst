Samplers
===============

Samplers decide which examples or spatial locations a training step sees.
:class:`~pyhealth.sampler.PatchSampler` is used by
:class:`~pyhealth.datasets.PatchDataset` for segmentation patch sampling and
nnU-Net-style foreground oversampling.

.. toctree::
    :maxdepth: 3

    sampler/pyhealth.sampler.PatchSampler
