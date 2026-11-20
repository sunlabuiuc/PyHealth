Samplers
===============

Samplers decide which examples or spatial locations a training step sees.
:class:`~pyhealth.sampler.PatchSampler` is used by
:class:`~pyhealth.datasets.PatchDataset` for segmentation patch sampling and
nnU-Net-style foreground oversampling.
:class:`~pyhealth.sampler.GroupBalancedSampler` resamples training data so
every subgroup of a sensitive attribute is drawn equally often.

.. toctree::
    :maxdepth: 3

    sampler/pyhealth.sampler.PatchSampler
    sampler/pyhealth.sampler.GroupBalancedSampler
