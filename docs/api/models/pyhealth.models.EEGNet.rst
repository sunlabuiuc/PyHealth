pyhealth.models.EEGNet
======================

Overview
--------

EEGNet is a compact convolutional network for EEG classification: a temporal
convolution, a depthwise spatial convolution over all channels, a separable
convolution and a linear classifier, with max-norm constraints. It takes one
``(channels, time)`` signal per sample, such as the ``signal`` field of the
EEGBCI, CHB-MIT, TUAB or TUEV tasks, and works with binary, multiclass and
multilabel labels. The defaults are EEGNet-8,2 from the paper; kernel lengths
follow ``sampling_rate`` (the paper uses 128 Hz).

Paper: V. J. Lawhern et al., `EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces <https://doi.org/10.1088/1741-2552/aace8c>`_,
Journal of Neural Engineering 15(5):056013, 2018.

API Reference
-------------

.. autoclass:: pyhealth.models.EEGNet
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.eegnet.Conv2dWithMaxNorm
    :members:
    :show-inheritance:

.. autoclass:: pyhealth.models.eegnet.LinearWithMaxNorm
    :members:
    :show-inheritance:
