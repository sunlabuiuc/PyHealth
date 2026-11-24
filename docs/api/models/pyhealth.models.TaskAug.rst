pyhealth.models.TaskAug
===================================

TaskAug: data augmentation policies for ECGs, learned per task and with
class-specific strengths
(`Raghu et al., CHIL 2022 <https://proceedings.mlr.press/v174/raghu22a.html>`_).
The classifier trains on augmented signals; the policy is trained on the
validation loss by implicit differentiation.

Signal transformations
----------------------

The transformations are adapted from the paper authors'
`ecg_aug repository <https://github.com/aniruddhraghu/ecg_aug>`_
(Apache-2.0). The source names below link to the implementations used as
references.

Inputs and outputs have shape ``(batch, channels, length)``. At each policy
stage, one operation is selected for the whole batch. For operations with a
learned magnitude, each signal uses the parameter for its binary class at
that stage. These parameters are transformed as described below; a magnitude
of zero does not generally mean no augmentation.

* **Temporal warp (``warp``):** changes local timing by reading the signal
  along a smooth random displacement field. Each signal has its own field,
  shared by its leads to preserve their alignment. Initial field noise is
  scaled by ``100 * magnitude**2``, integrated at quarter resolution,
  upsampled and smoothed. This scale is not a maximum displacement.
  See ``rand_temporal_warp`` in `functional.py`_ and ``RandWarpAug`` in
  `warp_ops.py`_.
* **Baseline wander (``wander``):** adds a slowly varying sinusoidal offset,
  with independent amplitude, phase and frequency draws per signal and the
  same drift across its leads. Amplitude is at most
  ``0.25 * sigmoid(magnitude)`` in input-signal units. Following the reference
  code, the sinusoid spans less than one cycle over normalized signal time;
  its frequency is not specified in Hz. See ``baseline_wander`` in
  `functional.py`_.
* **Gaussian noise (``noise``):** adds independent noise at each lead and
  time point. Noise standard deviation is
  ``0.25 * sigmoid(magnitude)`` times that lead's temporal standard deviation.
  See ``gaussian_noise`` in `functional.py`_.
* **Time mask (``mask``):** zeroes a contiguous window without shortening the
  signal. One window is shared by the whole batch and all leads. Its width
  is a fixed fraction of signal length, not learned: PyHealth defaults to
  10%, while the reference ``RandCrop`` class defaults to 5%. The reference
  names this operation ``rand_crop``, although it masks rather than crops.
  See ``rand_crop`` in `functional.py`_ and ``RandCrop`` in `operations.py`_.
* **Temporal displacement (``displace``):** shifts a whole signal by a
  random offset of up to ``100 * magnitude**2`` samples in either direction.
  The offset is shared by its leads and constant over time, unlike a warp.
  See ``rand_displacement`` in `functional.py`_ and ``DispAug`` in
  `warp_ops.py`_.
* **Magnitude scale (``scale``):** multiplies a signal by
  ``sigmoid(magnitude) * Uniform(0.75, 1.25)``. One factor is shared across
  its leads and time points. With the initial magnitude of zero, factors
  range from 0.375 to 0.625, rather than being centered on one. See
  ``magnitude_scale`` in `functional.py`_.
* **No operation (``noop``):** leaves the signal unchanged and has no
  magnitude parameter. See ``NoOp`` in `operations.py`_.

Warp and displacement use linear interpolation and read zeros beyond the
signal boundaries. Their offsets are measured in samples, so their physical
duration depends on the sampling rate. All operations preserve tensor shape.

.. _functional.py: https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py
.. _warp_ops.py: https://github.com/aniruddhraghu/ecg_aug/blob/main/warp_ops.py
.. _operations.py: https://github.com/aniruddhraghu/ecg_aug/blob/main/operations.py

API reference
-------------

.. autoclass:: pyhealth.models.TaskAug
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TaskAugPolicy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ResNet1D
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.train_taskaug

.. autofunction:: pyhealth.models.hyper_step
