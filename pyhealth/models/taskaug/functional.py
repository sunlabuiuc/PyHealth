# Adapted from https://github.com/aniruddhraghu/ecg_aug (Apache License 2.0,
# see LICENSE in this folder). Modified for PyHealth: package-relative imports
# GPU device fixes and lint cleanups (the unused spec_aug was removed).
import numpy as np
import torch


def rand_temporal_warp(x, mag, warp_obj):
    """Applies a random smooth warp of the time axis to an ECG batch.

    The magnitude is squared and multiplied by 100 before it is passed on.

    Args:
        x (torch.Tensor): ECG batch of shape ``(batch, leads, length)``.
        mag (torch.Tensor): Per-example magnitude, broadcastable to
            ``(batch, 1, 1)``.
        warp_obj (RandWarpAug): Warp module built for signals of this length.

    Returns:
        torch.Tensor: The warped batch, same shape as ``x``.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.functional import rand_temporal_warp
        >>> from pyhealth.models.taskaug.warp_ops import RandWarpAug
        >>> x = torch.randn(2, 12, 2496)
        >>> rand_temporal_warp(x, torch.ones(2, 1, 1), RandWarpAug([2496])).shape
        torch.Size([2, 12, 2496])
    """
    mag = 100*(mag**2)
    return warp_obj(x, mag)

def baseline_wander(x, mag):
    """Adds a slow sinusoidal baseline drift, like breathing artifacts, to each lead.

    The drift has a random frequency and phase and an amplitude of
    ``0.25 * sigmoid(mag)`` times a random factor in [0, 1).

    Args:
        x (torch.Tensor): ECG batch of shape ``(batch, leads, length)``.
        mag (torch.Tensor): Per-example magnitude, broadcastable to
            ``(batch, 1, 1)``.

    Returns:
        torch.Tensor: The batch with the drift added, same shape as ``x``.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.functional import baseline_wander
        >>> x = torch.randn(4, 12, 2496)
        >>> baseline_wander(x, torch.zeros(4, 1, 1)).shape
        torch.Size([4, 12, 2496])
    """
    BS, _, L = x.shape

    # form baseline drift
    strength = 0.25*torch.sigmoid(mag) * (torch.rand(BS).to(x.device).view(BS,1,1))
    strength = strength.view(BS, 1, 1)

    frequency = ((torch.rand(BS) * 20 + 10) * 10 / 60).view(BS, 1, 1)  # typical breaths per second for an adult
    phase = (torch.rand(BS) * 2 * np.pi).view(BS, 1, 1)
    drift = strength*torch.sin(torch.linspace(0, 1, L).view(1, 1, -1) * frequency.float() + phase.float()).to(x.device)
    return x + drift

def gaussian_noise(x, mag):
    """Adds Gaussian noise to each lead.

    The noise standard deviation is ``0.25 * sigmoid(mag)`` times the standard
    deviation of the lead.

    Args:
        x (torch.Tensor): ECG batch of shape ``(batch, leads, length)``.
        mag (torch.Tensor): Per-example magnitude, broadcastable to
            ``(batch, 1, 1)``.

    Returns:
        torch.Tensor: The noisy batch, same shape as ``x``.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.functional import gaussian_noise
        >>> x = torch.randn(4, 12, 2496)
        >>> gaussian_noise(x, torch.zeros(4, 1, 1)).shape
        torch.Size([4, 12, 2496])
    """
    BS, C, L = x.shape
    stdval = torch.std(x, dim=2).view(BS, C, 1).detach()
    noise = 0.25*stdval*torch.sigmoid(mag)*torch.randn(BS, C, L).to(x.device)
    return x + noise

def rand_crop(x, mag):
    """Sets one random contiguous segment of the signal to zero in all leads.

    Args:
        x (torch.Tensor): ECG batch of shape ``(batch, leads, length)``.
        mag (torch.Tensor): One-element tensor with the fraction of the signal to
            zero; the segment has ``int(mag * length)`` samples.

    Returns:
        torch.Tensor: A copy of ``x`` with the segment zeroed.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.functional import rand_crop
        >>> x = torch.ones(1, 12, 100)
        >>> out = rand_crop(x, torch.tensor([0.1]))
        >>> int((out == 0).sum())
        120
    """
    x_aug = x.clone()
    # get shapes
    L = x.shape[-1]
    mag = mag.item()

    nmf = int(mag*L)
    start = torch.randint(0, L-nmf,[1]).long()
    end = (start + nmf).long()
    x_aug[:, :, start:end] = 0.
    return x_aug


def rand_displacement(x, mag, warp_obj):
    """Shifts the whole signal in time by a random amount.

    The magnitude is squared and multiplied by 100; the shift, in samples, is
    drawn uniformly between minus and plus that value.

    Args:
        x (torch.Tensor): ECG batch of shape ``(batch, leads, length)``.
        mag (torch.Tensor): Per-example magnitude, broadcastable to
            ``(batch, 1, 1)``.
        warp_obj (DispAug): Displacement module built for signals of this length.

    Returns:
        torch.Tensor: The shifted batch, same shape as ``x``.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.functional import rand_displacement
        >>> from pyhealth.models.taskaug.warp_ops import DispAug
        >>> x = torch.randn(2, 12, 2496)
        >>> rand_displacement(x, torch.ones(2, 1, 1), DispAug([2496])).shape
        torch.Size([2, 12, 2496])
    """
    disp_mag = 100*(mag**2)
    return warp_obj(x, disp_mag)

def magnitude_scale(x, mag):
    """Scales the amplitude of each ECG by a random factor.

    The factor is ``sigmoid(mag)`` times a random value in [0.75, 1.25].

    Args:
        x (torch.Tensor): ECG batch of shape ``(batch, leads, length)``.
        mag (torch.Tensor): Per-example magnitude, broadcastable to
            ``(batch, 1, 1)``.

    Returns:
        torch.Tensor: The scaled batch, same shape as ``x``.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.functional import magnitude_scale
        >>> x = torch.randn(4, 12, 2496)
        >>> magnitude_scale(x, torch.zeros(4, 1, 1)).shape
        torch.Size([4, 12, 2496])
    """
    BS = x.shape[0]
    strength = torch.sigmoid(mag)*(-0.5 * (torch.rand(BS).to(x.device)).view(BS,1,1) + 1.25)
    strength = strength.view(BS, 1, 1)
    return x*strength



