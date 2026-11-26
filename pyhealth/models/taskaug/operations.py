# Adapted from https://github.com/aniruddhraghu/ecg_aug (Apache License 2.0,
# see LICENSE in this folder). Modified for PyHealth: package-relative imports
# GPU device fixes and lint cleanups (the unused spec_aug was removed).
############ Adapted from  https://github.com/moskomule/dda #############


""" Operations

"""
import torch
from scipy.special import logit
from torch import nn
from torch.distributions import Bernoulli, RelaxedBernoulli

from . import warp_ops
from .functional import (
    baseline_wander,
    gaussian_noise,
    magnitude_scale,
    rand_crop,
    rand_displacement,
    rand_temporal_warp,
)


class _Operation(nn.Module):
    """ Base class of operation

    :param operation:
    :param initial_magnitude:
    :param initial_probability:
    :param learn_magnitude:
    :param learn_probability:
    :param temperature: Temperature for RelaxedBernoulli distribution used during training
    """

    def __init__(self,
                 operation,
                 initial_magnitude,
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__()
        self.operation = operation

        if initial_magnitude is not None and learn_magnitude:
            self.magnitude = nn.Parameter(torch.Tensor(initial_magnitude))
        else:
            self.magnitude = torch.Tensor(initial_magnitude)
        
        if learn_probability:
            self.probability = nn.Parameter(torch.Tensor([float(logit(i)) for i in initial_probability]))
        else:
            self.probability = torch.Tensor([float(logit(i)) for i in initial_probability])

        assert 0 < temperature
        self.temperature = temperature


    def forward(self,input, label):
        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor. 
        # First repeat in BS dimension
        BS = input.shape[0]
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries. 
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel)
        mask = mask.view(BS, 1, 1)
        retval = (mask * transformed + (1 - mask) * input)
        return retval

    def get_mask(self, label,
                 batch_size=None):

        prob = torch.sigmoid(self.probability).to(label.device).unsqueeze(0)
        prob = prob.repeat(batch_size, 1)
        prob = 0*prob.sum() + prob[torch.arange(batch_size, device=label.device), label.long()]
        if self.training:
            return RelaxedBernoulli(self.temperature, prob).rsample()
        else:
            return Bernoulli(prob).sample()


        
        
class NoOp(_Operation):
    """Operation that returns its input unchanged.

    It lets a TaskAug stage choose to apply no augmentation.

    Args:
        initial_magnitude (tuple[float, ...]): Unused placeholder values; the operation does nothing.
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import NoOp
        >>> op = NoOp()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> torch.equal(op(x, y), x)
        True
    """
    def __init__(self,
                 initial_magnitude=(2., 2.),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=False,
                 learn_probability=False,
                 temperature = 0.1,
                ):
        super().__init__(None, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)
        
    def forward(self, input, label):
        return input


class RandTemporalWarp(_Operation):
    """Applies a random smooth warp of the time axis to an ECG.

    All leads of an ECG share the warp. Signals must have 2496 samples.

    Args:
        initial_magnitude (tuple[float, ...]): Starting magnitude, one value per class label.
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import RandTemporalWarp
        >>> op = RandTemporalWarp()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> op(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 initial_magnitude=(2., 2.),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(rand_temporal_warp, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

        # create the warp obj here.
        self.warp_obj = warp_ops.RandWarpAug([2496])

    def forward(self,input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor. 
        # First repeat in BS dimension
        BS = input.shape[0]
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries. 
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel, self.warp_obj)
        B = transformed.shape[0]
        mask = mask.view(B, 1, 1).to(input.device, dtype=input.dtype)
        retval = (mask * transformed + (1 - mask) * input)
        return retval



class BaselineWander(_Operation):
    """Adds a slow sinusoidal baseline drift to an ECG.

    The drift mimics breathing artifacts; see ``baseline_wander``.

    Args:
        initial_magnitude (tuple[float, ...]): Starting magnitude, one value per class label.
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import BaselineWander
        >>> op = BaselineWander()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> op(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 initial_magnitude=(0.0, 0.0),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(baseline_wander, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)


class GaussianNoise(_Operation):
    """Adds Gaussian noise to an ECG.

    The noise is scaled by the standard deviation of each lead; see ``gaussian_noise``.

    Args:
        initial_magnitude (tuple[float, ...]): Starting magnitude, one value per class label.
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import GaussianNoise
        >>> op = GaussianNoise()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> op(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 initial_magnitude=(0.0, 0.0),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(gaussian_noise, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

class RandCrop(_Operation):
    """Zeroes one random segment of an ECG.

    The segment covers all leads; see ``rand_crop``.

    Args:
        initial_magnitude (tuple[float, ...]): One value, the fraction of the signal that is zeroed (0.05 is 5%).
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import RandCrop
        >>> op = RandCrop()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> op(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 initial_magnitude=(0.05,),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=False,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(rand_crop, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

    def forward(self,input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device)
        transformed = self.operation(input, mag)
        B = transformed.shape[0]
        mask = mask.view(B, 1, 1).to(input.device, dtype=input.dtype)
        retval = (mask * transformed + (1 - mask) * input)
        return retval



class RandDisplacement(_Operation):
    """Shifts an ECG in time by a random amount.

    Signals must have 2496 samples; see ``rand_displacement``.

    Args:
        initial_magnitude (tuple[float, ...]): Starting magnitude, one value per class label.
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import RandDisplacement
        >>> op = RandDisplacement()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> op(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 initial_magnitude=(0.5, 0.5),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(rand_displacement, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)

    # create the warp obj here.
        self.warp_obj = warp_ops.DispAug([2496])

    def forward(self,input, label):

        mask = self.get_mask(label, input.size(0)).to(input.device)
        mag = self.magnitude.to(input.device).unsqueeze(0)
        # we need a per-ex mag based on the class label. Right now, the mag is a (1x2) tensor. 
        # First repeat in BS dimension
        BS = input.shape[0]
        mag = mag.repeat(BS, 1)
        # Now it is BS x 2, or BS by class num more generally. Select out the relevant entries. 
        # Also add a sum over all elems to make sure we don't get an error in autograd.
        mag_rel = 0*mag.sum() + mag[torch.arange(BS), label.long()]
        mag_rel = mag_rel.view(BS, 1, 1)
        transformed = self.operation(input, mag_rel, self.warp_obj)
        B = transformed.shape[0]
        mask = mask.view(B, 1, 1).to(input.device, dtype=input.dtype)
        retval = (mask * transformed + (1 - mask) * input)
        return retval

class MagnitudeScale(_Operation):
    """Scales the amplitude of an ECG by a random factor.

    See ``magnitude_scale``.

    Args:
        initial_magnitude (tuple[float, ...]): Starting magnitude, one value per class label.
        initial_probability (tuple[float, ...]): Starting probability of applying
            the operation, one value per class label.
        learn_magnitude (bool): Whether the magnitudes are trained.
        learn_probability (bool): Whether the probabilities are trained.
        temperature (float): Temperature of the relaxed Bernoulli distribution
            that decides, during training, whether the operation is applied.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.operations import MagnitudeScale
        >>> op = MagnitudeScale()
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> op(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 initial_magnitude=(0.0, 0.0),
                 initial_probability=(0.9999999, 0.9999999),
                 learn_magnitude=True,
                 learn_probability=True,
                 temperature = 0.1,
                ):
        super().__init__(magnitude_scale, initial_magnitude, initial_probability, learn_magnitude,
                                     learn_probability, temperature)


