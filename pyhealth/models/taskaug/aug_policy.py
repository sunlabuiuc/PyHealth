# Adapted from https://github.com/aniruddhraghu/ecg_aug (Apache License 2.0,
# see LICENSE in this folder). Modified for PyHealth: package-relative imports
# GPU device fixes and lint cleanups (the unused spec_aug was removed).
############ Adapted from  https://github.com/moskomule/dda #############


import random
from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Categorical, RelaxedOneHotCategorical

from .operations import (
    BaselineWander,
    GaussianNoise,
    MagnitudeScale,
    NoOp,
    RandCrop,
    RandDisplacement,
    RandTemporalWarp,
)


class SubPolicyStage(nn.Module):
    """One stage of the TaskAug policy: a learned choice among operations.

    In training mode one operation is chosen from a relaxed categorical
    distribution over learned logits, and its output is scaled by a factor equal to
    1 that passes the gradient to the logits. In evaluation mode an operation is
    sampled from the categorical distribution.

    Args:
        operations (torch.nn.ModuleList): The candidate augmentation operations.
        temperature (float): Temperature used by :attr:`weights`. Defaults to 0.05.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.aug_policy import SubPolicyStage, all_ops
        >>> stage = SubPolicyStage(torch.nn.ModuleList(all_ops(learn_prob=False)))
        >>> len(stage.operations)
        7
        >>> stage.weights.shape
        torch.Size([7])
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> stage(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,
                 operations,
                 temperature=0.05,
                 ):
        super().__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self,
                input, y):
        if self.training:
            relaxcat = RelaxedOneHotCategorical(torch.Tensor([0.1]).to(input.device), logits=self._weights)
            wt = relaxcat.rsample()
            op_idx = wt.argmax().detach()
            op_mag = wt[op_idx] / wt[op_idx].detach()
            op_weights = torch.zeros(len(self.operations)).to(input.device)
            op_weights[op_idx] = op_mag
            return torch.stack([op_weights[i]*op(input, y) for i, op in enumerate(self.operations)]).sum(0)
        else:
            return self.operations[Categorical(logits=self._weights).sample()](input, y)

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0)



class SubPolicy(nn.Module):
    """A sequence of TaskAug stages applied one after another.

    Args:
        sub_policy_stage (SubPolicyStage): Stage that is copied
            ``operation_count`` times.
        operation_count (int): Number of stages. Defaults to 1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.aug_policy import (
        ...     SubPolicy, SubPolicyStage, all_ops)
        >>> stage = SubPolicyStage(torch.nn.ModuleList(all_ops(learn_prob=False)))
        >>> sub_policy = SubPolicy(stage, operation_count=3)
        >>> len(sub_policy.stages)
        3
    """
    def __init__(self, sub_policy_stage, operation_count=1):
        super().__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self,input,y):
        for stage in self.stages:
            input = stage(input,y)
        return input


class Policy(nn.Module):
    """TaskAug policy: applies one randomly chosen sub-policy to a batch.

    Args:
        operations (torch.nn.ModuleList): The candidate augmentation operations.
        num_sub_policies (int): Number of sub-policies. Defaults to 1.
        operation_count (int): Stages per sub-policy. Defaults to 2.
        num_chunks (int): Stored but not used by the forward pass. Defaults to 1.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.aug_policy import Policy, all_ops
        >>> operations = torch.nn.ModuleList(all_ops(learn_prob=False))
        >>> policy = Policy(operations, operation_count=2)
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> policy(x, y).shape
        torch.Size([4, 12, 2496])
    """
    def __init__(self,operations,num_sub_policies=1,operation_count=2,num_chunks=1,):
        super().__init__()
        self.sub_policies = nn.ModuleList([SubPolicy(SubPolicyStage(operations), operation_count)
                                           for _ in range(num_sub_policies)])
        self.num_sub_policies = num_sub_policies
        self.operation_count = operation_count
        self.num_chunks = num_chunks

    def forward(self,x,y):
        x = self._forward(x,y)
        return x

    def _forward(self,input,y):
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input,y)



def all_ops(learn_mag=True, learn_prob=True):
    """Builds the seven candidate operations of the TaskAug policy.

    Args:
        learn_mag (bool): Whether the operation magnitudes are trained.
        learn_prob (bool): Whether the application probabilities are trained.

    Returns:
        list: Temporal warp, baseline wander, Gaussian noise, random crop, random
        displacement, magnitude scale and no-op, in this order.

    Examples:
        >>> from pyhealth.models.taskaug.aug_policy import all_ops
        >>> ops = all_ops()
        >>> len(ops)
        7
        >>> type(ops[0]).__name__
        'RandTemporalWarp'
    """

    return [
        RandTemporalWarp(initial_magnitude=[1.0,1.0], learn_magnitude=learn_mag,learn_probability=learn_prob),
        BaselineWander(learn_magnitude=learn_mag,learn_probability=learn_prob),
        GaussianNoise(learn_magnitude=learn_mag,learn_probability=learn_prob),
        RandCrop(learn_probability=learn_prob),
        RandDisplacement(learn_magnitude=learn_mag,learn_probability=learn_prob),
        MagnitudeScale(learn_magnitude=learn_mag,learn_probability=learn_prob),
        NoOp(),
    ]



def full_policy(num_sub_policies= 1,
                    operation_count= 2,
                    num_chunks= 1,
                    learn_mag=True, learn_prob=True):
    """Builds the full TaskAug policy over all candidate operations.

    Args:
        num_sub_policies (int): Number of sub-policies. Defaults to 1.
        operation_count (int): Number of chained stages ``K`` per sub-policy.
            Defaults to 2.
        num_chunks (int): Stored but not used by the forward pass. Defaults to 1.
        learn_mag (bool): Whether the operation magnitudes are trained.
        learn_prob (bool): Whether the application probabilities are trained.

    Returns:
        Policy: The policy; call it as ``policy(x, y)`` with ECGs and class labels.

    Examples:
        >>> import torch
        >>> from pyhealth.models.taskaug.aug_policy import full_policy
        >>> policy = full_policy(operation_count=3, learn_prob=False)
        >>> len(policy.sub_policies[0].stages)
        3
        >>> x = torch.randn(4, 12, 2496)
        >>> y = torch.tensor([0, 1, 0, 1])
        >>> policy(x, y).shape
        torch.Size([4, 12, 2496])
    """
    return Policy(nn.ModuleList(all_ops(learn_mag, learn_prob)), num_sub_policies,  operation_count,
                    num_chunks)

