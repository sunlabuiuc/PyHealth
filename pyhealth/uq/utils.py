from typing import Optional, Union

import torch
from torch import Tensor


def agg_loss(loss:torch.Tensor, reduction: str):
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum': return loss.sum()
    return loss



class LogLoss(torch.nn.Module):
    reduction: str
    #Cross entropy, but takes in the probability instead of the logits
    def __init__(self, weight: Optional[Tensor] = None,  ignore_index: int = -100, reduction: str = 'mean', clip=1e-10) -> None:
        super(LogLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.clip = clip

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        dim = input.dim()
        assert dim == 2, f"Expected 2 dimensions (got {dim})"
        input = input.clip(self.clip)#this weight should be trivial, so I won't normalize
        input = -torch.log(input)
        if self.weight is not None:
            input = input * self.weight.unsqueeze(0)
        loss = torch.gather(input, -1, target.unsqueeze(-1)).squeeze(-1)
        return agg_loss(loss, self.reduction)