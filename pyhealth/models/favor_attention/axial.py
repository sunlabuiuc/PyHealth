# Adapted from https://github.com/lucidrains/axial-positional-embedding/blob/c1fc8cbe20424cbdba7960c13451481097baa20a/axial_positional_embedding/axial_positional_embedding.py

from operator import mul
from functools import reduce

import torch
from torch import nn
from torch.nn import Module

from einops import rearrange, pack, unpack

# helper functions

def exists(v):
    return v is not None

# main class

class AxialPositionalEmbedding(Module):
    def __init__(
        self,
        dim,
        axial_shape: tuple[int, ...],
        axial_dims: tuple[int, ...] | None = None
    ):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = not exists(axial_dims)
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        assert self.summed or not self.summed and sum(axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        self.weights = nn.ParameterList([])

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x):

        batch, seq_len, _ = x.shape
        assert (seq_len <= self.max_seq_len), f'Sequence length ({seq_len}) must be less than the maximum sequence length allowed ({self.max_seq_len})'

        embs = []

        for ax_emb in self.weights:
            axial_dim = ax_emb.shape[-1]
            expand_shape = (batch, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(batch, self.max_seq_len, axial_dim)
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        return pos_emb[:, :seq_len].to(x)