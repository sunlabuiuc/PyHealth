"""Conditional prompt encoder for demographic conditioning.

This module provides demographic conditioning through prompt-based learning
with reparameterization to prevent overfitting.
"""

import torch
import torch.nn as nn


class ConditionalPromptEncoder(nn.Module):
    """Encodes patient demographics into prompt vectors.

    Uses reparameterization with d_hidden=128 bottleneck before projecting
    to BART's 768-dimensional hidden space to prevent overfitting.

    Args:
        TODO: Add arguments after porting from pehr_scratch
    """

    def __init__(self, **kwargs):
        super(ConditionalPromptEncoder, self).__init__()
        # TODO: Port from ~/pehr_scratch/conditional_prompt.py
        raise NotImplementedError("ConditionalPromptEncoder porting in progress")

    def forward(self, **kwargs):
        """Encode demographics into prompt vectors."""
        raise NotImplementedError("ConditionalPromptEncoder porting in progress")
