"""Modified BART encoder with prompt injection.

This module extends BART's encoder to inject demographic prompts at every layer.
"""

import torch
import torch.nn as nn


class PromptBartEncoder(nn.Module):
    """BART encoder with conditional prompt injection.

    Injects demographic prompts into every encoder layer to condition
    the encoding process on patient demographics.

    Args:
        TODO: Add arguments after porting from pehr_scratch
    """

    def __init__(self, **kwargs):
        super(PromptBartEncoder, self).__init__()
        # TODO: Port from ~/pehr_scratch/prompt_bart_encoder.py
        raise NotImplementedError("PromptBartEncoder porting in progress")

    def forward(self, **kwargs):
        """Encode with prompt injection."""
        raise NotImplementedError("PromptBartEncoder porting in progress")
