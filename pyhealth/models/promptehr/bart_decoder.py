"""Modified BART decoder with prompt injection.

This module extends BART's decoder to inject demographic prompts at every layer.
"""

import torch
import torch.nn as nn


class PromptBartDecoder(nn.Module):
    """BART decoder with conditional prompt injection.

    Injects demographic prompts into every decoder layer to condition
    the generation process on patient demographics.

    Args:
        TODO: Add arguments after porting from pehr_scratch
    """

    def __init__(self, **kwargs):
        super(PromptBartDecoder, self).__init__()
        # TODO: Port from ~/pehr_scratch/prompt_bart_decoder.py
        raise NotImplementedError("PromptBartDecoder porting in progress")

    def forward(self, **kwargs):
        """Decode with prompt injection."""
        raise NotImplementedError("PromptBartDecoder porting in progress")
