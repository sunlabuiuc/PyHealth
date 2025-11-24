"""BART encoder with prompt injection for demographic conditioning.

This module provides a modified BART encoder that accepts demographic prompt
embeddings and prepends them to input sequences for conditioning.

Ported from pehr_scratch/prompt_bart_encoder.py (lines 1-149).
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers.models.bart.modeling_bart import BartEncoder
from transformers.modeling_outputs import BaseModelOutput


class PromptBartEncoder(BartEncoder):
    """BART encoder modified to accept and prepend demographic prompt embeddings.

    Extends the standard BART encoder to support prompt-based conditioning by:
    1. Accepting optional prompt embeddings as input
    2. Prepending prompts to input token embeddings
    3. Extending attention masks to cover prepended prompts
    4. Processing through standard BART encoder layers

    This enables demographic conditioning (age + gender) by injecting learned
    prompt vectors at the encoder input.

    Args:
        config: BartConfig from transformers
        embed_tokens: Token embedding layer (optional)

    Example:
        >>> from transformers import BartConfig
        >>> config = BartConfig.from_pretrained("facebook/bart-base")
        >>> encoder = PromptBartEncoder(config)
        >>> # Encode with prompts
        >>> prompt_embeds = torch.randn(16, 2, 768)  # [batch, n_prompts, hidden]
        >>> input_ids = torch.randint(0, 1000, (16, 100))  # [batch, seq_len]
        >>> outputs = encoder(input_ids, inputs_prompt_embeds=prompt_embeds)
    """

    def __init__(self, config, embed_tokens=None):
        """Initialize prompt-aware BART encoder.

        Args:
            config: BartConfig from transformers
            embed_tokens: Optional token embedding layer
        """
        super().__init__(config, embed_tokens)

        # Initialize embedding scale factor (BART uses sqrt(d_model) scaling)
        self.embed_scale = None
        if config.scale_embedding:
            self.embed_scale = (config.d_model ** 0.5)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput:
        """Forward pass with optional demographic prompt embeddings.

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask (1=attend, 0=ignore)
            head_mask: [num_layers, num_heads] mask for attention heads
            inputs_embeds: [batch, seq_len, hidden_dim] pre-computed embeddings (optional)
            inputs_prompt_embeds: [batch, n_prompts, hidden_dim] demographic prompts (optional)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return BaseModelOutput or tuple

        Returns:
            BaseModelOutput with:
                - last_hidden_state: [batch, n_prompts + seq_len, hidden_dim]
                - hidden_states: Tuple of all layer outputs (if output_hidden_states=True)
                - attentions: Tuple of attention weights (if output_attentions=True)
        """
        # Set output flags from config defaults
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get input embeddings from token IDs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Apply embedding scaling if configured
            if self.embed_scale is not None:
                inputs_embeds = inputs_embeds * self.embed_scale

        # Prepend prompt embeddings if provided
        if inputs_prompt_embeds is not None:
            # Concatenate prompts before input embeddings
            # inputs_prompt_embeds: [batch, n_prompts, hidden_dim]
            # inputs_embeds: [batch, seq_len, hidden_dim]
            # Result: [batch, n_prompts + seq_len, hidden_dim]
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)

            # Extend attention mask to account for prepended prompts
            batch_size, n_prompts = inputs_prompt_embeds.shape[:2]

            if attention_mask is not None:
                # Create attention mask for prompts matching existing mask dtype/device
                prompt_attention_mask = torch.ones(
                    batch_size, n_prompts,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                # Concatenate prompt mask with original mask
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                # Create full attention mask for prompts + sequence
                seq_len = inputs_embeds.shape[1]  # Total length including prompts already prepended
                attention_mask = torch.ones(
                    batch_size, seq_len,
                    dtype=torch.long,
                    device=inputs_embeds.device
                )

        # Get positional embeddings (BART uses learned positional embeddings)
        embed_pos = self.embed_positions(inputs_embeds)

        # Combine input embeddings + positional embeddings
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Expand attention mask from [batch, seq_len] to [batch, 1, tgt_len, src_len]
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        # Initialize output containers
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Validate head_mask dimensionality
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"head_mask should have {len(self.layers)} layers, but has {head_mask.size()[0]}"
                )

        # Pass through encoder layers
        for idx, encoder_layer in enumerate(self.layers):
            # Save hidden state before layer if requested
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # Get layer-specific head mask
            layer_head_mask = head_mask[idx] if head_mask is not None else None

            # Forward through encoder layer
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            # Update hidden states
            hidden_states = layer_outputs[0]

            # Save attention weights if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Save final hidden state if requested
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # Return tuple format if not using return_dict
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        # Return BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """Expand attention mask from [batch, src_len] to [batch, 1, tgt_len, src_len].

    Inverts the mask (1→0, 0→1) and fills masked positions with -inf to prevent attention.

    Args:
        mask: [batch, src_len] attention mask (1=attend, 0=ignore)
        dtype: Target data type for the expanded mask
        tgt_len: Target sequence length (defaults to src_len for encoder self-attention)

    Returns:
        [batch, 1, tgt_len, src_len] expanded mask with -inf for masked positions
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    # Expand dimensions: [batch, src_len] → [batch, 1, tgt_len, src_len]
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)

    # Invert mask: 1 (attend) → 0, 0 (ignore) → 1
    inverted_mask = 1.0 - expanded_mask

    # Fill masked positions with -inf (prevents attention)
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
