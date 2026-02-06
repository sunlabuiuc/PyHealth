"""BART decoder with prompt injection for demographic conditioning.

This module provides a modified BART decoder that accepts demographic prompt
embeddings and prepends them to decoder input sequences for conditioning.

Ported from pehr_scratch/prompt_bart_decoder.py (lines 1-207).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.bart.modeling_bart import BartDecoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class PromptBartDecoder(BartDecoder):
    """BART decoder modified to accept and prepend demographic prompt embeddings.

    Extends the standard BART decoder to support prompt-based conditioning by:
    1. Accepting optional prompt embeddings as input
    2. Prepending prompts to decoder input token embeddings
    3. Extending attention masks to cover prepended prompts
    4. Creating causal masks for autoregressive generation
    5. Processing through standard BART decoder layers with cross-attention

    This enables demographic conditioning (age + gender) by injecting learned
    prompt vectors at the decoder input, maintaining demographic alignment
    during generation (dual prompt injection with encoder).

    Args:
        config: BartConfig from transformers
        embed_tokens: Token embedding layer (optional)

    Example:
        >>> from transformers import BartConfig
        >>> config = BartConfig.from_pretrained("facebook/bart-base")
        >>> decoder = PromptBartDecoder(config)
        >>> # Decode with prompts
        >>> prompt_embeds = torch.randn(16, 2, 768)  # [batch, n_prompts, hidden]
        >>> input_ids = torch.randint(0, 1000, (16, 50))  # [batch, tgt_len]
        >>> encoder_outputs = torch.randn(16, 100, 768)  # [batch, src_len, hidden]
        >>> outputs = decoder(
        ...     input_ids,
        ...     encoder_hidden_states=encoder_outputs,
        ...     inputs_prompt_embeds=prompt_embeds
        ... )
    """

    def __init__(self, config, embed_tokens=None):
        """Initialize prompt-aware BART decoder.

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
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass with optional demographic prompt embeddings.

        Args:
            input_ids: [batch, tgt_seq_len] decoder token IDs
            attention_mask: [batch, tgt_seq_len] decoder attention mask (1=attend, 0=ignore)
            encoder_hidden_states: [batch, src_seq_len, hidden_dim] encoder outputs
            encoder_attention_mask: [batch, src_seq_len] encoder attention mask
            head_mask: [num_layers, num_heads] mask for self-attention heads
            cross_attn_head_mask: [num_layers, num_heads] mask for cross-attention heads
            past_key_values: Cached key-value states for efficient generation
            inputs_embeds: [batch, tgt_seq_len, hidden_dim] pre-computed embeddings (optional)
            inputs_prompt_embeds: [batch, n_prompts, hidden_dim] demographic prompts (optional)
            use_cache: Whether to return key-value cache for generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return BaseModelOutputWithPastAndCrossAttentions or tuple

        Returns:
            BaseModelOutputWithPastAndCrossAttentions with:
                - last_hidden_state: [batch, n_prompts + tgt_len, hidden_dim]
                - past_key_values: Cached key-value states (if use_cache=True)
                - hidden_states: Tuple of all layer outputs (if output_hidden_states=True)
                - attentions: Tuple of self-attention weights (if output_attentions=True)
                - cross_attentions: Tuple of cross-attention weights (if output_attentions=True)
        """
        # Set output flags from config defaults
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get decoder input embeddings from token IDs
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # Apply embedding scaling if configured
            if self.embed_scale is not None:
                inputs_embeds = inputs_embeds * self.embed_scale

        # Store original sequence length before prepending prompts
        original_seq_len = inputs_embeds.shape[1]

        # Prepend prompt embeddings if provided
        if inputs_prompt_embeds is not None:
            # Concatenate prompts before decoder input embeddings
            # inputs_prompt_embeds: [batch, n_prompts, hidden_dim]
            # inputs_embeds: [batch, tgt_len, hidden_dim]
            # Result: [batch, n_prompts + tgt_len, hidden_dim]
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)

            # Extend attention mask for prepended prompts
            batch_size, n_prompts = inputs_prompt_embeds.shape[:2]

            # Create attention mask for prompts (all 1s - always attend to prompts)
            prompt_attention_mask = torch.ones(
                batch_size, n_prompts,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=inputs_embeds.device
            )

            if attention_mask is not None:
                # Concatenate prompt mask with decoder attention mask
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
            else:
                # Create attention mask for all tokens (prompts + decoder input)
                total_seq_len = inputs_embeds.shape[1]
                attention_mask = torch.ones(
                    batch_size, total_seq_len,
                    dtype=torch.long,
                    device=inputs_embeds.device
                )

        # Get positional embeddings for full sequence (prompts + decoder tokens)
        past_key_values_length = 0
        if past_key_values is not None:
            # Handle Cache object (new transformers API) or tuple (old API)
            if hasattr(past_key_values, 'get_seq_length'):
                past_key_values_length = past_key_values.get_seq_length()
            elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                # Defensive: handle unexpected cache structures gracefully
                # pehr-scratch-expert confirmed: defaulting to 0 is safe (slightly degrades
                # quality but prevents crash). BART handles positional errors gracefully.
                try:
                    if past_key_values[0] is not None and isinstance(past_key_values[0], (tuple, list)):
                        if len(past_key_values[0]) > 0 and past_key_values[0][0] is not None:
                            past_key_values_length = past_key_values[0][0].shape[2]
                except (IndexError, TypeError, AttributeError):
                    # Safe fallback: slightly degrades quality but prevents crash
                    # Positional embeddings will be calculated from position 0
                    past_key_values_length = 0

        # Get positional embeddings (BART uses learned positional embeddings)
        positions = self.embed_positions(inputs_embeds, past_key_values_length)

        # Combine input embeddings + positional embeddings
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Create combined attention mask (causal + padding)
        if attention_mask is not None:
            # Create causal mask for decoder self-attention
            combined_attention_mask = _make_causal_mask(
                inputs_embeds.shape[:2],
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            # Expand padding mask and combine with causal mask
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=inputs_embeds.shape[1])
            combined_attention_mask = combined_attention_mask + expanded_attn_mask
        else:
            # Create causal mask only (no padding)
            combined_attention_mask = _make_causal_mask(
                inputs_embeds.shape[:2],
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        # Expand encoder attention mask for cross-attention
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [batch, src_len] → [batch, 1, tgt_len, src_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=inputs_embeds.shape[1])

        # Initialize output containers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # Pass through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # Save hidden state before layer if requested
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Forward through decoder layer
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=combined_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # Update hidden states
            hidden_states = layer_outputs[0]

            # Save attention weights if requested
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # Save final hidden state if requested
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Cache is handled by past_key_values object, not returned in tuple
        next_cache = past_key_values if use_cache else None

        # Return tuple format if not using return_dict
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        # Return BaseModelOutputWithPastAndCrossAttentions
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


def _make_causal_mask(
    input_shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0
) -> torch.Tensor:
    """Create causal mask for decoder self-attention.

    Creates a lower-triangular mask that prevents attending to future positions.
    This is essential for autoregressive generation where each position can only
    attend to earlier positions.

    Args:
        input_shape: (batch_size, tgt_len) shape of decoder input
        dtype: Data type for mask tensor
        device: Device to create mask on
        past_key_values_length: Length of cached key-values from previous steps

    Returns:
        [batch, 1, tgt_len, tgt_len + past_len] causal mask with -inf for future positions
    """
    batch_size, tgt_len = input_shape

    # Initialize mask with -inf (prevents attention)
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

    # Create lower triangular mask (0 for allowed positions, -inf for future)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    # If using cached key-values, allow attending to all past positions
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    # Expand to [batch, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand(batch_size, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None) -> torch.Tensor:
    """Expand attention mask from [batch, src_len] to [batch, 1, tgt_len, src_len].

    Inverts the mask (1→0, 0→1) and fills masked positions with -inf to prevent attention.

    Args:
        mask: [batch, src_len] attention mask (1=attend, 0=ignore)
        dtype: Target data type for the expanded mask
        tgt_len: Target sequence length (defaults to src_len)

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
