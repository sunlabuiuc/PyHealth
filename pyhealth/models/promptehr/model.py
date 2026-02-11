"""PromptEHR: BART-based generative model for synthetic EHR generation.

This module provides the main PromptEHR model that combines demographic-conditioned
prompts with BART encoder-decoder architecture for realistic patient record generation.

Ported from pehr_scratch/prompt_bart_model.py (lines 16-276, excluding auxiliary losses).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from pyhealth.models import BaseModel
from .conditional_prompt import ConditionalPromptEncoder
from .bart_encoder import PromptBartEncoder
from .bart_decoder import PromptBartDecoder


class PromptBartModel(BartForConditionalGeneration):
    """BART model with demographic prompt conditioning for EHR generation.

    Extends HuggingFace's BartForConditionalGeneration with:
    1. Dual prompt encoders (separate for encoder/decoder)
    2. Demographic conditioning via learned prompt vectors
    3. Label smoothing for diverse generation

    This is the core generative model WITHOUT auxiliary losses (those caused
    mode collapse and are excluded per implementation decision D003).

    Args:
        config: BART configuration from transformers
        n_num_features: Number of continuous features (1 for age)
        cat_cardinalities: Category counts for categorical features ([2] for gender M/F)
        d_hidden: Intermediate reparameterization dimension (default: 128)
        prompt_length: Number of prompt vectors per feature (default: 1)

    Example:
        >>> from transformers import BartConfig
        >>> config = BartConfig.from_pretrained("facebook/bart-base")
        >>> model = PromptBartModel(
        ...     config,
        ...     n_num_features=1,           # age
        ...     cat_cardinalities=[2],      # gender (M/F)
        ...     d_hidden=128,
        ...     prompt_length=1
        ... )
        >>> # Forward pass with demographics
        >>> age = torch.randn(16, 1)        # [batch, 1]
        >>> gender = torch.randint(0, 2, (16, 1))  # [batch, 1]
        >>> input_ids = torch.randint(0, 1000, (16, 100))
        >>> labels = torch.randint(0, 1000, (16, 50))
        >>> output = model(
        ...     input_ids=input_ids,
        ...     labels=labels,
        ...     x_num=age,
        ...     x_cat=gender
        ... )
        >>> loss = output.loss
    """

    def __init__(
        self,
        config: BartConfig,
        n_num_features: Optional[int] = None,
        cat_cardinalities: Optional[list] = None,
        d_hidden: int = 128,
        prompt_length: int = 1
    ):
        """Initialize PromptBART model with dual prompt conditioning.

        Args:
            config: BART configuration
            n_num_features: Number of continuous features (e.g., 1 for age)
            cat_cardinalities: Category counts for categorical features [n_genders]
            d_hidden: Intermediate reparameterization dimension (default: 128)
            prompt_length: Number of prompt vectors per feature (default: 1)
        """
        super().__init__(config)

        # Replace encoder and decoder with prompt-aware versions
        self.model.encoder = PromptBartEncoder(config, self.model.shared)
        self.model.decoder = PromptBartDecoder(config, self.model.shared)

        # Add SEPARATE conditional prompt encoders for encoder and decoder
        # This provides stronger demographic conditioning than shared prompts (dual injection)
        if n_num_features is not None or cat_cardinalities is not None:
            # Encoder prompt encoder
            self.encoder_prompt_encoder = ConditionalPromptEncoder(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dim=config.d_model,
                d_hidden=d_hidden,
                prompt_length=prompt_length
            )
            # Decoder prompt encoder (separate parameters for dual injection)
            self.decoder_prompt_encoder = ConditionalPromptEncoder(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                hidden_dim=config.d_model,
                d_hidden=d_hidden,
                prompt_length=prompt_length
            )
            self.num_prompts = self.encoder_prompt_encoder.get_num_prompts()
        else:
            self.encoder_prompt_encoder = None
            self.decoder_prompt_encoder = None
            self.num_prompts = 0

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        x_num: Optional[torch.FloatTensor] = None,
        x_cat: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        """Forward pass with demographic conditioning.

        Args:
            input_ids: [batch, seq_len] encoder input token IDs
            attention_mask: [batch, seq_len] encoder attention mask
            decoder_input_ids: [batch, tgt_len] decoder input token IDs
            decoder_attention_mask: [batch, tgt_len] decoder attention mask
            labels: [batch, tgt_len] target labels for loss computation
            x_num: [batch, n_num_features] continuous demographic features (e.g., age)
            x_cat: [batch, n_cat_features] categorical demographic features (e.g., gender)
            Other args: Standard BART arguments

        Returns:
            Seq2SeqLMOutput with:
                - loss: Cross-entropy loss with label smoothing=0.1
                - logits: [batch, tgt_len, vocab_size] prediction logits
                - past_key_values: Cached key-value states (if use_cache=True)
                - decoder_hidden_states: Decoder layer outputs (if output_hidden_states=True)
                - encoder_last_hidden_state: Final encoder output
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode demographic prompts separately for encoder and decoder
        # Only prepend prompts on first step (when no cache exists)
        encoder_prompt_embeds = None
        decoder_prompt_embeds = None
        if (x_num is not None or x_cat is not None) and past_key_values is None:
            if self.encoder_prompt_encoder is not None:
                encoder_prompt_embeds = self.encoder_prompt_encoder(x_num=x_num, x_cat=x_cat)
            if self.decoder_prompt_encoder is not None:
                decoder_prompt_embeds = self.decoder_prompt_encoder(x_num=x_num, x_cat=x_cat)

        # Prepare decoder input IDs (shift labels right for teacher forcing)
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Encoder forward pass (with encoder prompts)
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                inputs_prompt_embeds=encoder_prompt_embeds,  # Encoder-specific prompts
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Extend encoder attention mask for prompts
        encoder_attention_mask = attention_mask
        if encoder_prompt_embeds is not None and attention_mask is not None:
            batch_size, n_prompts = encoder_prompt_embeds.shape[:2]
            prompt_mask = torch.ones(batch_size, n_prompts, dtype=attention_mask.dtype, device=attention_mask.device)
            encoder_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Decoder forward pass (with decoder prompts)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            inputs_prompt_embeds=decoder_prompt_embeds,  # Decoder-specific prompts
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Language modeling head
        lm_logits = self.lm_head(decoder_outputs[0])

        # If decoder prompts were prepended, slice them off before loss computation
        if decoder_prompt_embeds is not None and labels is not None:
            # decoder_outputs[0] shape: [batch, n_prompts + seq_len, hidden_dim]
            # We only want logits for the actual sequence positions
            n_prompts = decoder_prompt_embeds.shape[1]
            lm_logits = lm_logits[:, n_prompts:, :]  # Remove prompt positions

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Label smoothing = 0.1 to prevent overconfidence and encourage diversity
            # Softens target distributions: 90% on correct token, 10% distributed to alternatives
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        x_num=None,
        x_cat=None,
        **kwargs
    ):
        """Prepare inputs for autoregressive generation.

        Args:
            decoder_input_ids: [batch, cur_len] current decoder input IDs
            past_key_values: Cached key-value states from previous steps
            x_num: [batch, n_num_features] continuous demographics (passed through)
            x_cat: [batch, n_cat_features] categorical demographics (passed through)
            Other args: Standard BART generation arguments

        Returns:
            Dictionary of inputs for next generation step
        """
        # Cut decoder_input_ids if past is used (only need last token)
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "x_num": x_num,  # Pass demographics through generation
            "x_cat": x_cat,
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids,
        expand_size=1,
        is_encoder_decoder=True,
        attention_mask=None,
        encoder_outputs=None,
        x_num=None,
        x_cat=None,
        **model_kwargs,
    ):
        """Expand inputs for beam search or multiple samples.

        Args:
            input_ids: [batch, seq_len] input token IDs
            expand_size: Number of beams/samples per input
            x_num: [batch, n_num_features] continuous demographics
            x_cat: [batch, n_cat_features] categorical demographics
            Other args: Standard expansion arguments

        Returns:
            Expanded input_ids and model_kwargs
        """
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if encoder_outputs is not None:
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        # Expand demographics for beam search
        if x_num is not None:
            model_kwargs["x_num"] = x_num.index_select(0, expanded_return_idx)

        if x_cat is not None:
            model_kwargs["x_cat"] = x_cat.index_select(0, expanded_return_idx)

        return input_ids, model_kwargs


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """Shift input ids one token to the right for teacher forcing.

    Args:
        input_ids: [batch, seq_len] target token IDs
        pad_token_id: ID for padding token
        decoder_start_token_id: ID for decoder start token (BOS)

    Returns:
        [batch, seq_len] shifted token IDs with BOS prepended
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("config.pad_token_id must be defined for sequence generation")

    # Replace -100 in labels with pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class PromptEHR(BaseModel):
    """PromptEHR: PyHealth wrapper for prompt-based BART EHR generation.

    This class extends PyHealth's BaseModel to integrate PromptBartModel into
    the PyHealth ecosystem while maintaining compatibility with PyHealth's
    Trainer and evaluation infrastructure.

    Args:
        dataset: PyHealth dataset (required by BaseModel, can be None for generative)
        n_num_features: Number of continuous features (1 for age)
        cat_cardinalities: Category counts for categorical features ([2] for gender)
        d_hidden: Intermediate reparameterization dimension (default: 128)
        prompt_length: Number of prompt vectors per feature (default: 1)
        bart_config_name: Pretrained BART model name (default: "facebook/bart-base")
        **kwargs: Additional BaseModel arguments

    Example:
        >>> from pyhealth.datasets import PromptEHRDataset
        >>> dataset = PromptEHRDataset(...)
        >>> model = PromptEHR(
        ...     dataset=dataset,
        ...     n_num_features=1,
        ...     cat_cardinalities=[2],
        ...     d_hidden=128
        ... )
        >>> # Training
        >>> output = model(input_ids=..., labels=..., x_num=..., x_cat=...)
        >>> loss = output["loss"]
        >>> # Generation
        >>> generated = model.generate(input_ids=..., x_num=..., x_cat=...)
    """

    def __init__(
        self,
        dataset=None,
        n_num_features: int = 1,
        cat_cardinalities: Optional[list] = None,
        d_hidden: int = 128,
        prompt_length: int = 1,
        bart_config_name: str = "facebook/bart-base",
        **kwargs
    ):
        """Initialize PromptEHR model with PyHealth BaseModel integration.

        Args:
            dataset: PyHealth dataset (can be None for generative models)
            n_num_features: Number of continuous features (default: 1 for age)
            cat_cardinalities: Category counts (default: [2] for gender M/F)
            d_hidden: Reparameterization dimension (default: 128)
            prompt_length: Prompt vectors per feature (default: 1)
            bart_config_name: Pretrained BART model (default: "facebook/bart-base")
            **kwargs: Additional BaseModel arguments (including _custom_vocab_size for checkpoint loading)
        """
        # Extract custom vocab size if provided (used by load_from_checkpoint)
        custom_vocab_size = kwargs.pop('_custom_vocab_size', None)

        super().__init__(dataset=dataset, **kwargs)

        # Set mode to None to skip discriminative evaluation (generative model)
        self.mode = None

        # Default categorical cardinalities if not provided
        if cat_cardinalities is None:
            cat_cardinalities = [2]  # Gender (M/F)

        # Initialize BART config from pretrained
        bart_config = BartConfig.from_pretrained(bart_config_name)

        # Override vocab_size if loading from custom checkpoint
        if custom_vocab_size is not None:
            bart_config.vocab_size = custom_vocab_size

        # Apply dropout configuration (increased from BART default 0.1 to 0.3)
        bart_config.dropout = 0.3
        bart_config.attention_dropout = 0.3
        bart_config.activation_dropout = 0.3

        # Initialize PromptBartModel
        self.bart_model = PromptBartModel(
            config=bart_config,
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_hidden=d_hidden,
            prompt_length=prompt_length
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            **kwargs: Arguments passed to PromptBartModel.forward()
                Required: input_ids, labels, x_num, x_cat
                Optional: attention_mask, decoder_attention_mask, etc.

        Returns:
            Dictionary with:
                - loss: Cross-entropy loss with label smoothing
                - logits: Prediction logits (optional)
        """
        output = self.bart_model(**kwargs)

        # Return PyHealth-compatible dict (minimum: {"loss": ...})
        result = {
            "loss": output.loss,
        }

        # Add optional fields if available
        if hasattr(output, "logits"):
            result["logits"] = output.logits

        return result

    def generate(self, **kwargs):
        """Generate synthetic patient sequences.

        Args:
            **kwargs: Arguments passed to PromptBartModel.generate()
                Required: input_ids (demographics encoded), x_num, x_cat
                Optional: max_length, num_beams, temperature, etc.

        Returns:
            Generated token IDs [batch, seq_len]
        """
        return self.bart_model.generate(**kwargs)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, dataset=None, **model_kwargs):
        """Load PromptEHR model from pehr_scratch checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (e.g., best_model.pt)
            dataset: PyHealth dataset (optional, can be None for generative models)
            **model_kwargs: Model initialization arguments (n_num_features, cat_cardinalities, etc.)

        Returns:
            Loaded PromptEHR model with checkpoint weights

        Example:
            >>> model = PromptEHR.load_from_checkpoint(
            ...     "/scratch/jalenj4/promptehr_checkpoints/best_model.pt",
            ...     n_num_features=1,
            ...     cat_cardinalities=[2]
            ... )
        """
        import torch

        # Load checkpoint (weights_only=False needed for custom tokenizer class)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract model state dict (pehr_scratch format has extra keys)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            epoch = checkpoint.get('epoch', None)
            val_loss = checkpoint.get('val_loss', None)
        else:
            # Direct state dict
            state_dict = checkpoint
            epoch = None
            val_loss = None

        # Auto-detect vocab_size from checkpoint
        # pehr_scratch uses custom vocabulary (6992 tokens) vs BART default (50265)
        if 'model.shared.weight' in state_dict:
            checkpoint_vocab_size = state_dict['model.shared.weight'].shape[0]

            # Override bart_config_name if vocab size differs from default
            if 'bart_config_name' not in model_kwargs:
                # Load default config to check vocab size
                from transformers import BartConfig
                default_config = BartConfig.from_pretrained("facebook/bart-base")

                if checkpoint_vocab_size != default_config.vocab_size:
                    # Create custom config with detected vocab size
                    print(f"Detected custom vocab_size={checkpoint_vocab_size} in checkpoint "
                          f"(BART default: {default_config.vocab_size})")

                    # Store custom config by temporarily modifying the config
                    model_kwargs['_custom_vocab_size'] = checkpoint_vocab_size

        # Create model instance
        model = cls(dataset=dataset, **model_kwargs)

        # Load weights
        model.bart_model.load_state_dict(state_dict, strict=True)

        # Print checkpoint info
        if epoch is not None:
            print(f"Loaded checkpoint from epoch {epoch}, val_loss={val_loss:.4f}")

        return model
