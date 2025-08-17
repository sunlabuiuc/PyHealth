'''
User interface to use promptEHR models.
Adapted from original PromptEHR implementation for PyHealth integration.
Preserves original BART-based architecture with conditional prompts.
'''
import os
import pdb
import json
import math
import glob
import random
import copy
import time
from collections import defaultdict
import warnings

import pickle
import torch
from torch import nn, Tensor
from torch.utils.data.dataloader import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Added from original PromptEHR for training support
from transformers import TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import InputDataClass
from transformers.trainer_pt_utils import (
    nested_detach, nested_concat, nested_truncate, nested_numpify, find_batch_size
)
from transformers.trainer_utils import has_length, denumpify_detensorize, EvalLoopOutput, EvalPrediction
from transformers.trainer_pt_utils import IterableDatasetShard

from transformers import BartTokenizer, BartConfig
from transformers.generation.utils import GenerationMixin
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, BartEncoder, BartDecoder
from transformers.models.bart.modeling_bart import shift_tokens_right, BartLearnedPositionalEmbedding as TransformersBartLearnedPositionalEmbedding
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.file_utils import ModelOutput
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordLevel


# Constants from original PromptEHR implementation
CODE_TYPES = ['tbd']
SPECIAL_TOKEN_DICT = {'tbd':['<tbd>','</tbd>']}
UNKNOWN_TOKEN = '<unk>'
MODEL_MAX_LENGTH = 512
EPS = 1e-16

# Additional constants from original constants.py
PRETRAINED_MODEL_URL = 'https://storage.googleapis.com/pytrial/promptEHR_pretrained.zip'
SYNTHETIC_DATA_URL = 'https://github.com/RyanWangZf/PromptEHR/raw/main/demo_data/synthetic_ehr/data.pkl'

# a name mapping from the original promptehr config to the training_args
config_to_train_args = {
    'epoch': 'num_train_epochs',
    'num_worker': 'dataloader_num_workers',
    'batch_size': 'per_device_train_batch_size',
    'eval_batch_size': 'per_device_eval_batch_size',
    'eval_step': 'eval_steps',
}

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def _all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)

class NumericalConditionalPrompt(nn.Module):
    '''Embedding for conditional prompts based on numerical input patient features,
    take reparametrization trick.

    Parameters
    ----------
    n_feature: number of input features.
    d_model: dimension of output embeddings.
    d_hidden: dimension of intermediate embeddings for reparametrization.
    '''
    def __init__(self, n_feature, d_model, d_hidden) -> None:
        super().__init__()
        self.weight = nn.init.xavier_uniform_(nn.Parameter(Tensor(n_feature, d_hidden)))
        self.bias = nn.init.xavier_uniform_(nn.Parameter(Tensor(n_feature, d_hidden)))
        self.proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        # Ensure weight and bias are on the same device as input
        device = x.device
        weight = self.weight.to(device)
        bias = self.bias.to(device)
        
        x = weight[None] * x[..., None]
        x = x + bias[None]
        
        # Ensure projection layer is on the same device as input
        self.proj = self.proj.to(device)
        x = self.proj(x)
        return x

class CategoricalConditionalPrompt(nn.Module):
    '''Embedding for conditional prompts based on categorical input patient features,
    take reparametrization trick.

    Parameters
    ----------
    cardinalities: the number of distinct values for each feature, e.g., [2, 3, 5] indicates the first cat has 2 possible categories and so on.
    d_model: the output embedding dimension.
    d_hidden: the intermediate layer dimension for reparameterization.
    '''
    def __init__(self,
        cardinalities,
        d_model,
        d_hidden
        ) -> None:
        super().__init__()
        assert cardinalities, 'cardinalities must be non-empty'
        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets, persistent=False)
        self.embeddings = nn.Embedding(sum(cardinalities), d_hidden)
        self.bias = nn.init.xavier_uniform_(nn.Parameter(Tensor(len(cardinalities),d_hidden)))
        self.proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        # Ensure category_offsets and bias are on the same device as input
        device = x.device
        category_offsets = self.category_offsets.to(device)
        bias = self.bias.to(device)
        
        # Ensure embeddings and projection layer are on the same device as input
        self.embeddings = self.embeddings.to(device)
        self.proj = self.proj.to(device)
        
        x = self.embeddings(x + category_offsets[None])
        x = x + bias[None]
        x = self.proj(x)
        return x

class ConditionalPrompt(nn.Module):
    '''Provide conditional prompt embedding for both categorical and numerical features.

    Parameters
    ----------
    n_num_feature: number of input numerical features.
    cat_cardinalities: a list of unique numbers of each feature.
    d_model: the output dimension.
    d_hidden: the intermediate layer dimension for reparametrization.
    '''
    def __init__(self,
        n_num_feature=None,
        cat_cardinalities=None,
        d_model=None,
        d_hidden=None,
        ) -> None:
        super().__init__()
        if n_num_feature is not None: 
            assert isinstance(n_num_feature, int), 'the passed `n_num_feature` to `promptehr` must be an integer, {} with type {} found.'.format(n_num_feature, type(n_num_feature))
            assert n_num_feature >= 0, 'n_num_feature must be non-negative'
        assert (n_num_feature or cat_cardinalities), 'at least one of n_num_feature or cat_cardinalities must be positive/non-empty'
        self.num_tokenizer = (
            NumericalConditionalPrompt(
                n_feature=n_num_feature,
                d_model=d_model,
                d_hidden=d_hidden,
            )
            if n_num_feature
            else None
        )
        self.cat_tokenizer = (
            CategoricalConditionalPrompt(
                cat_cardinalities,
                d_model=d_model,
                d_hidden=d_hidden,
            )
            if cat_cardinalities
            else None
        )

    def forward(self, x_num=None, x_cat=None):
        '''Perform the forward pass to encode features into prompt context vectors.

        Parameters
        ----------
        x_num: continuous features. Must be presented if :code:`n_num_feature > 0` was passed.
        x_cat: categorical features. Must be presented if non-empty :code:`cat_cardinalities` was passed.
        '''
        assert (
            x_num is not None or x_cat is not None
        ), 'At least one of x_num and x_cat must be presented'
        assert _all_or_none(
            [self.num_tokenizer, x_num]
        ), 'If self.num_tokenizer is (not) None, then x_num must (not) be None'
        assert _all_or_none(
            [self.cat_tokenizer, x_cat]
        ), 'If self.cat_tokenizer is (not) None, then x_cat must (not) be None'
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        # Handle the case where seq_len might be a tensor or torch.Size element
        if torch.is_tensor(seq_len):
            if seq_len.numel() == 1:
                seq_len = seq_len.item()
            else:
                # If it's a multi-element tensor, take the first element or max
                seq_len = seq_len.max().item() if seq_len.numel() > 0 else 1
        else:
            seq_len = int(seq_len)
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        positions = positions + self.offset
        positions = torch.minimum(positions, torch.ones_like(positions).to(positions.device)*1024)
        res = super().forward(positions)
        return res


class PromptBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        embed_dim = config.d_model
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # Add missing embed_scale (standard BART uses sqrt of d_model if scale_embedding is True)
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor]=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        '''Make encoding.
        Parameters
        ----------
        inputs_prompt_embeds: Embeded conditional prompt embeddings.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if inputs_prompt_embeds is not None:
            # concatenate prompt embeddings in front of the input embeds
            # modify input_shape and attention_mask at the same time
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            input_shape = inputs_embeds.size()[:-1]
            if attention_mask is not None:
                add_att_mask = torch.ones(inputs_prompt_embeds.shape[:-1]).to(attention_mask.device)
                attention_mask = torch.cat([add_att_mask, attention_mask], dim=1)

        embed_pos = self.embed_positions(input_shape)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class PromptBartDecoder(BartDecoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # Add missing embed_scale (standard BART uses sqrt of d_model if scale_embedding is True)
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_prompt_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
        ):
        '''Make forward pass by the decoder.

        Parameters
        ----------
        inputs_prompt_embeds: the embeddings of conditional prompts for the decoder.

        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = int(past_key_values[0][0].shape[2]) if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if inputs_prompt_embeds is not None:
            # concatenate prompt embeddings in front of the input embeds
            # modify input_shape and attention_mask at the same time
            inputs_embeds = torch.cat([inputs_prompt_embeds, inputs_embeds], dim=1)
            input_shape = inputs_embeds.size()[:-1]
            if attention_mask is not None:
                add_att_mask = torch.ones(inputs_prompt_embeds.shape[:-1]).to(attention_mask.device)
                attention_mask = torch.cat([add_att_mask, attention_mask], dim=1)

        # Handle different transformers versions - method was renamed
        if hasattr(self, '_prepare_decoder_attention_mask'):
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )
        elif hasattr(self, 'create_extended_attention_mask_for_decoder'):
            if attention_mask is not None:
                attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, past_key_values_length
                )
        else:
            # Fallback for newer transformers versions
            pass

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if inputs_prompt_embeds is not None:
                # adjust for input prompt embeddings
                add_att_mask = torch.ones(inputs_prompt_embeds.shape[:-1]).to(encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([add_att_mask, encoder_attention_mask], dim=1)

            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        dummy_input_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
        
        # Debug and bounds check
        seq_len = input_shape[-1]
        max_pos_embeddings = self.embed_positions.num_embeddings
        max_allowed_pos = max_pos_embeddings - 10  # Conservative buffer
        
        # Calculate safe sequence length considering past key values
        max_safe_seq_len = max(1, max_allowed_pos - past_key_values_length)
        
        # Ensure we don't truncate to less than 1
        if seq_len > max_safe_seq_len and max_safe_seq_len > 0:
            truncated_seq_len = max_safe_seq_len
            truncated_shape = input_shape[:-1] + (truncated_seq_len,)
            dummy_input_ids = torch.zeros(truncated_shape, dtype=torch.long, device=inputs_embeds.device)
            inputs_embeds = inputs_embeds[:, :truncated_seq_len, :]
            # Update input_shape for attention mask compatibility
            input_shape = truncated_shape
        else:
            # Don't truncate if it would make sequence too short
            truncated_seq_len = seq_len
            dummy_input_ids = torch.zeros(input_shape, dtype=torch.long, device=inputs_embeds.device)
            
        positions = self.embed_positions(dummy_input_ids.shape, past_key_values_length)

        # Ensure both tensors are on the same device before addition
        if positions.device != inputs_embeds.device:
            try:
                positions = positions.to(inputs_embeds.device)
            except RuntimeError:
                # If device transfer fails, move inputs_embeds to positions device
                inputs_embeds = inputs_embeds.to(positions.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    print(
                        "[warning] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else: # testing/generating                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class PromptBartModel(BartModel):
    '''a subclass of BartModel by using additional prompts for controllable EHR generation.
    '''
    def __init__(self, config: BartConfig):
        super().__init__(config)
        # Store config for later access
        self.config = config

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PromptBartEncoder(config, self.shared)
        self.decoder = PromptBartDecoder(config, self.shared)

        # build encoder & decoder prompts
        n_num_feature = config.n_num_feature
        cat_cardinalities = config.cat_cardinalities
        if n_num_feature is not None or cat_cardinalities is not None:
            self.encoder_conditional_prompt = ConditionalPrompt(n_num_feature=n_num_feature,
                cat_cardinalities=cat_cardinalities,
                d_model=config.d_model,
                d_hidden=config.d_prompt_hidden)
            self.decoder_conditional_prompt = ConditionalPrompt(n_num_feature=n_num_feature,
                cat_cardinalities=cat_cardinalities,
                d_model=config.d_model,
                d_hidden=config.d_prompt_hidden)
        else:
            # fix when no baseline feature is provided.
            warnings.warn('No numerical or categorical baseline features are provided, `ConditionalPrompt` is not used in the model.')
            self.encoder_conditional_prompt = None
            self.decoder_conditional_prompt = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        x_num: Optional[torch.FloatTensor] = None,
        x_cat: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        '''Make the forward pass to encode inputs with Bart model.

        Parameters
        ----------
        x_num: the input numerical features, shape (bs, num_feat)
        x_cat: the input categorical features, shape (bs, num_cat)
        '''
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if x_num is not None or x_cat is not None:
                if self.encoder_conditional_prompt is None:
                    warnings.warn('Detect input baseline features in the data,` \
                        but `ConditionalPrompt was not built because no numerical or categorical baseline features are provided when model was initialized. \
                        Consider setting `config.n_num_feature` or `config.cat_cardinalities` when initializing the model.')
                    prompt_embeds = None
                else:
                    prompt_embeds = self.encoder_conditional_prompt(x_num=x_num, x_cat=x_cat)
            else:
                prompt_embeds = None
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                inputs_prompt_embeds=prompt_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if x_num is not None or x_cat is not None:
            if self.decoder_conditional_prompt is None:
                warnings.warn('{} {} {}'.format('Detect input baseline features in the data, but `ConditionalPrompt`',
                    'was not built because no numerical or categorical baseline features',
                    'Consider setting `config.n_num_feature` or `config.cat_cardinalities` when initializing the model.')
                )
                decoder_prompt_embeds = None 
            else:
                decoder_prompt_embeds = self.decoder_conditional_prompt(x_num=x_num, x_cat=x_cat)
        else:
            decoder_prompt_embeds = None
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            inputs_prompt_embeds=decoder_prompt_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

def EHRBartConfig(data_tokenizer, model_tokenizer, **kwargs):
    '''Build the config used for building the promptBart model.
    '''
    bart_config = BartConfig.from_pretrained('facebook/bart-base')
    # Store the num_tokens dict separately so we can access it later
    num_tokens_dict = model_tokenizer.get_num_tokens
    kwargs.update(num_tokens_dict)
    bart_config.__dict__['_num_tokens_dict'] = num_tokens_dict  # Store it with a special key
    kwargs['data_tokenizer_num_vocab'] = len(data_tokenizer)
    
    # CRITICAL FIX: Update vocab_size to match the extended tokenizer vocabulary
    # The data_tokenizer has been extended with medical codes, so we need to update
    # the BART config to match this larger vocabulary size
    original_vocab_size = bart_config.vocab_size
    extended_vocab_size = len(data_tokenizer)
    bart_config.vocab_size = extended_vocab_size
    print(f"Updated BART config vocab_size from {original_vocab_size} to {extended_vocab_size}")
    
    if 'd_prompt_hidden' not in kwargs:
        kwargs['d_prompt_hidden'] = 128
    if 'n_num_feature' not in kwargs:
        kwargs['n_num_feature'] = None
    if 'cat_cardinalities' not in kwargs:
        kwargs['cat_cardinalities'] = None
    bart_config.__dict__.update(kwargs)

    # specify bos, eos token id
    bart_config.__dict__['decoder_start_token_id'] = 0
    bart_config.__dict__['bos_token_id'] = 0
    bart_config.__dict__['eos_token_id'] = 1
    bart_config.__dict__['forced_eos_token_id'] = 1
    return bart_config

class DataTokenizer(BartTokenizer):
    r'''construct tokenizer to process the input raw records.
    '''
    new_token_type_list = CODE_TYPES
    special_token_dict = SPECIAL_TOKEN_DICT
    code_vocab = defaultdict(list)

    def add_token_to_code_vocab(self, tokens, code):
        # Only add tokens that aren't already in the tokenizer vocabulary
        new_tokens = [token for token in tokens if token not in self.get_vocab()]
        if new_tokens:
            self.add_tokens(new_tokens)

        if code not in self.code_vocab:
            self.code_vocab[code] = np.array(tokens)
        else:
            origin_tokens = self.code_vocab[code]
            new_tokens = np.array(tokens)
            self.code_vocab[code] = np.unique(np.concatenate([origin_tokens, new_tokens]))

    def update_special_token_config(self, code_types):
        self.new_token_type_list = code_types
        self.special_token_dict = {}
        special_token_list = []
        for code_type in code_types:
            l = [f'<{code_type}>', f'</{code_type}>']
            self.special_token_dict[code_type] = l
            special_token_list.extend(l)
        self.add_tokens(special_token_list)

    def extend_vocab(self, token_dict):
        '''
        Parameters:
        ----------
        token_dict: dict
            key: code type, value: a list of tokens.
        '''
        for key in token_dict.keys():
            self.code_vocab[key] = np.array(token_dict[key])
            self.add_tokens(token_dict[key])

    def extend_vocab_from_dir(self, data_dir):
        # add new tokens from the data dir
        for key in self.new_token_type_list:
            filename = os.path.join(data_dir,'{}_token_list.txt'.format(key))
            with open(filename, 'r', encoding='utf-8') as f:
                token_list = [line.strip() for line in f.readlines()]
            self.code_vocab[key] = np.array(token_list)
            self.add_tokens(token_list)

        # add special tokens indicating different modality
        for key, value in self.special_token_dict.items():
            self.add_tokens(value, special_tokens=True)

class ModelTokenizer:
    r'''construct an EHR tokenizer that converts tokenized indices to code-specific token indices.
    '''
    def __init__(self, tokenizer: DataTokenizer):
        # map_token = lambda x: str(tokenizer(x).input_ids[1])
        org_vocab = tokenizer.get_vocab()
        tokenizer_dict = {}
        num_token_dict = {}
        label_offset = 1  # Default offset for special tokens (UNKNOWN_TOKEN = 0, so offset starts at 1)
        
        for key, value in tokenizer.code_vocab.items():
            vocab = defaultdict(int)
            vocab[UNKNOWN_TOKEN] = 0
            for i,token in enumerate(tokenizer.special_token_dict[key]):
                vocab[str(org_vocab[token])] = i+1
            offset = len(vocab)
            label_offset = offset  # Update with the last computed offset

            for i, token in enumerate(value): # str token = 'diag_xxx'
                # fix: if token has more than one '_', e.g., 'diag_t_a_b_100', will only take the last '100' as the index. 
                # _, index = token.split('_')
                indexes = token.split('_')
                try:
                    index = int(indexes[-1])
                except:
                    raise ValueError(f"Token {token} is not a valid token, it should be splited by '_' and the last part should be a number, e.g., 'diag_100'. ")
                vocab[str(org_vocab[token])] = index + offset
            
            # new tokenizer
            specific_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNKNOWN_TOKEN))
            specific_tokenizer.pre_tokenizer = Whitespace()

            # num_token_dict is decided by the max index instead of number of tokens
            num_token_dict[key] = (max(vocab.values())+1) - offset
            tokenizer_dict[key] = specific_tokenizer

        # each code type has its own tokenizer corresponding to specific LM heads
        self.tokenizer_dict = tokenizer_dict
        self.num_token_dict = num_token_dict
        self.label_offset = label_offset


    def encode(self, input_ids, code_type):
        if len(input_ids.shape) > 1: # a batch
            ids = self.encode_batch(input_ids, code_type)
        else:
            ids = self.tokenizer_dict[code_type].encode(input_ids.cpu().numpy().astype(str), is_pretokenized=True).ids
            ids = torch.tensor(ids, device=input_ids.device)
        
        return ids

    def encode_batch(self, input_ids, code_type):
        ids_list = self.tokenizer_dict[code_type].encode_batch(input_ids.cpu().numpy().astype(str).tolist(), is_pretokenized=True)

        ids = torch.tensor([x.ids for x in ids_list], device=input_ids.device)
        
        return ids

    @property
    def get_num_tokens(self):
        return self.num_token_dict

@dataclass
class EHRBartOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        perplexity:
            perplexity calculated when the label mask is given.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    perplexity: Optional[torch.FloatTensor] = None

class BartForEHRSimulation(BartPretrainedModel, GenerationMixin):
    '''BART model for EHR sequence simulation.
    Extend the BartPretrainedModel to support code-specific output and conditional prompt.
    '''
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = PromptBartModel(config)
        # build LM heads for different code types
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # build LM head for each code type
        self.lm_head_list = nn.ModuleDict()
        # get_num_tokens was stored in config during EHRBartConfig
        if hasattr(config, '_num_tokens_dict'):
            num_tokens_dict = config._num_tokens_dict
        else:
            # Fallback: try to find the token counts from config attributes
            num_tokens_dict = {}
            standard_attrs = set(dir(BartConfig()))
            for attr_name in dir(config):
                if (attr_name not in standard_attrs and 
                    not attr_name.startswith('_') and 
                    hasattr(config, attr_name)):
                    attr_value = getattr(config, attr_name)
                    if isinstance(attr_value, int) and attr_value > 0:
                        num_tokens_dict[attr_name] = attr_value
        
        for code_type in num_tokens_dict.keys():
            lm_head = nn.Linear(config.d_model, num_tokens_dict[code_type], bias=False)
            self.lm_head_list[code_type] = lm_head

        # Create a main lm_head for compatibility with transformers library
        # Use the vocabulary size from the shared embeddings
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _compute_loss(self,
        sequence_output,
        target_label,
        target_mask,
        code_type,
        return_logits=False,
        **kwargs
        ):
        '''Compute the cross entropy loss for given code type.
        '''
        lm_head = self.lm_head_list[code_type]
        
        # Ensure lm_head is on the same device as sequence_output
        if sequence_output is not None:
            device = sequence_output.device
            lm_head = lm_head.to(device)
            self.lm_head_list[code_type] = lm_head
        
        # if return logits only, does not compute loss
        if target_label is None:
            lm_logits = lm_head(sequence_output)
            return lm_logits

        # compute loss per code type
        lm_logits = lm_head(sequence_output)
        
        # Handle empty tensor case
        if lm_logits.numel() == 0 or target_label.numel() == 0:
            # Return zero loss for empty tensors
            return torch.tensor(0.0, device=lm_logits.device, requires_grad=True)
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # Ensure lm_logits and target_label have the same sequence length
        min_seq_len = min(lm_logits.size(1), target_label.size(1))
        lm_logits_trimmed = lm_logits[:, :min_seq_len, :]
        target_label_trimmed = target_label[:, :min_seq_len]
        
        # Clamp target labels to valid vocabulary range to prevent CUDA assertion errors
        vocab_size = lm_logits_trimmed.size(-1)
        target_label_trimmed = torch.clamp(target_label_trimmed, min=0, max=vocab_size-1)
        
        masked_lm_loss = loss_fct(lm_logits_trimmed.reshape(-1, lm_logits_trimmed.size(-1)), target_label_trimmed.reshape(-1))
        # mask out the loss for non-active predictions
        masked_lm_loss = masked_lm_loss.reshape(lm_logits_trimmed.size(0), lm_logits_trimmed.size(1))
        target_mask_trimmed = target_mask[:, :min_seq_len] if target_mask is not None else None
        if target_mask_trimmed is not None:
            masked_lm_loss = masked_lm_loss * target_mask_trimmed

        loss = masked_lm_loss.sum() / (target_mask_trimmed.sum() if target_mask_trimmed is not None else 1)

        if return_logits:
            return loss, lm_logits
        else:
            return loss

    def _get_perplexity(self,
        sequence_output,
        target_label,
        target_mask,
        code_type,
        **kwargs
        ):
        '''compute perplexity.
        '''
        with torch.no_grad():
            lm_head = self.lm_head_list[code_type]
            
            # Ensure lm_head is on the same device as sequence_output
            if sequence_output is not None:
                device = sequence_output.device
                lm_head = lm_head.to(device)
                self.lm_head_list[code_type] = lm_head
            
            lm_logits = lm_head(sequence_output)
            lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            
            # Ensure target_label indices are within vocabulary bounds
            vocab_size = lm_logits.size(-1)
            target_label_clamped = torch.clamp(target_label, min=0, max=vocab_size-1)
            
            # (bs, seq_len)
            picked_logits = torch.gather(lm_logits, 2, target_label_clamped.unsqueeze(-1)).squeeze(-1) 
            picked_logits = picked_logits * target_mask
            sum_picked_logits = picked_logits.sum(dim=-1) # (bs,)
            sum_target_mask = target_mask.sum(dim=-1) + EPS # (bs,)
            perplexity = torch.exp(-sum_picked_logits / sum_target_mask)
        return perplexity

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        x_num: Optional[torch.FloatTensor] = None,
        x_cat: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_mask: Optional[torch.LongTensor] = None,
        code_type: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_perplexity: Optional[bool] = None,
        **kwargs,
        ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        # Ensure all model components are on the same device as the input
        if input_ids is not None:
            device = input_ids.device
            if hasattr(self.model, 'device') and self.model.device != device:
                self.model = self.model.to(device)
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            x_num=x_num,
            x_cat=x_cat,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model.shared.embedding_dim ** -0.5)

        # compute loss / prediction
        loss = None
        perplexity = None
        if labels is not None and code_type is not None:
            # make sure label_mask exists when computing the loss
            assert label_mask is not None
            loss = self._compute_loss(sequence_output, labels, label_mask, code_type)
            if return_perplexity:
                perplexity = self._get_perplexity(sequence_output, labels, label_mask, code_type)

        logits = {}
        # compute all types logits when not training
        if labels is None:
            for code_type in self.lm_head_list.keys():
                logits[code_type] = self._compute_loss(sequence_output, None, None, code_type)
        else:
            # only return interested logits at training stage
            logits[code_type] = self._compute_loss(sequence_output, None, None, code_type)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return EHRBartOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            perplexity=perplexity,
        )

    def prepare_inputs_for_generation(self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        x_num=None,
        x_cat=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
        ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "x_num": x_num,
            "x_cat": x_cat,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

# Full MimicDataCollator from original PromptEHR implementation
class FullMimicDataCollator:
    '''Data collator for train/evaluate the EHR-BART model.
    Should keep the whole batch all with features or all without features,
    otherwise raise error!
    '''
    __code_type_list__ = CODE_TYPES
    __special_token_dict__ = SPECIAL_TOKEN_DICT
    __del_or_rep__ = ['rep', 'del']

    def __init__(self, 
        tokenizer,
        code_types,
        n_num_feature,
        mlm_prob=0.15, 
        lambda_poisson=3.0, 
        del_prob=0.15,
        max_train_batch_size=16, 
        drop_feature=False, 
        mode='train',
        eval_code_type=None,
        eval_ppl_type='span'
        ):
        '''mlm_prob: probability of masked tokens
        lambda_poisoon: span infilling parameters
        del_prob: probability of delete tokens
        max_train_batch_size: sample batch to avoid OOM, because for each patient we will generate a batch of series
        '''
        # update code_types
        self.__code_type_list__ = code_types
        self.__special_token_dict__ = {}
        for code in code_types: self.__special_token_dict__[code] = [f'<{code}>', f'</{code}>']

        self.mlm_prob = mlm_prob
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = MODEL_MAX_LENGTH
        self.mlm_probability = mlm_prob
        self.lambda_poisson = lambda_poisson
        self.del_probability = del_prob
        self.max_train_batch_size = max_train_batch_size # sample batch to avoid OOM
        self.eval_code_type = eval_code_type if eval_code_type is not None else (code_types[0] if code_types else None)
        self.eval_ppl_type = eval_ppl_type
        self.drop_feature = drop_feature
        self.n_num_feature = n_num_feature

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.is_training = (mode == 'train')
        self.is_testing = (mode == 'test')
    
    def __getstate__(self):
        """Custom pickling to ensure eval attributes are preserved in multiprocessing"""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore eval attributes in multiprocessing"""
        self.__dict__.update(state)
        # Ensure eval attributes exist after unpickling
        if not hasattr(self, 'eval_code_type'):
            self.eval_code_type = self.__code_type_list__[0] if hasattr(self, '__code_type_list__') and self.__code_type_list__ else None
        if not hasattr(self, 'eval_ppl_type'):
            self.eval_ppl_type = 'span'
        # Ensure mode attributes exist after unpickling
        if not hasattr(self, 'mode'):
            self.mode = 'val'  # Default to validation mode
        if not hasattr(self, 'is_training'):
            self.is_training = (self.mode == 'train')
        if not hasattr(self, 'is_testing'):
            self.is_testing = (self.mode == 'test')

    def __call__(self, samples: List[InputDataClass]) -> Dict[str, Any]:
        # samples format
        # [{'pid': 'x_num':[], 'x_cat':[], 'diagnosis':[[],[],[],...], 'procedure': [[],[]...], 'drug':[[],[],...] }]
        def _seq_patient_to_promptehr(samples):
            post_samples = []
            for sample in samples:
                post_sample = {}
                visit = sample['v']
                post_sample.update(visit)
                if ('x' in sample) and (self.n_num_feature is not None):
                    if not isinstance(sample['x'], list):
                        sample['x'] = sample['x'].tolist()
                    post_sample['x_num'] = sample['x'][:self.n_num_feature]
                    # Only add x_cat if there are categorical features remaining
                    remaining_features = sample['x'][self.n_num_feature:]
                    if remaining_features:  # Only add if non-empty
                        post_sample['x_cat'] = remaining_features
                post_samples.append(post_sample)
            return post_samples
        
        samples = _seq_patient_to_promptehr(samples)

        if self.is_training:
            batch = self.call_train(samples)
        elif self.is_testing:
            batch = self.call_test(samples)
        else:
            batch = self.call_val(samples)
        return batch

    def call_train(self, samples: List[InputDataClass]) -> Dict[str, Any]:
        '''label mask should not be used during training.
        '''
        batch = defaultdict(list)

        # randomly pick one of code types for prediction, keep the same for this batch
        code_type = random.sample(self.__code_type_list__, 1)[0]
        batch['code_type'] = code_type

        for sample in samples:
            num_adm = len(sample[code_type])

            # accumulated during enumerating all admisions
            input_str_all = []
            label_str_all = []
            num_token_all = []

            # cope with too long labtest codes
            # start from the offset if the labtest is too long
            adm = 0
            while adm < num_adm:
                span_str_list = [] # input ids
                span_label_str_list = [] # label ids
                num_token_this_adm = 0

                # shuffle the code order
                code_list = list(sample.keys())
                random.shuffle(code_list)
                for code in sample.keys():
                    if code in ['pid','x_num','x_cat']: continue

                    span = sample[code][adm]

                    if len(span) == 0: continue

                    # restrict the num of tokens in each span
                    span = random.sample(span, min(20, len(span)))

                    # translate span to code_span
                    span = self._process_span(span, code)

                    span_str = self._pad_special_token_head_tail(' '.join(span), code)
                    span_label_str_list.append(span_str)
                    num_token_this_adm += len(span) + 2

                    if code == code_type:
                        # do mask infilling / mask
                        infill_span, _, _ = self.mask_infill([span])
                        span_str = self._pad_special_token_head_tail(' '.join(infill_span[0]), code) 
                        span_str_list.append(span_str)
                    else:
                        if self.__del_or_rep__[random.randint(0,1)] == 'rep': 
                            rep_del_span = self.rep_token([span], code)
                        else: 
                            rep_del_span = self.del_token([span])

                        span_str = self._pad_special_token_head_tail(' '.join(rep_del_span[0]), code) 
                        span_str_list.append(span_str)

                span_str_list.append('</s>')
                span_label_str_list.append('</s>')
                num_token_this_adm += 1

                input_str_all.append(' '.join(span_str_list))
                label_str_all.append(' '.join(span_label_str_list))
                num_token_all.append(num_token_this_adm)

                # check break condition
                if len(input_str_all) >= self.max_train_batch_size:
                    break # do not sample too many examples to avoid OOM
                
                if adm < num_adm - 1:
                    total_token_next_adm = sum(num_token_all) + len(sample[code_type][adm+1]) + 10
                    if total_token_next_adm >= self.tokenizer.model_max_length - 10:
                        break # do not sample too many tokens to avoid break
                adm += 1

            # tokenization
            batch['input_ids'].extend(self.tokenizer(input_str_all, return_tensors='pt', padding=True, truncation=True, max_length=MODEL_MAX_LENGTH)['input_ids'])
            batch['labels'].extend(self.tokenizer(label_str_all, return_tensors='pt', padding=True, truncation=True, max_length=MODEL_MAX_LENGTH)['input_ids'])
            if 'x_num' in sample:
                if not self.drop_feature:
                    batch['x_num'].extend([torch.tensor(sample['x_num'], dtype=torch.float32)] * len(input_str_all))
            if 'x_cat' in sample:
                if not self.drop_feature:
                    batch['x_cat'].extend([torch.tensor(sample['x_cat'], dtype=torch.long)] * len(input_str_all))

        # padding
        batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True)
        batch['labels'] = pad_sequence(batch['labels'], batch_first=True)
        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).float()
        batch['label_mask'] = (batch['labels'] != self.tokenizer.pad_token_id).float()

        if 'x_num' in batch:
            batch['x_num'] = torch.stack(batch['x_num'])
        if 'x_cat' in batch:
            batch['x_cat'] = torch.stack(batch['x_cat'])

        return dict(batch)

    def call_val(self, samples: List[InputDataClass]) -> Dict[str, Any]:
        return self.call_test(samples)

    def call_test(self, samples: List[InputDataClass]) -> Dict[str, Any]:
        '''compute the preplexity for each code type.
        '''
        assert self.eval_code_type is not None
        code_type = self.eval_code_type
        assert self.eval_ppl_type is not None
        ppl_type = self.eval_ppl_type

        batch = defaultdict(list)
        batch['code_type'] = code_type

        for sample in samples:
            num_adm = len(sample[code_type])

            # accumulated during enumerating all admisions
            input_str_all = []
            label_str_all = []

            # cope with too long labtest codes
            # start from the offset if the labtest is too long
            adm = 0
            while adm < num_adm:
                span_str_list = [] # input ids
                span_label_str_list = [] # label ids

                for code in sample.keys():
                    if code in ['pid','x_num','x_cat']: continue

                    span = sample[code][adm]

                    if len(span) == 0: continue

                    # translate span to code_span
                    span = self._process_span(span, code)

                    span_str = self._pad_special_token_head_tail(' '.join(span), code)
                    span_label_str_list.append(span_str)

                    if code == code_type:
                        if ppl_type == 'spl': # single prediction loss
                            # do mask infilling / mask
                            infill_span, _, _ = self.mask_infill([span])
                            span_str = self._pad_special_token_head_tail(' '.join(infill_span[0]), code) 
                        elif ppl_type == 'tpl': # teacher forcing loss
                            span_str = self._pad_special_token_head_tail(' '.join(span), code) 
                        span_str_list.append(span_str)
                    else:
                        span_str = self._pad_special_token_head_tail(' '.join(span), code) 
                        span_str_list.append(span_str)

                span_str_list.append('</s>')
                span_label_str_list.append('</s>')

                input_str_all.append(' '.join(span_str_list))
                label_str_all.append(' '.join(span_label_str_list))
                adm += 1

            # tokenization
            batch['input_ids'].extend(self.tokenizer(input_str_all, return_tensors='pt', padding=True, truncation=True, max_length=MODEL_MAX_LENGTH)['input_ids'])
            batch['labels'].extend(self.tokenizer(label_str_all, return_tensors='pt', padding=True, truncation=True, max_length=MODEL_MAX_LENGTH)['input_ids'])
            if 'x_num' in sample:
                batch['x_num'].extend([torch.tensor(sample['x_num'], dtype=torch.float32)] * len(input_str_all))
            if 'x_cat' in sample:
                batch['x_cat'].extend([torch.tensor(sample['x_cat'], dtype=torch.long)] * len(input_str_all))

        # padding
        batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True)
        batch['labels'] = pad_sequence(batch['labels'], batch_first=True)
        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).float()
        batch['label_mask'] = (batch['labels'] != self.tokenizer.pad_token_id).float()

        if 'x_num' in batch:
            batch['x_num'] = torch.stack(batch['x_num'])
        if 'x_cat' in batch:
            batch['x_cat'] = torch.stack(batch['x_cat'])

        return dict(batch)

    def set_eval_code_type(self, code_type):
        self.eval_code_type = code_type

    def set_eval_ppl_type(self, ppl_type):
        self.eval_ppl_type = ppl_type

    def _process_span(self, span, code):
        return [code+'_'+str(s) for s in span]

    def _pad_special_token_head_tail(self, span_str, code):
        head_tag = self.__special_token_dict__[code][0] # <xx>
        tail_tag = self.__special_token_dict__[code][1] # </xx>
        return head_tag + ' ' + span_str + ' ' + tail_tag

    def mask_infill(self, spans):
        '''mask tokens and infill with <mask> token
        '''
        results = []
        org_tokens = []
        labels = []
        for span in spans:
            num_to_mask = max(1, int(self.mlm_probability * len(span)))
            
            if num_to_mask == len(span): 
                num_to_mask = len(span) - 1
            
            # randomly decide the mask length
            mask_length = np.random.poisson(self.lambda_poisson)
            mask_length = min(mask_length, num_to_mask)
            mask_length = max(mask_length, 1)

            # randomly decide the start position to mask
            start_pos = random.randint(0, len(span) - mask_length)
            
            new_span = span.copy()
            # replace the selected tokens with <mask>
            new_span[start_pos:start_pos+mask_length] = ['<mask>'] * mask_length
            results.append(new_span)
            org_tokens.append(span[start_pos:start_pos+mask_length])
            labels.append([start_pos, start_pos+mask_length])
        return results, org_tokens, labels

    def rep_token(self, spans, code):
        '''replace some tokens to the same modality randomly
        '''
        results = []
        for span in spans:
            num_to_rep = max(1, int(self.mlm_probability * len(span)))
            rep_idx = random.sample(range(len(span)), num_to_rep)
            new_span = span.copy()
            for idx in rep_idx:
                # randomly pick tokens from the same code vocab
                rep_tokens = self.tokenizer.code_vocab[code]
                rep_token_str = random.sample(rep_tokens.tolist(), 1)[0]
                new_span[idx] = rep_token_str
            results.append(new_span)
        return results

    def del_token(self, spans):
        '''delete some tokens for data corruption
        '''
        results = []
        for span in spans:
            num_to_del = max(1, int(self.del_probability * len(span)))
            if num_to_del == len(span): num_to_del = len(span) - 1
            del_idx = random.sample(range(len(span)), num_to_del)
            new_span = [span[i] for i in range(len(span)) if i not in del_idx]
            results.append(new_span)
        return results

class MimicDataCollator:
    '''Data collator with masking for MIMIC data.
    '''
    def __init__(self, tokenizer, model_tokenizer, mlm_probability=0.15, **kwargs):
        self.tokenizer = tokenizer
        self.model_tokenizer = model_tokenizer
        self.mlm_probability = mlm_probability
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, examples):
        # Handle dict or list of dicts
        if isinstance(examples[0], dict):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = self.tokenizer.pad(
                {"input_ids": examples}, return_tensors="pt"
            )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in batch["input_ids"].tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        vocab_size = len(self.tokenizer.get_vocab()) if hasattr(self.tokenizer, 'get_vocab') else len(self.tokenizer)
        random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

# PromptEHRTrainer from original implementation
class PromptEHRTrainer(Trainer):
    def __init__(self,
        model= None,
        args = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        val_data_collator=None,
        ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset)
        self.val_data_collator = val_data_collator if val_data_collator is not None else self.data_collator

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset, code_type):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].
        Subclass and override this method if you want to inject some custom behavior.
        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        self.val_data_collator.set_eval_code_type(code_type) # set evaluation for this code
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.val_data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=['encoder_last_hidden_state', 'past_key_values'],
        metric_key_prefix: str = "eval",
        ):
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # Use validation collator for proper evaluation setup
        eval_collator = self.val_data_collator if hasattr(self, 'val_data_collator') else self.data_collator
        eval_dataloader = self.get_eval_dataloader(eval_dataset, eval_collator.eval_code_type if hasattr(eval_collator, 'eval_code_type') else None)
        start_time = time.time()

        # Run evaluation loop
        eval_loss = 0.0
        nb_eval_steps = 0
        ppl_lists = {}
        
        # Initialize perplexity lists for each code type using validation collator
        if hasattr(eval_collator, 'eval_code_type') and eval_collator.eval_code_type:
            code_type = eval_collator.eval_code_type
            ppl_lists[code_type] = []
        
        self.model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                # Explicitly request perplexity computation during evaluation
                batch['return_perplexity'] = True
                outputs = self.model(**batch)
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    eval_loss += outputs.loss.mean().item()
                
                # Collect perplexity if available
                if hasattr(outputs, 'perplexity') and outputs.perplexity is not None:
                    code_type = eval_collator.eval_code_type
                    if code_type and code_type in ppl_lists:
                        batch_ppl = outputs.perplexity.cpu().flatten().tolist()
                        ppl_lists[code_type].extend(batch_ppl)
                
                nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0.0
        
        metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_runtime": time.time() - start_time,
            f"{metric_key_prefix}_samples": len(eval_dataloader.dataset) if hasattr(eval_dataloader.dataset, '__len__') else 0,
        }
        
        # Add perplexity metrics
        for code_type, ppl_list in ppl_lists.items():
            if ppl_list:
                ppl_ar = np.array(ppl_list)
                metrics[f"{metric_key_prefix}_ppl_{code_type}"] = float(np.median(ppl_ar))

        return metrics

# Evaluator from original implementation
class Evaluator:
    def __init__(self, model, dataset, collate_fn, device=None):
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.device = 'cpu' if device is None else device

    def evaluate(self, code_type, ppl_type, eval_batch_size):
        mimic_val_dataset = self.dataset
        mimic_val_collator = self.collate_fn
        mimic_val_collator.set_eval_code_type(code_type)
        mimic_val_collator.set_eval_ppl_type(ppl_type)
        dataloader = DataLoader(mimic_val_dataset,
            batch_size=eval_batch_size,
            num_workers=0,
            drop_last=False,
            collate_fn=mimic_val_collator,
            shuffle=False,
            pin_memory=False)

        ppl_list = []
        for batch in dataloader:
            if batch is not None:
                batch = self._prepare_inputs(batch)
                with torch.no_grad():
                    outputs = self.model(**batch)
                batch_ppl = outputs.perplexity
                batch_ppl = batch_ppl.cpu().flatten().tolist()
                ppl_list.extend(batch_ppl)
        ppl_ar = np.array(ppl_list)
        return np.median(ppl_ar)

    def _prepare_inputs(self, data):
        return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.device)
            return data.to(**kwargs)
        return data

class PromptEHR(nn.Module):
    '''
    Initialize a PromptEHR model to leverage language models to simulate sequential patient EHR data.
    Adapted from original PromptEHR implementation for PyHealth integration.
    Preserves original BART-based architecture with conditional prompts.

    Parameters:
    -----------
    dataset: SampleDataset
        PyHealth dataset containing patient records.

    code_type: list[str]
        A list of code types that the model will learn and generate.
        For example, `code_type=['diag','prod','med']`.

    token_dict: dict[list]
        A dictionary of new tokens (code events, e.g., ICD code) that the model needs to learn and generate.

    n_num_feature: int (default=None)
        Number of numerical patient baseline features. Notice that it assumes that the input
        baseline features are `ALWAYS` numerical feature first. That is to say,
        the input baseline feature = [num1, num2, .., num_n, cat1, cat2,...].
        If not specified, the model will never include baseline features
        for conditional generation!

    cat_cardinalities: list[int]
        The number of categories for each categorical patient baseline features.
        The input baseline feature = [num1, num2, .., num_n, cat1, cat2,...].

    device: str or list[int]
        Should be str like `cuda:0` or `cpu`, otherwise should be a list GPU ids.
    '''
    sample_config = {
        'num_beams': 1, # >1: beam_sample; =1: sample_gen
        'no_repeat_ngram_size': 1,
        'do_sample': True,
        'num_return_sequences': 1,
        'code_type': 'diagnosis',
        'top_k': 1,
        'temperature': 1.0,
        'max_length': 6,
    }
    
    def __init__(self,
        code_type=None,
        n_num_feature=None,
        cat_cardinalities=None,
        token_dict=None,
        epoch=50,
        batch_size=16,
        eval_batch_size=16,
        eval_step=1000,
        learning_rate=5e-5,
        weight_decay=1e-4,
        num_worker=8,
        output_dir='./promptEHR_logs',
        device='cuda:0',
        seed=123,
        **kwargs
        ) -> None:
        super().__init__()
        
        # Initialize tokenizers from original implementation
        self.data_tokenizer = DataTokenizer.from_pretrained('facebook/bart-base')

        # will extend vocab after pass training data
        if code_type is not None:
            self.data_tokenizer.update_special_token_config(code_types=code_type)
        if token_dict is not None:
            self.data_tokenizer.extend_vocab(token_dict)
            
        self.model_tokenizer = None  # Will be created during fit() like in original
        
        # Debug: Print vocabulary sizes
        bart_vocab_size = len(self.data_tokenizer)  # Use data_tokenizer length, not model_tokenizer
        print(f"BART model vocabulary size: {bart_vocab_size}")
        for ct in code_type:
            # Use model_tokenizer.tokenizer_dict since that has the get_vocab() method
            if self.model_tokenizer and ct in self.model_tokenizer.tokenizer_dict:
                data_vocab_size = len(self.model_tokenizer.tokenizer_dict[ct].get_vocab())
                print(f"{ct} data tokenizer vocab size: {data_vocab_size}")
                if data_vocab_size > bart_vocab_size:
                    print(f"WARNING: {ct} vocab ({data_vocab_size}) exceeds BART vocab ({bart_vocab_size})")
            else:
                print(f"{ct}: tokenizer will be built during fit()")
        
        self.config = {
            'code_type': code_type,
            'n_num_feature':n_num_feature,
            'cat_cardinalities':cat_cardinalities,
            'epoch':epoch,
            'batch_size':batch_size,
            'eval_batch_size':eval_batch_size,
            'eval_step':eval_step,
            'learning_rate':learning_rate,
            'weight_decay':weight_decay,
        }
        self.device_name = device
        if isinstance(device, list):
            self._set_visible_device(device=device)
        
        # Add training arguments from original implementation (deferred to avoid accelerate dependency)
        self.training_args = None
        self._training_config = {
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': eval_batch_size,
            'gradient_accumulation_steps': 1,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'output_dir': output_dir,
            'num_train_epochs': epoch,
            'save_steps': eval_step,
            'eval_steps': eval_step,
            'warmup_ratio': 0.06,
            'max_grad_norm': 0.5,
            'save_total_limit': 5,
            'logging_steps': eval_step,
            'dataloader_num_workers': num_worker,
            'dataloader_pin_memory': True,
            'eval_strategy': 'steps',
            'metric_for_best_model': f'eval_ppl_{code_type[0]}' if code_type is not None else None,
            'greater_is_better': False,
            'eval_accumulation_steps': 10,
            'load_best_model_at_end': True,
            'logging_dir': output_dir,
            'overwrite_output_dir': True,
            'seed': seed,
            'no_cuda': True if self.device_name == 'cpu' else False,
        }

        # avoid dead clock when taking multiple workers for dataloaders
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Model will be built during fit() like in original
        self.model = None


    def predict(self, test_data, n_per_sample=None, n=None, sample_config=None, verbose=None):
        '''
        Generate synthetic records based on input real patient seq data.

        Parameters
        ----------
        test_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        n: int
            How many samples in total will be generated.

        n_per_sample: int
            How many samples generated based on each indivudals.

        sample_config: dict
            Configuration for sampling synthetic records, key parameters:
            'num_beams': Number of beams in beam search, if set `1` then beam search is deactivated;
            'top_k': Sampling from top k candidates.
            'temperature': temperature to make sampling distribution flater or skewer.
        
        verbose: bool
            If print the progress bar or not.

        Returns
        -------
        Synthetic patient records in `SequencePatient` format.
        '''
        if n is not None: assert isinstance(n, int), 'Input `n` should be integer.'
        if n_per_sample is not None: assert isinstance(n_per_sample, int), 'Input `n_per_sample` should be integer.'
        assert (not n_per_sample is None) or (not n is None), 'Either `n` or `n_per_sample` should be provided to generate.'
        assert isinstance(self.model, BartForEHRSimulation), 'Model not found! Please fit the model or load the model from pretrained checkpoint first.'

        n, n_per_sample = self._compute_n_per_sample(len(test_data), n, n_per_sample)

        if sample_config is not None:
            self.sample_config.update(sample_config)
            print('### Sampling Config ###')
            print(self.sample_config)

        # get test data loader
        test_dataloader = self._get_test_dataloader(test_data)

        # make generation
        outputs = self._predict_on_dataloader(test_dataloader, n, n_per_sample, verbose=verbose)

        # formulate outputs to standard sequencepatient data
        # need 'visit', 'order', 'feature', 'n_num_feature', 'cat_cardinalties'
        visits, features, labels = [], [], []
        for output in outputs:
            code_types = [c for c in self.config['code_type'] if c in output]
            num_visit = len(output[code_types[0]])
            visit, feature = [], []
            for n in range(num_visit):
                visit_ = [output[code][n] for code in code_types]
                visit.append(visit_)
            visits.append(visit)
            if 'x_num' in output:
                feature.extend(output['x_num'])
            if 'x_cat' in output:
                feature.extend(output['x_cat'])
            if len(feature) > 0:
                features.append(feature)
            if 'y' in output: labels.append(output['y'])
        
        if len(features) > 0:
            features = np.stack(features, 0)
        else:
            features = None

        return_res = {
            'visit':visits, 
            'feature':features, 
            'order':self.config['code_type'],
            'n_num_feature':self.config['n_num_feature'],
            'cat_cardinalties':self.config['cat_cardinalities'],
            'y':labels,
            'voc': test_data.metadata['voc'],
        }
        return return_res

    # fit() method from original PromptEHR implementation
    def fit(self, train_data, val_data=None):
        '''
        Fit PromptEHR model on the input training EHR data.

        Parameters
        ----------
        train_data: SequencePatient
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.

        val_data: dict
            A `SequencePatient` contains patient records where 'v' corresponds to 
            visit sequence of different events.
        '''
        # create tokenizers based on the input data
        self._create_tokenizers(train_data)

        # can only build model after fit
        self._build_model()

        # start training
        self._fit(train_data=train_data,val_data=val_data)

    def save_model(self, output_dir):
        '''
        Save the learned simulation model to the disk.

        Parameters
        ----------
        output_dir: str
            The dir to save the learned model.
        '''
        make_dir_if_not_exist(output_dir)
        self._save_config(config=self.config, output_dir=output_dir)
        self._save_checkpoint(output_dir=output_dir)
        print('Save the trained model to:', output_dir)
    
    def from_pretrained(self, input_dir='./simulation/pretrained_promptEHR'):
        '''
        Load pretrained PromptEHR model and make patient EHRs generation.
        Pretrained model was learned from MIMIC-III patient sequence data.
        '''
        if input_dir is None:
            input_dir = './simulation/pretrained_promptEHR'
        
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)            
            url = PRETRAINED_MODEL_URL
            download_pretrained(url, input_dir)
            print(f'Download pretrained PromptEHR model, save to {input_dir}.')
        
        print('Load pretrained PromptEHR model from', input_dir)
        self.load_model(input_dir)

    def load_model(self, checkpoint):
        '''
        Load model and the pre-encoded trial embeddings from the given
        checkpoint dir.

        Parameters
        ----------
        checkpoint: str
            The input dir that stores the pretrained model.
        '''
        checkpoint_filename = check_checkpoint_file(checkpoint)
        config_filename = check_model_config_file(checkpoint)
        data_tokenizer_file, model_tokenizer_file = check_tokenizer_file(checkpoint)

        # load config
        self.config = self._load_config(config_filename)

        # load data tokenizer and model tokenizer
        self._load_tokenizer(data_tokenizer_file, model_tokenizer_file)

        # load configuration
        self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=self.config['n_num_feature'], cat_cardinalities=self.config['cat_cardinalities'])
        self.configuration.from_pretrained(checkpoint)

        # build model
        self._build_model()

        # load checkpoint
        state_dict = torch.load(checkpoint_filename, map_location='cpu')
        self.load_state_dict(state_dict, strict=True)
        print('Load the pre-trained model from:', checkpoint)

    def _build_model(self):
        """Build the BartForEHRSimulation model using the current configuration."""
        self.model = BartForEHRSimulation(self.configuration)
        self._setup_device()

    def _setup_device(self):
        # check if cuda is available using torch
        if not torch.cuda.is_available():
            warnings.warn('No GPU found, using CPU instead.')
            self.device_name = 'cpu'

        if isinstance(self.device_name, list): 
            self._set_visible_device(self.device_name)
            self.model.cuda()
        elif 'cuda' in self.device_name: 
            self.model.cuda()
        else:
            # on cpu
            self._set_visible_device([])
            self.model.cpu()

    def _set_visible_device(self, device):
        if len(device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(d) for d in device])
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def _compute_n_per_sample(self, n_test_sample, n=None, n_per_sample=None):
        if n_per_sample is not None:
            n_total = n_test_sample*n_per_sample
            if n is not None:
                n_total = min(n_total, n)
            return n_total, n_per_sample
        else:
            return n, math.ceil(n / n_test_sample)

    def _get_test_dataloader(self, dataset):
        def _seq_patient_to_promptehr(samples):
            post_samples = []
            for sample in samples:
                post_sample = {}
                visit = sample['v']
                post_sample.update(visit)

                if ('x' in sample) and (self.config['n_num_feature'] is not None):
                    if not isinstance(sample['x'], list): 
                        sample['x'] = sample['x'].tolist()
                    post_sample['x_num'] = torch.tensor(sample['x'][:self.config['n_num_feature']])
                    post_sample['x_cat'] = torch.tensor(sample['x'][self.config['n_num_feature']:], dtype=int)

                if 'y' in sample:
                    post_sample['y'] = sample['y']

                post_samples.append(post_sample)
            return post_samples

        dataloader = DataLoader(dataset,
                batch_size=1, # one patient once
                drop_last=False,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                collate_fn=_seq_patient_to_promptehr,
                )
        return dataloader

    def _save_config(self, config, output_dir=None):        
        temp_path = os.path.join(output_dir, 'promptehr_config.json')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(config, indent=4)
            )

        # save the data tokenizer and model tokenizer of the model
        temp_path = os.path.join(output_dir, 'data_tokenizer.pkl')
        with open(temp_path, 'wb') as f:
            pickle.dump(self.data_tokenizer, f)

        temp_path = os.path.join(output_dir, 'model_tokenizer.pkl')
        with open(temp_path, 'wb') as f:
            pickle.dump(self.model_tokenizer, f)

        # save configuration
        self.configuration.save_pretrained(output_dir)

    def _load_tokenizer(self, data_tokenizer_file, model_tokenizer_file):
        with open(data_tokenizer_file, 'rb') as f:
            self.data_tokenizer = pickle.load(f)
        self.data_tokenizer._in_target_context_manager = False # fix bugs when upgrade transformers to 4.23

        with open(model_tokenizer_file, 'rb') as f:
            self.model_tokenizer = pickle.load(f)

    def _load_config(self, filename):
        with open(filename, 'r') as f:
            config = json.load(f)
        return config

    def _save_checkpoint(self,
                        epoch_id=0,
                        is_best=False,
                        output_dir=None,
                        filename='checkpoint.pth.tar'):

        if epoch_id < 1:
            filepath = os.path.join(output_dir, 'latest.' + filename)
        elif is_best:
            filepath = os.path.join(output_dir, 'best.' + filename)
        else:
            filepath = os.path.join(output_dir, str(epoch_id) + '.' + filename)
        
        # save statedict
        state_dict = self.state_dict()
        torch.save(state_dict, filepath)

    def _predict_on_dataloader(self, dataloader, n, n_per_sample, verbose=None):
        total_number = 0
        data_iterator = iter(dataloader)

        if verbose:
            pbar = tqdm(total=n)

        new_record_list = []
        while total_number < n:
            try:
                data = next(data_iterator)
            except:
                data_iterator = iter(dataloader)
                data = next(data_iterator)            
            data = data[0] # batch size is 1 when doing generation

            # to device
            device = 'cpu' if self.device_name == 'cpu' else 'cuda:0'
            if 'x_num' in data: data['x_num'] = data['x_num'].to(device)
            if 'x_cat' in data: data['x_cat'] = data['x_cat'].to(device)                
            
            inputs = self._prepare_input_for_generation(data) 

            # start generation
            for _ in range(n_per_sample):
                new_record = self._generation_loop(data, inputs)
                if 'x_cat' in data:
                    new_record.update({
                        'x_cat':data['x_cat'].cpu().numpy().tolist(),
                    })
                    
                if 'x_num' in data:
                    new_record.update({
                        'x_num':data['x_num'].cpu().numpy().tolist(),
                    })

                # add more features to new_record
                for k,v in data.items():
                    if k not in new_record:
                        new_record[k] = v
                new_record_list.append(new_record)
            
            total_number += n_per_sample
            if verbose:
                pbar.update(n_per_sample)
                    
        if verbose:
            pbar.close()
        return new_record_list

    def _prepare_input_for_generation(self, data):        
        def _process_span(span, code):
            return [code+'_'+str(s) for s in span]
        
        def _to_device(x, device):
            for k,v in x.items():
                x[k] = v.to(device)
            return x

        tokenizer = self.data_tokenizer
        code_type = [k for k in data.keys() if k in self.config['code_type']]
        num_visit = len(data[code_type[0]])
        
        # init codes
        init_code = random.sample(data[code_type[0]][0], 1)
        init_code_str = _process_span(init_code, code_type[0])
        init_codes = tokenizer(init_code_str, return_tensors='pt', add_special_tokens=False)
        bos = torch.tensor([tokenizer.bos_token_id], device=self.model.device)
        code_prompt_idx = tokenizer.encode(tokenizer.special_token_dict[code_type[0]], add_special_tokens=False, return_tensors='pt')
        init_input_ids = torch.cat([bos[:,None],code_prompt_idx[:,0,None],init_codes['input_ids']], dim=-1)
        init_input_ids = _to_device({'input_ids':init_input_ids}, self.model.device)['input_ids']
        input_ids = init_input_ids.clone()
        return {'input_ids':input_ids, 'init_input_ids':init_input_ids, 'num_visit':num_visit, 'init_code':init_code}

    def _generation_loop(self, data, inputs):
        new_record = defaultdict(list)
        tokenizer = self.data_tokenizer
        special_token_dict = self.data_tokenizer.special_token_dict
        sample_gen_kwargs = self.sample_config.copy()

        input_ids_list = []
        num_visit_code_list = []
        first_code_flag = True

        input_ids = inputs['input_ids']
        for visit in range(inputs['num_visit']):
            this_visit_ids_list = []
            for code in self.config['code_type']:
                target_list = data[code][visit]
                sample_gen_kwargs['code_type'] = code
                num_code = len(target_list)
                if num_code > 20:
                    num_code = min(num_code, 20)
                    target_list = np.random.choice(target_list, num_code, replace=False).tolist()

                # random select part of codes from target list
                target_ar = np.array(target_list)
                sub_code = target_ar[np.random.binomial(1, 0.5, num_code).astype(bool)]
                code_prompt_idx = [special_token_dict[code][0]] + sub_code.tolist() + [special_token_dict[code][1]]
                code_prompt_idx = tokenizer.encode(code_prompt_idx, add_special_tokens=False, return_tensors='pt')
                code_prompt_idx = code_prompt_idx.to(self.model.device)

                if num_code == 0:
                    if first_code_flag:
                        new_next_tokens = code_prompt_idx[:,-1,None]
                        first_code_flag = False
                    else:
                        new_next_tokens = code_prompt_idx

                    this_visit_ids_list.append(new_next_tokens)
                    input_ids = torch.cat([input_ids, new_next_tokens], dim=-1)
                    new_record[code].append([])
                
                else:
                    sample_gen_kwargs['max_length'] = num_code+2

                    # do conditional generation
                    if 'x_cat' in data:
                        sample_gen_kwargs['x_cat'] = data['x_cat']
                    if 'x_num' in data:
                        sample_gen_kwargs['x_num'] = data['x_num']

                    new_next_tokens = self.model.generate(input_ids, **sample_gen_kwargs)

                    # randomly pick / rm sub code overlap
                    new_next_tokens = new_next_tokens[:,1:-1]
                    new_next_tokens = np.setdiff1d(new_next_tokens[0].cpu(), code_prompt_idx[0].cpu())
                    
                    try:
                        if num_code-len(sub_code) > len(new_next_tokens):
                            new_sub_idxs = np.unique(np.random.choice(np.arange(len(new_next_tokens)), num_code-len(sub_code), replace=True))
                        else:
                            new_sub_idxs = np.unique(np.random.choice(np.arange(len(new_next_tokens)), num_code-len(sub_code), replace=False))
                    except:
                        pdb.set_trace()
                        pass
                    new_next_tokens = torch.tensor(new_next_tokens[None, new_sub_idxs]).to(code_prompt_idx.device)

                    # append to the synthetic record dict
                    code_str_list = tokenizer.batch_decode(new_next_tokens)[0]

                    # remove special tokens ahead of original code event
                    # e.g., `diag_384` -> `384`
                    code_str_list = code_str_list.replace(code+'_','')
                    code_str_list = code_str_list.split()
                    code_str_list = [int(c) for c in code_str_list+sub_code.tolist()]
                    new_record[code].append(list(set(code_str_list)))

                    if first_code_flag:
                        new_next_tokens = torch.cat([new_next_tokens, code_prompt_idx[:,1:]], dim=-1)
                        first_code_flag = False
                    else:
                        # cover by modality prompt
                        new_next_tokens = torch.cat([code_prompt_idx[:,:-1], new_next_tokens, code_prompt_idx[:,-1,None]], dim=-1)

                    if visit > 1:
                        # check input length
                        cur_len = input_ids.shape[1] + new_next_tokens.shape[1]
                        while cur_len >= tokenizer.model_max_length:
                            print(f'{cur_len} reach model max length {tokenizer.model_max_length}, do cut.')
                            input_ids_list = input_ids_list[1:]
                            num_visit_code_list = num_visit_code_list[1:]
                            input_ids = torch.cat(input_ids_list,dim=-1)
                            cur_len = input_ids.shape[1] + new_next_tokens.shape[1]

                    # concat
                    this_visit_ids_list.append(new_next_tokens)
                    input_ids = torch.cat([input_ids, new_next_tokens], dim=-1)

            # after one visit, add eos token id
            eos = torch.tensor([tokenizer.eos_token_id], device=self.model.device)
            input_ids = torch.cat([input_ids, eos[:,None]], dim=-1)
            this_visit_ids = torch.cat(this_visit_ids_list, dim=-1)
            this_visit_ids = torch.cat([this_visit_ids, eos[:,None]], dim=-1)
            if visit == 0: this_visit_ids = torch.cat([inputs['init_input_ids'], this_visit_ids], dim=-1)
            num_visit_code_list.append(this_visit_ids.shape[-1])
            input_ids_list.append(this_visit_ids)

        # add init code
        new_record[self.config['code_type'][0]][0] += inputs['init_code']
        return new_record

    # _create_tokenizers() method from original PromptEHR implementation
    def _create_tokenizers(self, train_data):
        # update data_tokenizer first
        def _collate_fn(inputs):
            outputs = defaultdict(list)
            for input in inputs:
                visit = input['v']
                for k,v in visit.items():
                    code_list = sum(v,[])
                    code_list = [k+'_'+str(c) for c in list(set(code_list))]
                    outputs[k].extend(code_list)
            return outputs
        dataloader = DataLoader(train_data, collate_fn=_collate_fn, batch_size=512, shuffle=False)
        for batch in dataloader:
            for k,v in batch.items():
                unq_codes = list(set(v))
                self.data_tokenizer.add_token_to_code_vocab(unq_codes, k)
        
        self.model_tokenizer = ModelTokenizer(self.data_tokenizer)
        self.configuration = EHRBartConfig(self.data_tokenizer, self.model_tokenizer, n_num_feature=self.config['n_num_feature'], cat_cardinalities=self.config['cat_cardinalities'])
        self.data_tokenizer.update_special_token_config(code_types=self.config['code_type'])

    def _build_model(self):
        """Build the BartForEHRSimulation model using the current configuration."""
        self.model = BartForEHRSimulation(self.configuration)
        self._setup_device()

    def _fit(self, train_data, val_data):
        # Create TrainingArguments only when training is actually called
        if self.training_args is None:
            self.training_args = TrainingArguments(**self._training_config)
        
        mimic_train_collator = FullMimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            max_train_batch_size=self.config['batch_size'], mode='train')

        # Create validation collator with eval parameters set during initialization
        eval_code_type = self.config['code_type'][0] if self.config['code_type'] else None
        mimic_val_collator = FullMimicDataCollator(self.data_tokenizer, 
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            mode='val',
            eval_code_type=eval_code_type,
            eval_ppl_type='span')

        trainer = PromptEHRTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            data_collator=mimic_train_collator,
            eval_dataset=val_data,
            val_data_collator=mimic_val_collator,
            )
        try:
            trainer.train()
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise e

    # evaluate() method from original PromptEHR implementation
    def evaluate(self, test_data):
        '''
        Evaluate the trained PromptEHR model on the input data, will test the perplexity
        for each type of codes.
        
        Parameters
        ----------
        test_data: PatientSequence
            Standard sequential patient records in `PatientSequence` format.
        '''
        self.model.eval()
        self.eval()

        collator = FullMimicDataCollator(
            self.data_tokenizer,
            code_types=self.config['code_type'],
            n_num_feature=self.config['n_num_feature'],
            mode='test', 
            drop_feature=False
            )

        evaluator = Evaluator(
            self.model,
            test_data,
            collator,
            device='cpu' if self.device_name == 'cpu' else 'cuda:0',
        )

        code_types = self.config['code_type']
        ppl_types = ['tpl','spl']
        for code_type in code_types:
            for ppl_type in ppl_types:
                ppl = evaluator.evaluate(code_type, ppl_type, eval_batch_size=self.config['eval_batch_size'])
                print(f'code: {code_type}, ppl_type: {ppl_type}, value: {ppl}')

    # update_config() method from original PromptEHR implementation    
    def update_config(self, config):
        '''
        Update the configuration of the model.

        Parameters
        ----------
        config: dict
            The configuration of the model.
            Refer to the `config` in `__init__` for more details.
        '''
        self.config.update(config)
        
        # update training args
        train_args = copy.deepcopy(config)
        for k, v in config.items():
            if k in config_to_train_args:
                train_args[config_to_train_args[k]] = v
                train_args.pop(k)
        
        for k,v in train_args.items():
            if hasattr(self.training_args, k):
                setattr(self.training_args, k, v)
        
        # important when you train the model with different datasets
        code_type = self.config['code_type']
        self.training_args.metric_for_best_model = \
            f'eval_ppl_{code_type[0]}' if code_type is not None else None,

        print('### Model Config ###')
        print(self.config)

        print('### Training Args ###')
        print(self.training_args)

# Utility functions from original implementation
def download_pretrained(url, output_dir):
    import wget
    import zipfile
    filename = wget.download(url=url, out=output_dir)
    zipf = zipfile.ZipFile(filename, 'r')
    zipf.extractall(output_dir)
    zipf.close()

def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_checkpoint_file(input_dir, suffix='pth.tar'):
    '''
    Check whether the `input_path` is directory or to the checkpoint file.
        If it is a directory, find the only 'pth.tar' file under it.

    Parameters
    ----------
    input_path: str
        The input path to the pretrained model.
    suffix: 'pth.tar' or 'model'
        The checkpoint file suffix;
        If 'pth.tar', the saved model is a torch model.
        If 'model', the saved model is a scikit-learn based model.
    '''
    suffix = '.' + suffix
    if input_dir.endswith(suffix):
        return input_dir

    ckpt_list = glob.glob(os.path.join(input_dir, '*'+suffix))
    assert len(ckpt_list) <= 1, f'Find more than one checkpoints under the dir {input_dir}, please specify the one to load.'
    assert len(ckpt_list) > 0, f'Do not find any checkpoint under the dir {input_dir}.'
    return ckpt_list[0]

def check_model_config_file(input_dir):
    '''
    Check whether the `input_path` is directory or to the `model_config.json` file.
        If it is a directory, find the only '.json' file under it.

    Parameters
    ----------
    input_path: str
        The input path to the pretrained model.
    '''
    if input_dir.endswith('.json'):
        return input_dir

    if not os.path.isdir(input_dir):
        # if the input_dir is the given checkpoint model path,
        # we need to find the config file under the same dir.
        input_dir = os.path.dirname(input_dir)

    ckpt_list = glob.glob(os.path.join(input_dir, '*.json'))

    if len(ckpt_list) == 0:
        return None

    # find model_config.json under this input_dir
    model_config_name = [config for config in ckpt_list if 'promptehr_config.json' in config]
    if len(model_config_name) == 1:
        return model_config_name[0]

    # if no model_config.json found, retrieve the only .json file.
    assert len(ckpt_list) <= 1, f'Find more than one config .json under the dir {input_dir}.'
    return ckpt_list[0]

def check_tokenizer_file(input_dir):
    return os.path.join(input_dir,'data_tokenizer.pkl'), os.path.join(input_dir,'model_tokenizer.pkl')