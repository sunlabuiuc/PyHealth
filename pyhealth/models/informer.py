# Author: Zhiping Yang
# NetID: zy55
# Description: informer model implementation for PyHealth

import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.interpret.api import CheferInterpretable
from datetime import datetime, timedelta

class TriangularCausalMask():
    """return a mask tensor with element set to 1 for element on and above diagonal and 0 else"""
    """Args:
        B: Batch size.
        L: Sequence length; the mask covers all L query positions.
        device: Target device for the mask tensor (default ``"cpu"``).

    Returns:
            _mask: Boolean tensor of shape ``[B, 1, L, L]`` where ``True``
                marks positions that must be blocked (upper triangle,
                diagonal excluded).

    Example:
            Typically passed to ``masked_fill`` before softmax inside a
            standard Transformer decoder self-attention layer::
            mask = TriangularCausalMask(B, L, device=x.device)
            scores.masked_fill(mask.mask, float('-inf'))
    """
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self)->torch.tensor:
        """Full causal attention mask for standard self-attention.
 
            Constructs an upper-triangular boolean mask that prevents every query
            position from attending to any future key position, enforcing
            autoregressive behaviour across the entire sequence.

        Args:
                B(int): Batch size.
                L(int): Sequence length; the mask covers all L query positions.
                device: Target device for the mask tensor (default ``"cpu"``).

        Returns:
                _mask: Boolean tensor of shape ``[B, 1, L, L]`` where ``True``
                    marks positions that must be blocked (upper triangle,
                    diagonal excluded).

        Example:
                Typically passed to ``masked_fill`` before softmax inside a
                standard Transformer decoder self-attention layer::
                mask = TriangularCausalMask(B, L, device=x.device)
                scores.masked_fill(mask.mask, float('-inf'))
        """
        return self._mask

class ProbMask():
    """Sparse causal attention mask for Informer's ProbSparse self-attention.

    Builds a causal mask aligned to only the top-u query indices selected
    by the ProbSparse sampling step, rather than the full sequence length.
    Ensures that sparse queries still cannot attend to future key positions,
    preserving causality on the reduced ``[B, H, L_sparse, L_keys]`` score
    grid.
    """
    """Args:
    B: Batch size.
    H: Number of attention heads.
    L: Full query sequence length used to construct the base causal mask.
    index: LongTensor of shape ``[B, H, L_sparse]`` containing the
            query-row indices selected by ProbSparse sampling.
    scores: Sparse attention score tensor of shape
            ``[B, H, L_sparse, L_keys]``; its last dimension determines
            the key length and its overall shape is used for the final
            reshape.
    device: Target device for the mask tensor (default ``"cpu"``).

    Returns:
    _mask: Boolean tensor of shape ``[B, H, L_sparse, L_keys]`` where
            ``True`` marks future key positions to be blocked.

    Example:
    Applied inside ProbSparse attention after scores are computed for
    the sampled query subset::

        mask = ProbMask(B, H, L, index, scores, device=x.device)
        scores.masked_fill(mask.mask, float('-inf'))"""


    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self)->torch.tensor:
        """Args:
        B: Batch size.
        H: Number of attention heads.
        L: Full query sequence length used to construct the base causal mask.
        index: LongTensor of shape ``[B, H, L_sparse]`` containing the
               query-row indices selected by ProbSparse sampling.
        scores: Sparse attention score tensor of shape
                ``[B, H, L_sparse, L_keys]``; its last dimension determines
                the key length and its overall shape is used for the final
                reshape.
        device: Target device for the mask tensor (default ``"cpu"``).

        Returns:
        _mask: Boolean tensor of shape ``[B, H, L_sparse, L_keys]`` where
               ``True`` marks future key positions to be blocked."""
        return self._mask
    
class FullAttention(nn.Module):
    """standard scale dot product for attention"""
    """Compute full scaled dot-product attention over the entire sequence.

    Calculates attention scores across all query-key pairs, optionally
    applies a causal mask, normalises with softmax, and aggregates value
    vectors into the output. Complexity is O(L * S) in both time and
    memory.

    Args:
        queries: Query tensor of shape ``[B, L, H, E]`` where ``B`` is
                batch size, ``L`` query length, ``H`` heads, ``E``
                head dimension.
        keys: Key tensor of shape ``[B, S, H, E]`` where ``S`` is the
            key sequence length.
        values: Value tensor of shape ``[B, S, H, D]`` where ``D`` is
                the value head dimension.
        attn_mask: Optional :class:`TriangularCausalMask` instance. If
                ``mask_flag`` is ``True`` and this is ``None``, a
                causal mask is created automatically from ``B`` and
                ``L``.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - **V** - Attention output of shape ``[B, L, H, D]``.
            - **A** - Attention weight matrix of shape
            ``[B, H, L, S]`` if ``output_attention=True``,
            otherwise ``None``.

    Example:
        Used as a drop-in full-attention backend inside
        :class:`AttentionLayer`, passed alongside
        :class:`TriangularCausalMask` in the decoder::

            attn = FullAttention(mask_flag=True, output_attention=True)
            out, weights = attn(q, k, v, mask)
"""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute full scaled dot-product attention over the entire sequence.

        Calculates attention scores across all query-key pairs, optionally
        applies a causal mask, normalises with softmax, and aggregates value
        vectors into the output. Complexity is O(L * S) in both time and
        memory.

        Args:
            queries: Query tensor of shape ``[B, L, H, E]`` where ``B`` is
                    batch size, ``L`` query length, ``H`` heads, ``E``
                    head dimension.
            keys: Key tensor of shape ``[B, S, H, E]`` where ``S`` is the
                key sequence length.
            values: Value tensor of shape ``[B, S, H, D]`` where ``D`` is
                    the value head dimension.
            attn_mask: Optional :class:`TriangularCausalMask` instance. If
                    ``mask_flag`` is ``True`` and this is ``None``, a
                    causal mask is created automatically from ``B`` and
                    ``L``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - **V** - Attention output of shape ``[B, L, H, D]``.
                - **A** - Attention weight matrix of shape
                ``[B, H, L, S]`` if ``output_attention=True``,
                otherwise ``None``.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    """probsparse mechanism from informer model. it only calculates part for matrix score. """
    """Args:
        Q: Query tensor of shape ``[B, H, L_Q, E]``.
        K: Key tensor of shape ``[B, H, L_K, E]``.
        sample_k: Number of keys randomly sampled per query for the cheap
                sparsity measurement; typically ``factor * ln(L_K)``.
        n_top: Number of top queries to select and compute fully;
            typically ``factor * ln(L_Q)``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - **Q_K** - Full attention scores for the selected queries,
            shape ``[B, H, n_top, L_K]``.
            - **M_top** - Indices of the selected top queries within
            ``L_Q``, shape ``[B, H, n_top]``; used downstream to place
            sparse outputs back into the full sequence and to build
            :class:`ProbMask`."""
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top)->Tuple[torch.Tensor, torch.Tensor]: # n_top: c*ln(L_q)
        """Select the most informative queries and compute their full attention scores.

        Implements the ProbSparse query selection procedure. Each query is first
        scored cheaply against a random subset of ``sample_k`` keys using a
        sparsity measure ``M``, which approximates how peaked (informative) the
        query's attention distribution would be. The top ``n_top`` queries by
        this measure are then used to compute exact dot-product scores against
        all keys, discarding the remaining uninformative queries entirely.

        Args:
            Q: Query tensor of shape ``[B, H, L_Q, E]``.
            K: Key tensor of shape ``[B, H, L_K, E]``.
            sample_k: Number of keys randomly sampled per query for the cheap
                    sparsity measurement; typically ``factor * ln(L_K)``.
            n_top: Number of top queries to select and compute fully;
                typically ``factor * ln(L_Q)``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - **Q_K** - Full attention scores for the selected queries,
                shape ``[B, H, n_top, L_K]``.
                - **M_top** - Indices of the selected top queries within
                ``L_Q``, shape ``[B, H, n_top]``; used downstream to place
                sparse outputs back into the full sequence and to build
                :class:`ProbMask`.

        Note:
            The sparsity measure for query ``i`` is defined as
            ``M(q_i) = max_j(q_i·k_j) - (1/L_K) Σ_j(q_i·k_j)``.
            A high value indicates a peaked distribution (informative);
            a low value indicates a near-uniform distribution (uninformative).
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q)->torch.tensor:
        """Initialise the context tensor before sparse attention updates.

        Fills every query slot with a default aggregation of the value
        sequence. Uninformative (non-selected) queries are never updated
        by :meth:`_update_context`, so this default acts as their final
        output.

        Two regimes depending on ``mask_flag``:

        - **No mask** (encoder): each slot is pre-filled with the mean of
        all value vectors — a uniform-attention baseline.
        - **With mask** (decoder / self-attention): each slot is filled
        with the cumulative sum of values up to that position, which
        approximates causal uniform attention and requires ``L_Q == L_V``.

        Args:
            V: Value tensor of shape ``[B, H, L_V, D]``.
            L_Q: Query sequence length; used to expand the context to the
                correct number of query slots.

        Returns:
            torch.Tensor: Initialised context of shape ``[B, H, L_Q, D]``,
            ready to be selectively overwritten by top-u query results.
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Overwrite selected query slots with true sparse attention outputs.

        For the top-u queries identified by :meth:`_prob_QK`, applies causal
        masking (if enabled), computes softmax attention weights, and writes
        the resulting weighted value aggregation back into the corresponding
        positions of the pre-initialised context tensor. All other positions
        retain the default values set by :meth:`_get_initial_context`.

        When ``output_attention=True``, constructs a full ``[B, H, L_V, L_V]``
        attention matrix initialised to uniform weights ``1/L_V`` and
        overwrites only the selected query rows with their true attention
        distributions.

        Args:
            context_in: Pre-initialised context tensor of shape
                        ``[B, H, L_Q, D]`` from :meth:`_get_initial_context`;
                        updated in-place at the selected query indices.
            V: Value tensor of shape ``[B, H, L_V, D]``.
            scores: Sparse attention scores of shape ``[B, H, n_top, L_K]``
                    for the selected queries only.
            index: LongTensor of shape ``[B, H, n_top]`` with the query
                positions corresponding to ``scores``, as returned by
                :meth:`_prob_QK`.
            L_Q: Full query sequence length, used when building
                :class:`ProbMask` and the output attention matrix.
            attn_mask: Unused external mask argument; internally replaced by
                    a :class:`ProbMask` when ``mask_flag=True``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - **context_in** – Updated context of shape
                ``[B, H, L_Q, D]`` with top-u slots overwritten.
                - **attns** – Full attention matrix of shape
                ``[B, H, L_V, L_V]`` if ``output_attention=True``,
                otherwise ``None``.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        """Run the full ProbSparse self-attention forward pass.

        Orchestrates the three-stage sparse attention pipeline:

        1. **Sample & select** — :meth:`_prob_QK` samples a random key
        subset per query, scores each query with the sparsity measure M,
        and returns full scores and indices for the top
        ``u = factor * ln(L_Q)`` queries.
        2. **Initialise context** — :meth:`_get_initial_context` fills all
        ``L_Q`` output slots with a default value aggregation so
        non-selected queries have a meaningful fallback output.
        3. **Update context** — :meth:`_update_context` applies masking and
        softmax to the sparse scores, then writes the true attention
        outputs into the selected query slots.

        Args:
            queries: Query tensor of shape ``[B, L_Q, H, D]``.
            keys: Key tensor of shape ``[B, L_K, H, D]``.
            values: Value tensor of shape ``[B, L_K, H, D]``.
            attn_mask: Passed to :meth:`_update_context`; superseded by an
                    auto-generated :class:`ProbMask` when
                    ``mask_flag=True``.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - **context** - Attention output of shape
                ``[B, L_Q, H, D]``.
                - **attn** - Full attention weight matrix of shape
                ``[B, H, L_K, L_K]`` if ``output_attention=True``,
                otherwise ``None``.

        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    """calculate multi head attention for input"""
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask)->Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Project inputs into multi-head space, apply attention, and project back.

        Performs the full multi-head attention pipeline: linear projection of
        queries, keys, and values into per-head subspaces, delegation to the
        inner attention module (either :class:`FullAttention` or
        :class:`ProbAttention`), and a final linear projection that merges all
        heads back to ``d_model``.

        Args:
            queries: Query tensor of shape ``[B, L, d_model]`` where ``B``
                    is batch size and ``L`` is the query sequence length.
            keys: Key tensor of shape ``[B, S, d_model]`` where ``S`` is
                the key/value sequence length.
            values: Value tensor of shape ``[B, S, d_model]``.
            attn_mask: Boolean mask forwarded unchanged to the inner
                    attention module; see :class:`TriangularCausalMask`
                    and :class:`ProbMask` for details.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - **out** - Attention output projected back to model
                dimension, shape ``[B, L, d_model]``.
                - **attn** - Attention weight matrix returned by the inner
                attention module; ``None`` if ``output_attention=False``
                was set on the inner module.

        Note:
            When ``mix=True`` the head and sequence axes of the inner
            attention output are transposed before the heads are merged.
            This is used in Informer's decoder to mix information across
            heads before the final projection.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    
class PositionalEmbedding(nn.Module):
    """implement positional embedding to show order of input"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x)->torch.Tensor:
        """Retrieve positional encodings for the input sequence length.

        Slices the pre-computed sinusoidal encoding table to match the
        sequence length of ``x``, returning a fixed positional signal that
        is added to the token embeddings by the caller. The encodings are
        not learned and require no gradient computation.

        Args:
            x: Input tensor of shape ``[B, L, d_model]``; only the sequence
            length ``L`` (``x.size(1)``) is used for slicing — the
            values of ``x`` itself are not read.

        Returns:
            torch.Tensor: Positional encoding slice of shape
            ``[1, L, d_model]``, broadcastable across the batch dimension.

        Note:
            The returned tensor is a buffer slice, not a copy — it shares
            memory with the pre-computed table. The caller is responsible
            for adding it to the token embeddings::

                x = token_embedding(x) + positional_embedding(x)
        """
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    """project raw input into d_model dimension"""
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x)->torch.Tensor:
        """Project raw input features into the model embedding space.

        Applies a circular-padded 1D convolution across the time axis to
        embed each timestep in context with its immediate neighbours,
        producing a ``d_model``-dimensional representation for every
        position in the sequence.

        Args:
            x: Raw input tensor of shape ``[B, L, c_in]`` where ``B`` is
            batch size, ``L`` is sequence length, and ``c_in`` is the
            number of input feature channels.

        Returns:
            torch.Tensor: Embedded sequence of shape ``[B, L, d_model]``,
            ready to be summed with positional and time-stamp embeddings
            inside :class:`DataEmbedding`.

        Note:
            The input is permuted to ``[B, c_in, L]`` before the
            convolution (which expects channels-first format) and
            transposed back to ``[B, L, d_model]`` afterward. Circular
            padding ensures the output length equals ``L`` regardless of
            kernel size, with no zero-padding artifacts at sequence
            boundaries.
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    """
    Initializes the FixedEmbedding layer with sinusoidal positional encodings.
    Args:
        c_in (int): The size of the input vocabulary (number of positions).
        d_model (int): The dimension of the embedding vectors.
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x)->torch.Tensor:
        """
        Forward pass to get the fixed positional embeddings for input indices.

        Args:
            x (torch.LongTensor): Input tensor of shape (batch_size, sequence_length)
                                  containing indices to embed.

        Returns:
            torch.Tensor: The embedded representations with shape
                          (batch_size, sequence_length, d_model).
                          Detaches the tensor from the computation graph.
        """
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    """
    Initializes the TemporalEmbedding layer, which combines multiple temporal features
    into a single embedding representation.

    Args:
        d_model (int): The dimension of the embedding vectors.
        embed_type (str): Type of embedding to use ('fixed' for sinusoidal, 
                            otherwise learnable embedding). Defaults to 'fixed'.
        freq (str): Frequency type; 't' indicates minute-level feature, 
                    otherwise hour-level features. Defaults to 'h'.

    Attributes:
        minute_embed (nn.Module or None): Embedding for minutes, if freq=='t'.
        hour_embed (nn.Module): Embedding for hours.
        weekday_embed (nn.Module): Embedding for weekdays.
        day_embed (nn.Module): Embedding for days.
        month_embed (nn.Module): Embedding for months.
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        """
        Forward pass to obtain combined temporal embeddings from input features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim),
                              where feature_dim contains temporal features in specific order:
                              [month, day, weekday, hour, minute (optional)].

        Returns:
            torch.Tensor: The summed temporal embeddings with shape 
                          (batch_size, sequence_length, d_model).
        """
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    """
    Initializes the TimeFeatureEmbedding layer, which projects time features into
    an embedding space using a linear transformation.

    Args:
        d_model (int): The dimension of the output embedding vectors.
        embed_type (str): Type of embedding (not used in this implementation but kept for consistency). Defaults to 'timeF'.
        freq (str): Frequency/type of time features, determines input feature size. 
                    Possible values: 'h', 't', 's', 'm', 'a', 'w', 'd', 'b'.
                    Defaults to 'h'.

    Attributes:
        embed (nn.Linear): Linear layer to project input features to embedding space.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x)->torch.Tensor:
        """
        Forward pass to project input time features into embedding space.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_inp), where d_inp 
                              depends on the frequency type.

        Returns:
            torch.Tensor: Embedded representations with shape (..., d_model).
        """
        return self.embed(x)

class DataEmbedding(nn.Module):
    """
    Initializes the DataEmbedding layer, which combines value, positional, and temporal embeddings
    into a single representation with dropout regularization.

    Args:
        c_in (int): Number of input features (e.g., input channels or feature dimensions).
        d_model (int): Dimension of the embedding vectors.
        embed_type (str): Type of temporal embedding ('fixed', 'timeF', etc.). Defaults to 'fixed'.
        freq (str): Frequency of temporal features ('h', 't', 's', etc.). Defaults to 'h'.
        dropout (float): Dropout probability for regularization. Defaults to 0.1.

    Attributes:
        value_embedding (TokenEmbedding): Embedding for input values.
        position_embedding (PositionalEmbedding): Positional encoding.
        temporal_embedding (TemporalEmbedding or TimeFeatureEmbedding): Temporal feature embedding.
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, c_in:int, d_model:int, embed_type:str='fixed', freq:str='h', dropout:float=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x:torch.Tensor, x_mark:torch.Tensor)->torch.Tensor:
        """
        Forward pass to combine different embeddings and apply dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, c_in),
                              representing the raw input features.
            x_mark (torch.Tensor): Temporal features tensor of shape (batch_size, sequence_length, feature_dim).

        Returns:
            torch.Tensor: The combined embedding tensor with shape (batch_size, sequence_length, d_model),
                          after applying value, positional, and temporal embeddings, followed by dropout.
        """
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)
    
class ConvLayer(nn.Module):
    """
    Initializes a convolutional layer with normalization, activation, and pooling,
    typically used for sequence feature extraction in models like Informer.

    Args:
        c_in (int): Number of input channels/features.

    Attributes:
        downConv (nn.Conv1d): 1D convolution layer with circular padding.
        norm (nn.BatchNorm1d): Batch normalization layer.
        activation (nn.ELU): ELU activation function.
        maxPool (nn.MaxPool1d): Max pooling layer.
    """
    def __init__(self, c_in:int):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass through the convolutional layer with normalization, activation, and pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, c_in).

        Returns:
            torch.Tensor: Output tensor after convolution, normalization, activation, and pooling,
                          with shape (batch_size, new_sequence_length, c_in).
        """
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    """
    Initializes an encoder layer with attention, convolutional feed-forward network,
    normalization, dropout, and activation functions, similar to Transformer encoder blocks.

    Args:
        attention (nn.Module): Attention module (e.g., MultiHeadAttention).
        d_model (int): Dimensionality of input embeddings.
        d_ff (int, optional): Dimensionality of the feed-forward layer. Defaults to 4 * d_model.
        dropout (float): Dropout rate for regularization. Defaults to 0.1.
        activation (str): Activation function to use ('relu' or 'gelu'). Defaults to "relu".

    Attributes:
        attention (nn.Module): The attention mechanism.
        conv1 (nn.Conv1d): 1D convolutional layer for the feed-forward network.
        conv2 (nn.Conv1d): 1D convolutional layer for the feed-forward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feed-forward network.
        dropout (nn.Dropout): Dropout layer.
        activation (nn.ReLU or nn.GELU): Activation function.
    """
    def __init__(self, attention, d_model:int, d_ff=None, dropout:float=0.1, activation:str="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x:torch.Tensor, attn_mask=None)->tuple:
        """
        Forward pass through the encoder layer, applying attention, residual connection,
        normalization, convolutional feed-forward network, and dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after processing, shape (batch_size, sequence_length, d_model).
                - torch.Tensor: Attention weights from the attention mechanism.
        """
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn

class Encoder(nn.Module):
    """
    Initializes the Encoder module, which stacks multiple attention and optional convolutional layers.

    Args:
        attn_layers (list): List of attention layer modules (e.g., EncoderLayer instances).
        conv_layers (list, optional): List of convolutional layers corresponding to attention layers. Defaults to None.
        norm_layer (nn.Module, optional): Normalization layer applied at the end. Defaults to None.

    Attributes:
        attn_layers (nn.ModuleList): List of attention layers.
        conv_layers (nn.ModuleList or None): List of convolutional layers or None.
        norm (nn.Module or None): Normalization layer.
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x:torch.Tensor, attn_mask:torch.Tensor=None)->tuple:
        """
        Forward pass through the encoder, applying attention and convolutional layers sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor after passing through all layers, shape (batch_size, sequence_length, feature_dim).
                - list: List of attention weights from each attention layer.
        """
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    """
    Initializes the EncoderStack, which stacks multiple encoder modules applied to different input segments.

    Args:
        encoders (list): List of encoder modules (e.g., Encoder instances).
        inp_lens (list): List of integers indicating the input length scaling factors for each encoder.
                            Typically, these relate to different resolutions or downsampling levels.

    Attributes:
        encoders (nn.ModuleList): List of encoder modules.
        inp_lens (list): List of length scaling factors for input segmentation.
    """
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x:torch.Tensor, attn_mask:torch.Tensor=None)->tuple:
        """
        Forward pass through the encoder stack, processing different segments of input with different encoders.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Concatenated output from all encoders, shape (batch_size, total_concat_length, feature_dim).
                - list: List of attention weights from each encoder.
        """
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
    
class DecoderLayer(nn.Module):
    """
    Initializes a decoder layer with self-attention, cross-attention, feed-forward network,
    normalization, dropout, and activation functions, similar to Transformer decoder blocks.

    Args:
        self_attention (nn.Module): Self-attention module (e.g., MultiHeadAttention).
        cross_attention (nn.Module): Cross-attention module (e.g., MultiHeadAttention).
        d_model (int): Dimensionality of input embeddings.
        d_ff (int, optional): Dimensionality of the feed-forward layer. Defaults to 4 * d_model.
        dropout (float): Dropout rate for regularization. Defaults to 0.1.
        activation (str): Activation function to use ('relu' or 'gelu'). Defaults to "relu".

    Attributes:
        self_attention (nn.Module): Self-attention mechanism.
        cross_attention (nn.Module): Cross-attention mechanism.
        conv1 (nn.Conv1d): First convolutional layer in feed-forward network.
        conv2 (nn.Conv1d): Second convolutional layer in feed-forward network.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        norm3 (nn.LayerNorm): Layer normalization after feed-forward network.
        dropout (nn.Dropout): Dropout layer.
        activation (nn.ReLU or nn.GELU): Activation function.
    """
    def __init__(self, self_attention, cross_attention, d_model:int, d_ff:int=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x:torch.Tensor, cross:torch.Tensor, x_mask:torch.Tensor=None, cross_mask:torch.Tensor=None):
        """
        Forward pass through the decoder layer, applying self-attention, cross-attention,
        residual connections, normalization, convolutional feed-forward network, and dropout.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).
            cross (torch.Tensor): Encoder output to attend to in cross-attention.
            x_mask (torch.Tensor, optional): Self-attention mask. Defaults to None.
            cross_mask (torch.Tensor, optional): Cross-attention mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after processing, shape (batch_size, sequence_length, d_model).
        """
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    """
    Initializes the Decoder, which stacks multiple decoder layers and optionally applies normalization.

    Args:
        layers (list): List of decoder layer modules (e.g., DecoderLayer instances).
        norm_layer (nn.Module, optional): Normalization layer to be applied after all decoder layers. Defaults to None.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        norm (nn.Module or None): Optional normalization layer.
    """
    def __init__(self, layers:list, norm_layer:nn.Module=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x:torch.Tensor, cross:torch.Tensor, x_mask:torch.Tensor=None, cross_mask:torch.Tensor=None)->torch.Tensor:
        """
        Forward pass through the decoder, applying stacked decoder layers sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, feature_dim).
            cross (torch.Tensor): Encoder output tensor for cross-attention.
            x_mask (torch.Tensor, optional): Self-attention mask. Defaults to None.
            cross_mask (torch.Tensor, optional): Cross-attention mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after processing through all decoder layers, shape (batch_size, sequence_length, feature_dim).
        """
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x

class Informer(BaseModel):
    """Informer model for long-sequence time-series forecasting on EHR data.

    Paper: Haoyi Zhou et al. Informer: Beyond Efficient Transformer for
    Long Sequence Time-Series Forecasting. AAAI 2021.

    This model adapts the original Informer architecture to the PyHealth
    ``BaseModel`` interface, supporting binary, multiclass, and multilabel
    classification tasks over EHR time-series features. The core architecture
    is preserved unchanged: a ProbSparse encoder with optional distillation
    and a full-attention decoder that autoregressively produces ``out_len``
    future steps, with the final prediction taken from the last ``pred_len``
    decoder tokens.

    Args:
        dataset: A ``SampleEHRDataset`` instance used to infer token
                 vocabularies and task metadata.
        feature_keys: multivariate or univariate prediction
        label_key: Name of the label field in the dataset samples.
        mode: Task type — one of ``"binary"``, ``"multiclass"``, or
              ``"multilabel"``.
        enc_in: Number of encoder input features (raw channel count).
        dec_in: Number of decoder input features (raw channel count).
        c_out: Number of output channels/classes.
        seq_len: Encoder input sequence length.
        label_len: Decoder start token length (overlap with encoder).
        out_len: Total decoder output sequence length.
        factor: ProbSparse attention sampling factor ``c``; controls
                ``sample_k = factor * ln(L)``. Default ``5``.
        d_model: Model embedding dimension. Default ``512``.
        n_heads: Number of attention heads. Default ``8``.
        e_layers: Number of encoder layers. Default ``3``.
        d_layers: Number of decoder layers. Default ``2``.
        d_ff: Feed-forward inner dimension. Default ``512``.
        dropout: Dropout rate applied throughout the model. Default ``0.0``.
        attn: Attention type — ``"prob"`` for ProbSparse or ``"full"`` for
              standard full attention. Default ``"prob"``.
        embed: Time feature embedding type — ``"fixed"`` or ``"timeF"``.
               Default ``"fixed"``.
        freq: Time frequency string used by ``timeF`` embedding (e.g.
              ``"h"`` for hourly). Default ``"h"``.
        activation: Activation function name for feed-forward layers.
                    Default ``"gelu"``.
        output_attention: If ``True``, ``forward`` additionally returns
                          encoder attention weights. Default ``False``.
        distil: If ``True``, applies convolutional distillation between
                encoder layers to halve the sequence length progressively.
                Default ``True``.
        mix: If ``True``, enables head-mixing in the decoder cross-attention
             layer. Default ``True``.
        device: Target device for mask creation. Default ``cuda:0``.

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> dataset = SampleDataset(samples=[...], dataset_name="test")
        >>> model = Informer(
        ...     dataset=dataset,
        ...     feature_keys=["timeseries"],
        ...     label_key="label",
        ...     mode="binary",
        ...     enc_in=7,
        ...     dec_in=7,
        ...     c_out=1,
        ...     seq_len=96,
        ...     label_len=48,
        ...     out_len=24,
        ... )
        >>> data_batch = next(iter(train_loader))
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...),
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        enc_in: int,
        dec_in: int,
        c_out: int,
        seq_len: int,
        label_len: int,
        out_len: int,
        factor: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.0,
        attn: str = "prob",
        embed: str = "fixed",
        freq: str = "h",
        activation: str = "gelu",
        output_attention: bool = False,
        distil: bool = True,
        mix: bool = True,
        device: torch.device = torch.device("cuda:0"),
    ):
        # Initialise BaseModel with dataset metadata and task settings.
        super(Informer, self).__init__(
            dataset=dataset,
        )

        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # --- Embedding layers (unchanged from original) ---
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # --- Attention selection ---
        Attn = ProbAttention if attn == "prob" else FullAttention

        # --- Encoder ---
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model),
        )

        # --- Decoder ---
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # --- Output projection: d_model → c_out, then → task output size ---
        self.projection = nn.Linear(d_model, c_out, bias=True)

        # --- PyHealth task head: maps c_out * pred_len → num_labels ---
        # BaseModel exposes self.get_output_size() for the number of classes.
        self.task_head = nn.Linear(c_out * out_len, self.get_output_size())

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run the Informer forward pass and return PyHealth-compatible outputs.

        Preserves the original three-stage pipeline — encoder embedding,
        encoder stack, decoder — and appends a linear task head that maps
        the ``pred_len`` forecast horizon into the class space required by
        the downstream PyHealth task. Loss, probability, and label tensors
        are computed via ``BaseModel`` helpers so that the output dict is
        compatible with ``pyhealth.trainer.Trainer``.

        Args:
            x_enc: Encoder input of shape ``[B, seq_len, enc_in]``.
            x_mark_enc: Encoder time-stamp features of shape
                        ``[B, seq_len, time_features]``.
            x_dec: Decoder input of shape ``[B, label_len + out_len, dec_in]``.
            x_mark_dec: Decoder time-stamp features of shape
                        ``[B, label_len + out_len, time_features]``.
            enc_self_mask: Optional mask for encoder self-attention.
            dec_self_mask: Optional mask for decoder self-attention.
            dec_enc_mask: Optional mask for decoder cross-attention.
            **kwargs: Absorbs extra keyword arguments passed by the PyHealth
                      dataloader (e.g. ``patient_id``, ``visit_id``).

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys:
                - **loss** - Scalar task loss computed by ``BaseModel``.
                - **y_prob** - Predicted class probabilities.
                - **y_true** - Ground-truth label tensor.
                - **logit** - Raw logits before activation.

        Note:
            The label tensor is extracted from ``kwargs[self.label_key]``
            by ``BaseModel.prepare_labels()``, matching the standard
            PyHealth convention. The ``**kwargs`` argument is therefore
            required to contain the label key at inference time.
        """
        # --- Original Informer forward (unchanged) ---
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)              # [B, label_len+out_len, c_out]
        dec_out = dec_out[:, -self.pred_len:, :]        # [B, pred_len, c_out]

        # --- PyHealth task head ---
        # Flatten forecast horizon into a single feature vector, then project
        # to the number of output classes required by the task.
        logit = self.task_head(dec_out.reshape(dec_out.size(0), -1))  # [B, num_classes]

        # Delegate loss computation, probability calibration, and label
        # extraction to BaseModel using the standard PyHealth interface.
        label_key = self.label_keys[0]
        y_true = kwargs[label_key]
        loss = self.get_loss_function()(logit, y_true)
        y_prob = self.prepare_y_prob(logit)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
        }
        
if __name__ == "__main__":
    import numpy as np
    import torch
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    # ------------------------------------------------------------------ #
    # 1. Simulate ETT-style time-series samples                           #
    #    Each sample mirrors one window produced by Dataset_ETT_hour:     #
    #      x_enc  : [seq_len,   enc_in]  — encoder input                 #
    #      x_mark_enc: [seq_len,   4]    — hourly time features           #
    #      x_dec  : [label_len+pred_len, dec_in] — decoder input         #
    #      x_mark_dec: [label_len+pred_len, 4]   — decoder time features  #
    #      label  : regression target, mean of the pred window            #
    # ------------------------------------------------------------------ #
    SEQ_LEN   = 96
    LABEL_LEN = 48
    PRED_LEN  = 24
    ENC_IN    = 7      # number of encoder input channels (ETTh1 has 7)
    DEC_IN    = 7      # number of decoder input channels
    C_OUT     = 7      # intermediate output channels
    TIME_FEAT = 4      # hourly time encoding: [month, day, weekday, hour]
    N_SAMPLES = 20     # small synthetic dataset for demonstration

    def make_sample(i):
        """Generate one synthetic ETT-style window."""
        # Generate timestamps starting from a base time at hourly intervals
        base_time = datetime(2021, 1, 1)

        enc_timestamps  = [base_time + timedelta(hours=t) for t in range(SEQ_LEN)]
        dec_timestamps  = [base_time + timedelta(hours=t) for t in range(LABEL_LEN + PRED_LEN)]

        x_enc       = np.random.randn(SEQ_LEN,              ENC_IN).astype(np.float32)
        x_mark_enc  = np.random.randn(SEQ_LEN,              TIME_FEAT).astype(np.float32)
        x_dec       = np.random.randn(LABEL_LEN + PRED_LEN, DEC_IN).astype(np.float32)
        x_mark_dec  = np.random.randn(LABEL_LEN + PRED_LEN, TIME_FEAT).astype(np.float32)

        label = float(x_dec[LABEL_LEN:].mean())
    
        return {
            "patient_id": f"patient-{i}",
            "visit_id":   f"visit-{i}",
            # Each timeseries field must be (List[datetime], np.ndarray)
            "x_enc":      (enc_timestamps, x_enc),
            "x_mark_enc": (enc_timestamps, x_mark_enc),
            "x_dec":      (dec_timestamps, x_dec),
            "x_mark_dec": (dec_timestamps, x_mark_dec),
            "label":      label,
        }

    samples = [make_sample(i) for i in range(N_SAMPLES)]

    # ------------------------------------------------------------------ #
    # 2. Build PyHealth SampleDataset                                     #
    #    "timeseries" processor handles 2-D float arrays [T, C].         #
    #    "regression" processor handles scalar float labels.              #
    # ------------------------------------------------------------------ #
    input_schema = {
        "x_enc":      "timeseries",
        "x_mark_enc": "timeseries",
        "x_dec":      "timeseries",
        "x_mark_dec": "timeseries",
    }
    output_schema = {"label": "regression"}

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="ETT_synthetic",
    )

    train_loader = get_dataloader(dataset, batch_size=4, shuffle=True)

    # ------------------------------------------------------------------ #
    # 3. Build the Informer model                                         #
    # ------------------------------------------------------------------ #
    model = Informer(
        dataset=dataset,
        enc_in=ENC_IN,
        dec_in=DEC_IN,
        c_out=C_OUT,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
    )

    # ------------------------------------------------------------------ #
    # 4. One forward pass                                                 #
    # ------------------------------------------------------------------ #
    data_batch = next(iter(train_loader))

    result = model(**data_batch)
    print(result)

    # ------------------------------------------------------------------ #
    # 5. Backward pass to verify gradients flow correctly                 #
    # ------------------------------------------------------------------ #
    result["loss"].backward()
    print("Backward pass OK — loss:", result["loss"].item())