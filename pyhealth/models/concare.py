"""ConCare Model for Personalized Clinical Feature Embedding.

Author: Joshua Steier
Paper: Concare: Personalized clinical feature embedding via capturing the
    healthcare context.
Paper Link: https://ojs.aaai.org/index.php/AAAI/article/view/5428

Description:
    This module implements the ConCare model which uses channel-wise GRUs and
    multi-head self-attention to capture feature correlations and temporal
    patterns in Electronic Health Records (EHR) data. The model learns
    personalized clinical feature embeddings by capturing healthcare context
    through a combination of:
    - Channel-wise GRU layers for temporal modeling of each feature
    - Multi-head self-attention for capturing feature interactions
    - DeCov regularization loss to reduce feature redundancy
    - Final attention mechanism for patient-level representation

    This implementation has been updated from PyHealth 1.0 to PyHealth 2.0 API,
    using the new SampleDataset and EmbeddingModel interfaces.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit

from .embedding import EmbeddingModel


class FinalAttentionQKV(nn.Module):
    """Final attention layer using Query-Key-Value mechanism.

    This layer computes attention weights over the input sequence and produces
    a weighted sum as the output representation.

    Args:
        attention_input_dim: Dimension of the input features.
        attention_hidden_dim: Dimension of the hidden attention space.
        attention_type: Type of attention mechanism. One of "add", "mul",
            or "concat". Default is "add".
        dropout: Dropout rate for attention weights. Default is 0.5.

    Examples:
        >>> layer = FinalAttentionQKV(128, 64, attention_type="mul")
        >>> input_tensor = torch.randn(32, 10, 128)  # [batch, seq, features]
        >>> output, attention_weights = layer(input_tensor)
        >>> output.shape
        torch.Size([32, 64])
    """

    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        dropout: float = 0.5,
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1,))
        self.b_out = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(
            torch.randn(2 * attention_input_dim, attention_hidden_dim)
        )
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the final attention layer.

        Args:
            input: Input tensor of shape [batch_size, time_step, input_dim].

        Returns:
            Tuple containing:
                - output: Weighted representation of shape [batch_size, hidden_dim].
                - attention_weights: Attention weights of shape [batch_size, time_step].
        """
        batch_size, time_step, input_dim = input.size()
        input_q = self.W_q(input[:, -1, :])  # [batch, hidden]
        input_k = self.W_k(input)  # [batch, time, hidden]
        input_v = self.W_v(input)  # [batch, time, hidden]

        if self.attention_type == "add":
            q = torch.reshape(
                input_q, (batch_size, 1, self.attention_hidden_dim)
            )  # [batch, 1, hidden]
            h = q + input_k + self.b_in  # [batch, time, hidden]
            h = self.tanh(h)
            e = self.W_out(h)  # [batch, time, 1]
            e = torch.reshape(e, (batch_size, time_step))  # [batch, time]

        elif self.attention_type == "mul":
            q = torch.reshape(
                input_q, (batch_size, self.attention_hidden_dim, 1)
            )  # [batch, hidden, 1]
            e = torch.matmul(input_k, q).squeeze()  # [batch, time]

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # [batch, time, hidden]
            k = input_k
            c = torch.cat((q, k), dim=-1)  # [batch, time, 2*hidden]
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # [batch, time, 1]
            e = torch.reshape(e, (batch_size, time_step))  # [batch, time]
        else:
            raise ValueError(
                f"Unknown attention type: {self.attention_type}, "
                "please use 'add', 'mul', or 'concat'"
            )

        a = self.softmax(e)  # [batch, time]
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # [batch, hidden]

        return v, a


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network.

    Implements the FFN equation: FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: Dimension of the model (input and output).
        d_ff: Dimension of the feed-forward hidden layer.
        dropout: Dropout rate. Default is 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """Forward pass of the feed-forward network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Tuple containing:
                - output: Output tensor of shape [batch_size, seq_len, d_model].
                - None: Placeholder for compatibility with SublayerConnection.
        """
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model: Dimension of the model.
        dropout: Dropout rate. Default is 0.
        max_len: Maximum sequence length. Default is 400.
    """

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Output tensor with positional encoding added.
        """
        pos = self.pe[:, : x.size(1)].clone().requires_grad_(False)
        x = x + pos
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention mechanism with DeCov regularization.

    Implements multi-head attention as described in "Attention Is All You Need"
    with additional DeCov loss for reducing feature redundancy.

    Args:
        h: Number of attention heads.
        d_model: Dimension of the model. Must be divisible by h.
        dropout: Dropout rate. Default is 0.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by number of heads"
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, self.d_k * self.h) for _ in range(3)]
        )
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Scaled Dot Product Attention.

        Args:
            query: Query tensor of shape [batch, heads, seq_len, d_k].
            key: Key tensor of shape [batch, heads, seq_len, d_k].
            value: Value tensor of shape [batch, heads, seq_len, d_k].
            mask: Optional attention mask.
            dropout: Optional dropout layer.

        Returns:
            Tuple containing:
                - output: Attention output of shape [batch, heads, seq_len, d_k].
                - p_attn: Attention weights.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def cov(self, m: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute covariance matrix.

        Args:
            m: Input tensor.
            y: Optional second tensor to concatenate.

        Returns:
            Covariance matrix.
        """
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of multi-head attention with DeCov loss.

        Args:
            query: Query tensor of shape [batch, seq_len, d_model].
            key: Key tensor of shape [batch, seq_len, d_model].
            value: Value tensor of shape [batch, seq_len, d_model].
            mask: Optional attention mask.

        Returns:
            Tuple containing:
                - output: Attention output of shape [batch, seq_len, d_model].
                - decov_loss: DeCov regularization loss.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        feature_dim = query.size(1)

        # Project to multiple heads
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )

        # Compute DeCov loss
        decov_contexts = x.transpose(0, 1).transpose(1, 2)
        covs = self.cov(decov_contexts[0, :, :])
        decov_loss = 0.5 * (
            torch.norm(covs, p="fro") ** 2 - torch.norm(torch.diag(covs)) ** 2
        )
        for i in range(feature_dim - 1):
            covs = self.cov(decov_contexts[i + 1, :, :])
            decov_loss += 0.5 * (
                torch.norm(covs, p="fro") ** 2 - torch.norm(torch.diag(covs)) ** 2
            )

        return self.final_linear(x), decov_loss


class LayerNorm(nn.Module):
    """Layer Normalization.

    Args:
        features: Number of features to normalize.
        eps: Small constant for numerical stability. Default is 1e-7.
    """

    def __init__(self, features: int, eps: float = 1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization.

    Note: For code simplicity, the norm is applied first (pre-norm).

    Args:
        size: Dimension of the layer.
        dropout: Dropout rate.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor, 
        sublayer: Callable[[torch.Tensor], Tuple[torch.Tensor, any]],
     ) -> Tuple[torch.Tensor, any]:
        
        """Apply residual connection to sublayer with same size.

        Args:
            x: Input tensor.
            sublayer: Sublayer function to apply.

        Returns:
            Tuple containing:
                - output: Output with residual connection.
                - sublayer_output: Additional output from sublayer (e.g., loss).
        """
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class SingleAttention(nn.Module):
    """Single-head attention mechanism with optional time-awareness.

    This layer implements various attention mechanisms for temporal modeling.

    Args:
        attention_input_dim: Dimension of the input features.
        attention_hidden_dim: Dimension of the hidden attention space.
        attention_type: Type of attention. One of "add", "mul", "concat", "new".
            Default is "add".
        time_aware: Whether to use time-aware attention. Default is False.
    """

    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        time_aware: bool = False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        if attention_type == "add":
            if self.time_aware:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
                self.Wtime_aware = nn.Parameter(
                    torch.randn(1, attention_hidden_dim)
                )
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.bh = nn.Parameter(torch.zeros(attention_hidden_dim,))
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))

            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "mul":
            self.Wa = nn.Parameter(
                torch.randn(attention_input_dim, attention_input_dim)
            )
            self.ba = nn.Parameter(torch.zeros(1,))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "concat":
            if self.time_aware:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim + 1, attention_hidden_dim)
                )
            else:
                self.Wh = nn.Parameter(
                    torch.randn(2 * attention_input_dim, attention_hidden_dim)
                )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.Wx = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError(
                "Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'."
            )

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of single attention.

        Args:
            input: Input tensor of shape [batch_size, time_step, hidden_dim].
            mask: Mask tensor of shape [batch_size, time_step].
            device: Device to place tensors on.

        Returns:
            Tuple containing:
                - output: Attention output of shape [batch_size, input_dim].
                - attention_weights: Attention weights of shape [batch_size, time_step].
        """
        batch_size, time_step, input_dim = input.size()

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device=device)
        )
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1

        if self.attention_type == "add":
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))
            if self.time_aware:
                k = torch.matmul(input, self.Wx)
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)
            else:
                k = torch.matmul(input, self.Wx)
            h = q + k + self.bh
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba
            e = torch.reshape(e, (batch_size, time_step))

        elif self.attention_type == "mul":
            last_visit = get_last_visit(input, mask)
            e = torch.matmul(last_visit, self.Wa)
            e = (
                torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).reshape(
                    batch_size, time_step
                )
                + self.ba
            )

        elif self.attention_type == "concat":
            last_visit = get_last_visit(input, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)
            k = input
            c = torch.cat((q, k), dim=-1)
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba
            e = torch.reshape(e, (batch_size, time_step))

        elif self.attention_type == "new":
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))
            k = torch.matmul(input, self.Wx)
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(
                batch_size, time_step
            )
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product)))
                * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / denominator)
        else:
            raise ValueError(
                "Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'."
            )

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)
        v = torch.matmul(a.unsqueeze(1), input).reshape(batch_size, input_dim)

        return v, a


class ConCareLayer(nn.Module):
    """ConCare layer for personalized clinical feature embedding.

    This layer implements the core ConCare architecture with channel-wise GRUs,
    multi-head self-attention, and final attention pooling.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding
        via capturing the healthcare context. AAAI 2020.

    Args:
        input_dim: Dynamic feature size (number of input features).
        static_dim: Static feature size. If 0, no static features used.
            Default is 0.
        hidden_dim: Hidden dimension of channel-wise GRU. Default is 128.
        num_head: Number of attention heads. Default is 4.
        pe_hidden: Hidden dimension of positional encoding FFN. Default is 64.
        dropout: Dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models.concare import ConCareLayer
        >>> layer = ConCareLayer(input_dim=64, hidden_dim=128)
        >>> input_tensor = torch.randn(3, 128, 64)  # [batch, seq_len, features]
        >>> output, decov_loss = layer(input_tensor)
        >>> output.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super(ConCareLayer, self).__init__()

        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        self.dropout = dropout
        self.static_dim = static_dim

        # Layers
        self.PositionalEncoding = PositionalEncoding(
            self.transformer_hidden, dropout=0, max_len=400
        )

        self.GRUs = nn.ModuleList(
            [
                nn.GRU(1, self.hidden_dim, batch_first=True)
                for _ in range(self.input_dim)
            ]
        )
        self.LastStepAttentions = nn.ModuleList(
            [
                SingleAttention(
                    self.hidden_dim,
                    8,
                    attention_type="new",
                    time_aware=True,
                )
                for _ in range(self.input_dim)
            ]
        )

        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type="mul",
            dropout=self.dropout,
        )

        self.MultiHeadedAttention = MultiHeadedAttention(
            self.num_head, self.transformer_hidden, dropout=self.dropout
        )
        self.SublayerConnection = SublayerConnection(
            self.transformer_hidden, dropout=self.dropout
        )

        self.PositionwiseFeedForward = PositionwiseFeedForward(
            self.transformer_hidden, self.pe_hidden, dropout=0.1
        )

        if self.static_dim > 0:
            self.demo_proj_main = nn.Linear(self.static_dim, self.hidden_dim)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def concare_encoder(
        self,
        input: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input through ConCare architecture.

        Args:
            input: Input tensor of shape [batch_size, timestep, feature_dim].
            static: Optional static features of shape [batch_size, static_dim].
            mask: Optional mask of shape [batch_size, timestep].

        Returns:
            Tuple containing:
                - weighted_contexts: Patient embedding of shape [batch_size, hidden_dim].
                - decov_loss: DeCov regularization loss.
        """
        if self.static_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(1)

        batch_size = input.size(0)
        feature_dim = input.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        # Channel-wise GRU encoding
        gru_embedded_input = self.GRUs[0](
            input[:, :, 0].unsqueeze(-1).to(device=input.device),
            torch.zeros(batch_size, self.hidden_dim)
            .to(device=input.device)
            .unsqueeze(0),
        )[0]
        attention_embedded_input = self.LastStepAttentions[0](
            gru_embedded_input, mask, input.device
        )[0].unsqueeze(1)

        for i in range(feature_dim - 1):
            embedded_input = self.GRUs[i + 1](
                input[:, :, i + 1].unsqueeze(-1),
                torch.zeros(batch_size, self.hidden_dim)
                .to(device=input.device)
                .unsqueeze(0),
            )[0]
            embedded_input = self.LastStepAttentions[i + 1](
                embedded_input, mask, input.device
            )[0].unsqueeze(1)
            attention_embedded_input = torch.cat(
                (attention_embedded_input, embedded_input), 1
            )

        if self.static_dim > 0:
            attention_embedded_input = torch.cat(
                (attention_embedded_input, demo_main), 1
            )

        posi_input = self.dropout_layer(attention_embedded_input)

        # Multi-head self-attention
        contexts = self.SublayerConnection(
            posi_input,
            lambda x: self.MultiHeadedAttention(
                posi_input, posi_input, posi_input, None
            ),
        )

        decov_loss = contexts[1]
        contexts = contexts[0]

        # Feed-forward network
        contexts = self.SublayerConnection(
            contexts, lambda x: self.PositionwiseFeedForward(contexts)
        )[0]

        # Final attention pooling
        weighted_contexts, _ = self.FinalAttentionQKV(contexts)
        return weighted_contexts, decov_loss

    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: Input tensor of shape [batch_size, sequence_len, input_dim].
            static: Optional static features of shape [batch_size, static_dim].
            mask: Optional mask of shape [batch_size, sequence_len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            Tuple containing:
                - output: Patient embedding of shape [batch_size, hidden_dim].
                - decov_loss: DeCov regularization loss value.
        """
        out, decov = self.concare_encoder(x, static, mask)
        out = self.dropout_layer(out)
        return out, decov


class ConCare(BaseModel):
    """ConCare model for EHR-based prediction tasks.

    ConCare (Concare: Personalized clinical feature embedding via capturing the
    healthcare context) uses channel-wise GRUs and multi-head self-attention to
    capture feature correlations and temporal patterns in Electronic Health
    Records (EHR) data.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding
        via capturing the healthcare context. AAAI 2020.
    Paper Link: https://ojs.aaai.org/index.php/AAAI/article/view/5428

    Note:
        We use separate ConCare layers for different feature_keys.
        The model automatically handles different input formats through the
        EmbeddingModel.

    Args:
        dataset: The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        static_key: The key in samples to use as static features, e.g.
            "demographics". Default is None. Only numerical static features
            are supported.
        embedding_dim: The embedding dimension. Default is 128.
        hidden_dim: The hidden dimension. Default is 128.
        **kwargs: Other parameters for the ConCare layer (num_head, pe_hidden,
            dropout).

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "list_codes": ["505800458", "50580045810", "50580045811"],
        ...         "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...         "demographic": [0.0, 2.0, 1.5],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-1",
        ...         "list_codes": ["55154191800", "551541928", "55154192800"],
        ...         "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7]],
        ...         "demographic": [0.0, 2.0, 1.5],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = SampleDataset(
        ...     samples=samples,
        ...     input_schema={"list_codes": "sequence", "list_vectors": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>> from pyhealth.models import ConCare
        >>> model = ConCare(
        ...     dataset=dataset,
        ...     static_key="demographic",
        ...     embedding_dim=64,
        ...     hidden_dim=64,
        ... )
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>> ret = model(**data_batch)
        >>> print(ret["loss"])
        tensor(..., grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(ConCare, self).__init__(dataset=dataset)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.static_key = static_key

        # Validate kwargs
        if "input_dim" in kwargs:
            raise ValueError("input_dim is determined by embedding_dim")

        assert len(self.label_keys) == 1, (
            "Only one label key is supported for ConCare"
        )
        self.label_key = self.label_keys[0]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Determine static dimension
        self.static_dim = 0
        if self.static_key is not None:
            first_sample = dataset[0]
            if self.static_key in first_sample:
                static_val = first_sample[self.static_key]
                if isinstance(static_val, torch.Tensor):
                    self.static_dim = (
                        static_val.shape[-1] if static_val.dim() > 0 else 1
                    )
                elif isinstance(static_val, (list, tuple)):
                    self.static_dim = len(static_val)
                else:
                    self.static_dim = 1

        # Get dynamic feature keys (excluding static key)
        self.dynamic_feature_keys = [
            k for k in self.feature_keys
            if k != self.static_key
        ]

        # ConCare layers for each dynamic feature
        self.concare = nn.ModuleDict()
        for feature_key in self.dynamic_feature_keys:
            self.concare[feature_key] = ConCareLayer(
                input_dim=embedding_dim,
                static_dim=self.static_dim,
                hidden_dim=self.hidden_dim,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(
            len(self.dynamic_feature_keys) * self.hidden_dim, output_size
        )
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the final loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                """
        patient_emb = []
        decov_loss = 0

        embedded, masks = self.embedding_model(kwargs, output_mask=True)

        # Get static features if available
        static = None
        if self.static_key is not None and self.static_key in kwargs:
            static_data = kwargs[self.static_key]
            if isinstance(static_data, torch.Tensor):
                static = static_data.float().to(self.device)
            else:
                static = torch.tensor(
                    static_data, dtype=torch.float, device=self.device
                )

        for feature_key in self.dynamic_feature_keys:
            x = embedded[feature_key]
            mask = masks[feature_key]

            x, decov = self.concare[feature_key](x, static=static, mask=mask)
            patient_emb.append(x)
            decov_loss += decov

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        # Compute loss and predictions
        y_true = kwargs[self.label_key].to(self.device)
        loss_task = self.get_loss_function()(logits, y_true)
        loss = decov_loss + loss_task
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "list_codes": ["505800458", "50580045810", "50580045811"],
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
            "label": 1,
            "demographic": [1.0, 2.0, 1.3],
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "label": 0,
            "demographic": [1.0, 2.0, 1.3],
        },
    ]

    # Create dataset
    dataset = SampleDataset(
        samples=samples,
        input_schema={
            "list_codes": "sequence",
            "list_vectors": "sequence",
            "list_list_codes": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    # Create data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # Create model
    model = ConCare(
        dataset=dataset,
        static_key="demographic",
        embedding_dim=64,
        hidden_dim=64,
    )

    # Get data batch
    data_batch = next(iter(train_loader))

    # Forward pass
    ret = model(**data_batch)
    print(ret)

    # Backward pass
    ret["loss"].backward()
