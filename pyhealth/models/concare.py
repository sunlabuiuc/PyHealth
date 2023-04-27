import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit


class FinalAttentionQKV(nn.Module):
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

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(
            torch.randn(2 * attention_input_dim, attention_hidden_dim)
        )
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        (
            batch_size,
            time_step,
            input_dim,
        ) = input.size()  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :])  # b h
        input_k = self.W_k(input)  # b t h
        input_v = self.W_v(input)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I

            q = torch.reshape(
                input_q, (batch_size, 1, self.attention_hidden_dim)
            )  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            q = torch.reshape(
                input_q, (batch_size, self.attention_hidden_dim, 1)
            )  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        else:
            raise ValueError(
                "Unknown attention type: {}, please use add, mul, concat".format(
                    self.attention_type
                )
            )

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


class PositionwiseFeedForward(nn.Module):  # new added
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):  # new added / not use anymore
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].clone().requires_grad_(False)
        x = x + pos
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(
            [nn.Linear(d_model, self.d_k * self.h) for _ in range(3)]
        )
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)  # b h t d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b h t t
        if mask is not None:  # 1 1 t t
            scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
        p_attn = torch.softmax(scores, dim=-1)  # b h t t
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        input_dim = query.size(1)  # i+1
        feature_dim = query.size(1)  # i+1

        # input size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]  # b num_head d_input d_k

        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # b num_head d_input d_v (d_k)

        x = (
            x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        )  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = self.cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (
            torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
        )
        for i in range(feature_dim - 1):
            Covs = self.cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (
                torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2
            )

        return self.final_linear(x), DeCov_loss


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        time_aware=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == "add":
            if self.time_aware:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(
                    torch.randn(attention_input_dim, attention_hidden_dim)
                )
            self.Wt = nn.Parameter(
                torch.randn(attention_input_dim, attention_hidden_dim)
            )
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(
                torch.randn(attention_input_dim, attention_input_dim)
            )
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

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
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

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

    def forward(self, input, mask, device):

        (
            batch_size,
            time_step,
            input_dim,
        ) = input.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(device=device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                k = torch.matmul(input, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(input, self.Wx)  # b t h
            h = q + k + self.bh  # b t h
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            last_visit = get_last_visit(input, mask)
            e = torch.matmul(last_visit, self.Wa)  # b i
            e = (
                torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).reshape(
                    batch_size, time_step
                )
                + self.ba
            )  # b t
        elif self.attention_type == "concat":
            last_visit = get_last_visit(input, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = input
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            last_visit = get_last_visit(input, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(input, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(
                batch_size, time_step
            )  # b t
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product)))
                * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))  # b * t
        else:
            raise ValueError(
                "Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'."
            )

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)  # B*T
        v = torch.matmul(a.unsqueeze(1), input).reshape(batch_size, input_dim)  # B*I

        return v, a


class ConCareLayer(nn.Module):
    """ConCare layer.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding via capturing the healthcare context. AAAI 2020.

    This layer is used in the ConCare model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: dynamic feature size.
        static_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the channel-wise GRU, default 128.
        transformer_hidden: hidden dimension of the transformer, default 128.
        num_head: number of heads in the transformer, default 4.
        pe_hidden: hidden dimension of the positional encoding, default 64.
        dropout: dropout rate, default 0.5.

    Examples:
        >>> from pyhealth.models import ConCareLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = ConCareLayer(64)
        >>> c, _ = layer(input)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.5,
    ):
        super(ConCareLayer, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = dropout
        self.static_dim = static_dim

        # layers
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

        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def concare_encoder(self, input, static=None, mask=None):
        # input shape [batch_size, timestep, feature_dim]

        if self.static_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(
                1
            )  # b hidden_dim

        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        # forward
        GRU_embeded_input = self.GRUs[0](
            input[:, :, 0].unsqueeze(-1).to(device=input.device),
            torch.zeros(batch_size, self.hidden_dim)
            .to(device=input.device)
            .unsqueeze(0),
        )[
            0
        ]  # b t h
        Attention_embeded_input = self.LastStepAttentions[0](
            GRU_embeded_input, mask, input.device
        )[0].unsqueeze(
            1
        )  # b 1 h

        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](
                input[:, :, i + 1].unsqueeze(-1),
                torch.zeros(batch_size, self.hidden_dim)
                .to(device=input.device)
                .unsqueeze(0),
            )[
                0
            ]  # b 1 h
            embeded_input = self.LastStepAttentions[i + 1](
                embeded_input, mask, input.device
            )[0].unsqueeze(
                1
            )  # b 1 h
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, embeded_input), 1
            )  # b i h

        if self.static_dim > 0:
            Attention_embeded_input = torch.cat(
                (Attention_embeded_input, demo_main), 1
            )  # b i+1 h
        posi_input = self.dropout(
            Attention_embeded_input
        )  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(
            posi_input,
            lambda x: self.MultiHeadedAttention(
                posi_input, posi_input, posi_input, None
            ),
        )  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(
            contexts, lambda x: self.PositionwiseFeedForward(contexts)
        )[0]

        weighted_contexts, a = self.FinalAttentionQKV(contexts)
        return weighted_contexts, DeCov_loss

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            static: a tensor of shape [batch size, static_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
            decov: the decov loss value
        """
        # rnn will only apply dropout between layers
        batch_size, time_steps, _ = x.size()
        out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        out, decov = self.concare_encoder(x, static, mask)
        out = self.dropout(out)
        return out, decov


class ConCare(BaseModel):
    """ConCare model.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding via capturing the healthcare context. AAAI 2020.

    Note:
        We use separate ConCare layers for different feature_keys.
        Currently, we automatically support different input formats:
            - code based input (need to use the embedding table later)
            - float/int based value input
        If you need the interpretable feature correlations provided by the ConCare model calculates the , we do not recommend use embeddings for the input features.
        We follow the current convention for the ConCare model:
            - case 1. [code1, code2, code3, ...]
                - we will assume the code follows the order; our model will encode
                each code into a vector and apply ConCare on the code level
            - case 2. [[code1, code2]] or [[code1, code2], [code3, code4, code5], ...]
                - we will assume the inner bracket follows the order; our model first
                use the embedding table to encode each code into a vector and then use
                average/mean pooling to get one vector for one inner bracket; then use
                ConCare one the braket level
            - case 3. [[1.5, 2.0, 0.0]] or [[1.5, 2.0, 0.0], [8, 1.2, 4.5], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run ConCare directly
                on the inner bracket level, similar to case 1 after embedding table
            - case 4. [[[1.5, 2.0, 0.0]]] or [[[1.5, 2.0, 0.0], [8, 1.2, 4.5]], ...]
                - this case only makes sense when each inner bracket has the same length;
                we assume each dimension has the same meaning; we run ConCare directly
                on the inner bracket level, similar to case 2 after embedding table

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        static_keys: the key in samples to use as static features, e.g. "demographics". Default is None.
                     we only support numerical static features.
        use_embedding: list of bools indicating whether to use embedding for each feature type,
            e.g. [True, False].
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the ConCare layer.


    Examples:
        >>> from pyhealth.datasets import SampleEHRDataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
        ...             "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...             "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
        ...             "list_list_vectors": [
        ...                 [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
        ...                 [[7.7, 8.5, 9.4]],
        ...             ],
        ...             "demographic": [0.0, 2.0, 1.5],
        ...             "label": 1,
        ...         },
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-1",
        ...             "list_codes": [
        ...                 "55154191800",
        ...                 "551541928",
        ...                 "55154192800",
        ...                 "705182798",
        ...                 "70518279800",
        ...             ],
        ...             "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
        ...             "list_list_codes": [["A04A", "B035", "C129"]],
        ...             "list_list_vectors": [
        ...                 [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
        ...             ],
        ...             "demographic": [0.0, 2.0, 1.5],
        ...             "label": 0,
        ...         },
        ...     ]
        >>> dataset = SampleEHRDataset(samples=samples, dataset_name="test")
        >>>
        >>> from pyhealth.models import ConCare
        >>> model = ConCare(
        ...         dataset=dataset,
        ...         feature_keys=[
        ...             "list_codes",
        ...             "list_vectors",
        ...             "list_list_codes",
        ...             "list_list_vectors",
        ...         ],
        ...         label_key="label",
        ...         static_key="demographic",
        ...         use_embedding=[True, False, True, False],
        ...         mode="binary"
        ...     )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(9.5541, grad_fn=<AddBackward0>), 
            'y_prob': tensor([[0.5323], [0.5363]], grad_fn=<SigmoidBackward0>), 
            'y_true': tensor([[1.], [0.]]), 
            'logit': tensor([[0.1293], [0.1454]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        use_embedding: List[bool],
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(ConCare, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.use_embedding = use_embedding
        self.hidden_dim = hidden_dim

        # validate kwargs for ConCare layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        # the key of self.feat_tokenizers only contains the code based inputs
        self.feat_tokenizers = {}
        self.static_key = static_key
        self.label_tokenizer = self.get_label_tokenizer()
        # the key of self.embeddings only contains the code based inputs
        self.embeddings = nn.ModuleDict()
        # the key of self.linear_layers only contains the float/int based inputs
        self.linear_layers = nn.ModuleDict()

        self.static_dim = 0
        if self.static_key is not None:
            self.static_dim = self.dataset.input_info[self.static_key]["len"]

        self.concare = nn.ModuleDict()
        # add feature ConCare layers
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            # sanity check
            if input_info["type"] not in [str, float, int]:
                raise ValueError(
                    "ConCare only supports str code, float and int as input types"
                )
            elif (input_info["type"] == str) and (input_info["dim"] not in [2, 3]):
                raise ValueError(
                    "ConCare only supports 2-dim or 3-dim str code as input types"
                )
            elif (input_info["type"] == str) and (use_embedding[idx] == False):
                raise ValueError(
                    "ConCare only supports embedding for str code as input types"
                )
            elif (input_info["type"] in [float, int]) and (
                input_info["dim"] not in [2, 3]
            ):
                raise ValueError(
                    "ConCare only supports 2-dim or 3-dim float and int as input types"
                )

            # for code based input, we need Type
            # for float/int based input, we need Type, input_dim
            if use_embedding[idx]:
                self.add_feature_transform_layer(feature_key, input_info)
                self.concare[feature_key] = ConCareLayer(
                    input_dim=embedding_dim,
                    static_dim=self.static_dim,
                    hidden_dim=self.hidden_dim,
                    **kwargs,
                )
            else:
                self.concare[feature_key] = ConCareLayer(
                    input_dim=input_info["len"],
                    static_dim=self.static_dim,
                    hidden_dim=self.hidden_dim,
                    **kwargs,
                )

        output_size = self.get_output_size(self.label_tokenizer)
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the final loss.
                loss_task: a scalar tensor representing the task loss.
                loss_decov: a scalar tensor representing the decov loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        decov_loss = 0
        for idx, feature_key in enumerate(self.feature_keys):
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_2d(
                    kwargs[feature_key]
                )
                # (patient, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, event)
                mask = torch.any(x !=0, dim=2)

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                x = self.feat_tokenizers[feature_key].batch_encode_3d(
                    kwargs[feature_key]
                )
                # (patient, visit, event)
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (patient, visit, event, embedding_dim)
                x = self.embeddings[feature_key](x)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)
                # (patient, visit)
                mask = torch.any(x !=0, dim=2)

            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                x, mask = self.padding2d(kwargs[feature_key])
                # (patient, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, event, embedding_dim)
                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask.bool().to(self.device)

            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                x, mask = self.padding3d(kwargs[feature_key])
                # (patient, visit, event, values)
                x = torch.tensor(x, dtype=torch.float, device=self.device)
                # (patient, visit, embedding_dim)
                x = torch.sum(x, dim=2)

                if self.use_embedding[idx]:
                    x = self.linear_layers[feature_key](x)
                # (patient, event)
                mask = mask[:, :, 0]
                mask = mask.bool().to(self.device)
            else:
                raise NotImplementedError

            if self.static_dim > 0:
                static = torch.tensor(
                    kwargs[self.static_key], dtype=torch.float, device=self.device
                )
                x, decov = self.concare[feature_key](x, static=static, mask=mask)
            else:
                x, decov = self.concare[feature_key](x, mask=mask)
            patient_emb.append(x)
            decov_loss += decov

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss_task = self.get_loss_function()(logits, y_true)
        loss = decov_loss + loss_task
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            'logit': logits,
        }
        if kwargs.get('embed', False):
            results['embed'] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            # "single_vector": [1, 2, 3],
            "list_codes": ["505800458", "50580045810", "50580045811"],  # NDC
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],  # ATC-4
            "list_list_vectors": [
                [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                [[7.7, 8.5, 9.4]],
            ],
            "label": 1,
            "demographic": [1.0, 2.0, 1.3],
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            # "single_vector": [1, 5, 8],
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "list_list_vectors": [
                [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
            ],
            "label": 0,
            "demographic": [1.0, 2.0, 1.3],
        },
    ]

    # dataset
    dataset = SampleEHRDataset(samples=samples, dataset_name="test")

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = ConCare(
        dataset=dataset,
        feature_keys=[
            "list_codes",
            "list_vectors",
            "list_list_codes",
            # "list_list_vectors",
        ],
        static_key="demographic",
        label_key="label",
        use_embedding=[True, False, True],
        mode="binary",
        hidden_dim=64,
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
