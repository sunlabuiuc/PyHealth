import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import (
    MultiHotProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.models.embedding import EmbeddingModel

# VALID_OPERATION_LEVEL = ["visit", "event"]


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
 
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
    
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

        self.attn_gradients = None
        self.attn_map = None

    # helper functions for interpretability
    def get_attn_map(self):
        return self.attn_map 
    
    def get_attn_grad(self):
        return self.attn_gradients

    def save_attn_grad(self, attn_grad):
        self.attn_gradients = attn_grad 

    # register_hook option allows us to save the gradients in backwarding
    def forward(self, query, key, value, mask=None, register_hook = False):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        self.attn_map = attn # save the attention map
        if register_hook:
            attn.register_hook(self.save_attn_grad)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
  
        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Transformer block.

    MultiHeadedAttention + PositionwiseFeedForward + SublayerConnection

    Args:
        hidden: hidden size of transformer.
        attn_heads: head sizes of multi-head attention.
        dropout: dropout rate.
    """

    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None, register_hook = False):
        """Forward propagation.

        Args:
            x: [batch_size, seq_len, hidden]
            mask: [batch_size, seq_len, seq_len]

        Returns:
            A tensor of shape [batch_size, seq_len, hidden]
        """
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask, register_hook=register_hook))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """Transformer layer.

    Paper: Ashish Vaswani et al. Attention is all you need. NIPS 2017.

    This layer is used in the Transformer model. But it can also be used
    as a standalone layer.

    Args:
        feature_size: the hidden feature size.
        heads: the number of attention heads. Default is 1.
        dropout: dropout rate. Default is 0.5.
        num_layers: number of transformer layers. Default is 1.
        register_hook: True to save gradients of attention layer, Default is False.
    Examples:
        >>> from pyhealth.models import TransformerLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = TransformerLayer(64)
        >>> emb, cls_emb = layer(input)
        >>> emb.shape
        torch.Size([3, 128, 64])
        >>> cls_emb.shape
        torch.Size([3, 64])
    """

    def __init__(self, feature_size, heads=1, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(feature_size, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, register_hook: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            emb: a tensor of shape [batch size, sequence len, feature_size],
                containing the output features for each time step.
            cls_emb: a tensor of shape [batch size, feature_size], containing
                the output features for the first time step.
        """
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask, register_hook)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class Transformer(BaseModel):
    """Transformer model for PyHealth 2.0 datasets.

    Each feature stream is embedded with :class:`EmbeddingModel` and encoded by
    an independent :class:`TransformerLayer`. The resulting [CLS]-style
    embeddings are concatenated and passed to a classification head.

    Args:
        dataset (SampleDataset): dataset providing processed inputs.
        embedding_dim (int): shared embedding dimension.
        heads (int): number of attention heads per transformer block.
        dropout (float): dropout rate applied inside transformer blocks.
        num_layers (int): number of transformer blocks per feature stream.

    Examples:
        >>> from pyhealth.datasets import SampleDataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "diagnoses": ["A", "B", "C"],
        ...         "procedures": ["X", "Y"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-0",
        ...         "diagnoses": ["D"],
        ...         "procedures": ["Z", "Y"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> input_schema = {"diagnoses": "sequence", "procedures": "sequence"}
        >>> output_schema = {"label": "binary"}
        >>> dataset = SampleDataset(
        ...     samples,
        ...     input_schema,
        ...     output_schema,
        ...     dataset_name="demo",
        ... )
        >>> model = Transformer(dataset=dataset)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> sorted(output.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        heads: int = 1,
        dropout: float = 0.5,
        num_layers: int = 1,
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.heads = heads
        self.dropout = dropout
        self.num_layers = num_layers

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if Transformer is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }

        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                feature_size=embedding_dim,
                heads=heads,
                dropout=dropout,
                num_layers=num_layers,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

    @staticmethod
    def _split_temporal(feature):
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, (SequenceProcessor, StageNetProcessor)):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _create_mask(self, feature_key: str, value: torch.Tensor) -> torch.Tensor:
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            mask = value != 0
        elif isinstance(processor, StageNetProcessor):
            if value.dim() >= 3:
                mask = torch.any(value != 0, dim=-1)
            else:
                mask = value != 0
        elif isinstance(processor, (TimeseriesProcessor, StageNetTensorProcessor)):
            if value.dim() >= 3:
                mask = torch.any(torch.abs(value) > 0, dim=-1)
            elif value.dim() == 2:
                mask = torch.any(torch.abs(value) > 0, dim=-1, keepdim=True)
            else:
                mask = torch.ones(value.size(0), 1, dtype=torch.bool, device=value.device)
        elif isinstance(processor, (TensorProcessor, MultiHotProcessor)):
            mask = torch.ones(value.size(0), 1, dtype=torch.bool, device=value.device)
        else:
            if value.dim() >= 2:
                mask = torch.any(value != 0, dim=-1)
            else:
                mask = torch.ones(value.size(0), 1, dtype=torch.bool, device=value.device)

        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        mask = mask.bool()
        if mask.dim() == 2:
            invalid_rows = ~mask.any(dim=1)
            if invalid_rows.any():
                mask[invalid_rows, 0] = True
        return mask

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation with PyHealth 2.0 inputs."""
        register_hook = bool(kwargs.get("register_hook", False))
        patient_emb = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(feature_key, value_tensor)

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            mask = masks[feature_key].to(self.device)
            x = self._pool_embedding(x)
            _, cls_emb = self.transformer[feature_key](x, mask, register_hook)
            patient_emb.append(cls_emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "diagnoses": ["A", "B", "C"],
            "procedures": ["X", "Y"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "diagnoses": ["D", "E"],
            "procedures": ["Z"],
            "label": 0,
        },
    ]

    input_schema: Dict[str, Union[str, type[FeatureProcessor]]] = {
        "diagnoses": "sequence",
        "procedures": "sequence",
    }
    output_schema: Dict[str, Union[str, type[FeatureProcessor]]] = {"label": "binary"}

    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = Transformer(dataset=dataset, embedding_dim=64, heads=2, num_layers=2)

    data_batch = next(iter(train_loader))

    result = model(**data_batch)
    print(result)

    result["loss"].backward()
