# Author(s): Ashrith Anumala, Edmon Guan, Tianyu Zhou
# NetID(s): anumala3, edmong2, aliang7
# Paper: When Attention Fails: Pitfalls of Attention-based Model Interpretability
#        for High-dimensional Clinical Time-Series (Yadav & Subbian, CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/yadav25a.html
# Description: AttentionLSTM model — temporal attention-weighted LSTM for
#              clinical time-series classification
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel
from .embedding import EmbeddingModel


class RNNLayer(nn.Module):
    """Recurrent neural network layer.

    This layer wraps the PyTorch RNN layer with masking and dropout support. It is
    used in the RNN model. But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        rnn_type: type of rnn, one of "RNN", "LSTM", "GRU". Default is "GRU".
        num_layers: number of recurrent layers. Default is 1.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            RNN layer. Default is 0.5.
        bidirectional: whether to use bidirectional recurrent layers. If True,
            a fully-connected layer is applied to the concatenation of the forward
            and backward hidden states to reduce the dimension to hidden_size.
            Default is False.

    Examples:
        >>> from pyhealth.models import RNNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = RNNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ) -> None:
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
        """
        # pytorch's rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        # Ensure tensor is contiguous for cuDNN compatibility
        x = x.contiguous()
        x = rnn_utils.pack_padded_sequence(
            x.contiguous(), lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        # Ensure outputs are contiguous after unpacking
        outputs = outputs.contiguous()

        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            return outputs, last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            # Ensure view result is contiguous for cuDNN
            outputs = outputs.view(batch_size, outputs.shape[1], -1).contiguous()
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return outputs, last_outputs


class AttentionLSTM(BaseModel):
    """Attention-based LSTM model for clinical time-series classification.

    Applies a separate LSTM layer for each input feature, computes temporal
    attention weights over the LSTM hidden-state sequence, and uses the
    attention-weighted context vector for prediction. Based on the architecture
    studied in Yadav & Subbian (CHIL 2025), which examines the reliability of
    attention-based interpretability in clinical settings.

    Args:
        dataset: SampleDataset used to configure input/output dimensions.
        embedding_dim: dimension of the token embedding. Default is 128.
        hidden_dim: LSTM hidden state dimension. Default is 128.
        **kwargs: additional keyword arguments forwarded to RNNLayer (e.g.
            ``num_layers``, ``dropout``, ``bidirectional``).

    Attributes:
        embedding_model: EmbeddingModel that embeds each input feature.
        rnn: ModuleDict mapping each feature key to a per-feature RNNLayer.
        attention: ModuleDict mapping each feature key to an attention linear.
        fc: final linear layer mapping concatenated context vectors to logits.

    Example:
        >>> # Assumes a SampleDataset with "conditions" (sequence) input and
        >>> # "label" (binary) output has been created.
        >>> model = AttentionLSTM(dataset, embedding_dim=128, hidden_dim=128)
        >>> # output = model(**batch)  # batch from get_dataloader(dataset)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ) -> None:
        """Initialize AttentionLSTM.

        Validates kwargs, builds per-feature LSTM and attention layers, and
        constructs the final classification head.
        """
        super(AttentionLSTM, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if AttentionLSTM is initialized"
        )
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.rnn = nn.ModuleDict()
        self.attention = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                rnn_type="LSTM",
                **kwargs,
            )
            self.attention[feature_key] = nn.Linear(hidden_dim, 1)
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(
        self,
        deletion_mask: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label ``kwargs[self.label_key]`` is a tensor of labels for each
        patient in the batch.

        Args:
            deletion_mask: optional dict mapping feature key to a ``(B, T)``
                boolean tensor. Where True, the corresponding embedded time
                step is zeroed out before the RNN call (used for faithfulness
                analysis). Default is None (no deletion).
            **kwargs: keyword arguments for the model. Must contain all
                feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: dictionary with keys:
                - ``loss``: scalar loss tensor.
                - ``y_prob``: predicted probability tensor.
                - ``y_true``: ground-truth label tensor.
                - ``logit``: raw logit tensor.
                - ``attention_weights``: dict mapping feature key to a
                  ``(B, T)`` tensor of softmax attention weights.
                - ``embed`` (optional): ``(B, D)`` patient embedding tensor,
                  included when ``kwargs["embed"]`` is True.
        """
        patient_emb = []
        inputs = {}
        masks = {}
        attn_dict = {}

        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            schema = self.dataset.input_processors[feature_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' in the schema."
                )

            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask

        embedded = self.embedding_model(inputs, masks=masks)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            x_dim_orig = x.dim()
            if x_dim_orig == 4:
                # nested_sequence: (B, num_visits, num_codes, D) — sum-pool codes
                x = x.sum(dim=2)  # (B, num_visits, D)
                if feature_key in masks:
                    mask = (
                        masks[feature_key].to(self.device).sum(dim=-1) > 0
                    ).int()
                else:
                    mask = (torch.abs(x).sum(dim=-1) != 0).int()
            elif x_dim_orig == 2:
                x = x.unsqueeze(1)
                mask = None
            else:
                # 3D: already (B, T, D)
                if feature_key in masks:
                    mask = masks[feature_key].to(self.device).int()
                    if mask.dim() == 3:
                        mask = (mask.sum(dim=-1) > 0).int()
                else:
                    mask = (torch.abs(x).sum(dim=-1) != 0).int()

            if deletion_mask is not None and feature_key in deletion_mask:
                dm = deletion_mask[feature_key].to(self.device)  # (B, T) bool
                if x.dim() == 3:
                    x = x * (~dm).unsqueeze(-1).float()

            outputs, _ = self.rnn[feature_key](x, mask)

            scores = self.attention[feature_key](outputs).squeeze(-1)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            weights = torch.softmax(scores, dim=1)
            attn_dict[feature_key] = weights.detach().cpu()

            x = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        results["attention_weights"] = attn_dict
        return results
