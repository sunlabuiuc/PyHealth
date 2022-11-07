from typing import List

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel

VALID_OPERATION_LEVEL = ["visit", "event"]


class TransformerLayer(nn.Module):
    """Transformer layer.

    This layer wraps the PyTorch Transformer layer with masking support. It is
    used in the Transformer model. But it can also be used as a standalone layer.

    Args:
        input_size: the number of expected features in the input
        hidden_size: the dimension of the feedforward network model.
        nhead: the number of heads in the multiheadattention models. Default is 1.
        dropout: dropout rate. Default is 0.5.
        num_layers: number of Transformer layers. Default is 1.

    Examples:
        >>> from pyhealth.models import TransformerLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> mask = torch.ones(3, 128).bool()
        >>> layer = TransformerLayer(5, 64)
        >>> outputs, last_outputs = layer(input, mask)
        >>> outputs.shape
        torch.Size([3, 128, 5])
        >>> last_outputs.shape
        torch.Size([3, 5])
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            nhead: int = 1,
            dropout: float = 0.5,
            num_layers: int = 1,
    ):
        super(TransformerLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            dim_feedforward=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(input_size)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers, encoder_norm
        )

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: a tensor of shape [batch size, sequence len] where 1 indicates
                valid and 0 indicates invalid.

        Returns:
            outputs: a tensor of shape [batch size, sequence len, input size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, input size], containing
                the output features for the last time step.
        """
        src_key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        outputs = x * mask.unsqueeze(-1).float()
        last_outputs = x.sum(dim=1)
        return outputs, last_outputs


class Transformer(BaseModel):
    """Transformer model.

    This model applies a separate Transformer layer for each feature, and then
    concatenates the final hidden states of each Transformer layer. The concatenated
    hidden states are then fed into a fully connected layer to make predictions.

    Note:
        This model can operate on both visit and event level, as designated by
            the operation_level parameter.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        operation_level: one of "visit", "event".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the Transformer layer.
    """

    def __init__(
            self,
            dataset: BaseDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            operation_level: str,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            **kwargs
    ):
        super(Transformer, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        assert operation_level in VALID_OPERATION_LEVEL, \
            f"operation_level must be one of {VALID_OPERATION_LEVEL}"
        self.operation_level = operation_level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)
        self.transformer = nn.ModuleDict()
        for feature_key in feature_keys:
            self.transformer[feature_key] = TransformerLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size(self.label_tokenizer)
        # transformer's output feature size is still embedding_dim
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def visit_level_forward(self, **kwargs):
        """Visit-level Transformer forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == list
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            # (patient, visit, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.sum(x, dim=2) != 0
            # (patient, hidden_dim)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def event_level_forward(self, **kwargs):
        """Event-level Transformer forward."""
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == str
            x = self.feat_tokenizers[feature_key].batch_encode_2d(kwargs[feature_key])
            # (patient, code)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, code)
            mask = torch.sum(x, dim=2) != 0
            # (patient, hidden_dim)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)
        # (patient, features * hidden_dim)
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }

    def forward(self, **kwargs):
        if self.operation_level == "visit":
            return self.visit_level_forward(**kwargs)
        elif self.operation_level == "event":
            return self.event_level_forward(**kwargs)
        else:
            raise NotImplementedError
