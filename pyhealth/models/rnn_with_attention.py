from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import (
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
    MultiHotProcessor,
    NestedFloatsProcessor,
    NestedSequenceProcessor,
    SequenceProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)

from .embedding import EmbeddingModel


class RNN_Attention_Layer(nn.Module):

    def __init__(self, feature_size: int, dropout: float = 0.5, num_heads=4):
        super(RNN_Attention_Layer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.num_heads = num_heads
        self.attention = self.MultiHeadAttention(feature_size, num_heads)

    class MultiHeadAttention(nn.Module):
        def __init__(self, hidden_dim, num_heads):
            super().__init__()

            self.num_heads = num_heads

            self.W = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_heads)])
            self.v = nn.ModuleList(
                [nn.Linear(hidden_dim, 1, bias=False) for _ in range(num_heads)])

        def forward(self, h_seq, mask=None):
            head_outputs = []

            for k in range(self.num_heads):
                # (B, T, H)
                h_proj = torch.tanh(self.W[k](h_seq))
                # (B, T, 1)
                scores = self.v[k](h_proj)
                if mask is not None:
                    # scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                    scores = scores.masked_fill(
                        ~mask.unsqueeze(-1), float('-inf'))
                # (B, T, 1)
                alpha = torch.softmax(scores, dim=1)
                # (B, H)
                z_k = torch.sum(alpha * h_seq, dim=1)
                head_outputs.append(z_k)

            # (B, K * H)
            z = torch.cat(head_outputs, dim=-1)

            return z

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            c: a tensor of shape [batch size, feature_size] representing the
                context vector.
        """
        x = self.dropout_layer(x)
        batch_size, sequence_len, feature_size = x.shape

        h_seq, _ = self.gru(x)   # (B, T, F)

        z = self.attention(h_seq, mask)  # (B, K * F)

        return z


class RNN_attention(BaseModel):

    def __init__(self, dataset: SampleDataset, embedding_dim: int = 128, **kwargs):
        super(RNN_attention, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim

        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Create RNN layers for each feature
        # self.rnn_attention = nn.ModuleDict()
        self.rnn_attention = RNN_Attention_Layer(
            feature_size=embedding_dim, **kwargs)

        output_size = self.get_output_size()
        num_features = len(self.feature_keys)
        self.fc = nn.Linear(num_features * self.embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        embedded = self.embedding_model(kwargs)

        visit_reps = []

        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            if len(x.shape) == 4:
                x = torch.mean(x, dim=2)   # (B, T, E)

            elif len(x.shape) == 3:
                pass

            elif len(x.shape) == 2:
                x = x.unsqueeze(1)

            else:
                raise ValueError(
                    f"Unexpected tensor shape {x.shape} for feature {feature_key}"
                )

            visit_reps.append(x)

        # x = torch.stack(visit_reps, dim=0).sum(dim=0)
        x = torch.stack(visit_reps, dim=0).mean(dim=0)
        mask = (x.abs().sum(dim=-1) > 0)

        z = self.rnn_attention(x, mask)
        # shape = (patient, label_size)
        logits = self.fc(z)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = z
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": [["A", "B"], ["C", "D", "E"]],
            "procedures": [["P1"], ["P2", "P3"]],
            "drugs_hist": [[], ["D1", "D2"]],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "conditions": [["F"], ["G", "H"]],
            "procedures": [["P4", "P5"], ["P6"]],
            "drugs_hist": [["D3"], ["D4", "D5"]],
            "label": 0,
        },
    ]

    # dataset
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
            "drugs_hist": "nested_sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = RNN_attention(dataset=dataset)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
