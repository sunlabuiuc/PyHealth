from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


class RNNAttentionLayer(nn.Module):
    """Recurrent neural network layer with multi-headed attention.

    This layer combines a GRU with multi-headed attention to capture dependencies
    in sequential patient visit data. Each attention head learns different
    representations of the sequence, and outputs are concatenated.

    Args:
        feature_size (int): Size of input features.
        dropout (float): Dropout rate. Default is 0.5.
        h (int): Number of attention heads. Default is 4.

    Examples:
        >>> import torch
        >>> layer = RNNAttentionLayer(feature_size=128, h=4)
        >>> x = torch.randn(8, 10, 128)  # (batch, seq_len, feature_size)
        >>> mask = torch.ones(8, 10, dtype=torch.bool)
        >>> output = layer(x, mask)
        >>> output.shape
        torch.Size([8, 512])  # (batch, num_heads * feature_size)
    """

    def __init__(self, feature_size: int, dropout: float = 0.5, h: int = 4):
        super(RNNAttentionLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.gru = nn.GRU(feature_size, feature_size, num_layers = 2, batch_first=True)
        self.num_heads = h
        self.attention = self.MultiHeadAttention(feature_size, h)

    class MultiHeadAttention(nn.Module):
        """Multi-head attention mechanism.

        Implements multi-head attention as described in the paper
        "Predicting utilization of healthcare services from individual
        disease trajectories using RNNs with multi-headed attention".

        Each head learns to attend to different aspects of the sequence.
        """

        def __init__(self, hidden_dim: int, h: int):
            super().__init__()
            self.num_heads = h
            self.W = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(h)]
            )
            self.v = nn.ModuleList(
                [nn.Linear(hidden_dim, 1, bias=False) for _ in range(h)]
            )

        def forward(
            self, h_seq: torch.Tensor, mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Forward pass for multi-head attention.

            Args:
                h_seq (torch.Tensor): GRU hidden states of shape
                    [batch_size, seq_len, hidden_dim].
                mask (torch.Tensor, optional): Mask of shape [batch_size, seq_len]
                    where True indicates valid positions. Default is None.

            Returns:
                torch.Tensor: Attention output of shape
                    [batch_size, num_heads * hidden_dim].
            """
            head_outputs = []

            for k in range(self.num_heads):
                # Project to attention space: (B, T, H)
                h_proj = torch.tanh(self.W[k](h_seq))
                # Compute attention scores: (B, T, 1)
                scores = self.v[k](h_proj)
                if mask is not None:
                    scores = scores.masked_fill(
                        ~mask.unsqueeze(-1), float("-inf"))
                # Softmax to get attention weights: (B, T, 1)
                alpha = torch.softmax(scores, dim=1)
                # Weighted sum of sequences: (B, H)
                z_k = torch.sum(alpha * h_seq, dim=1)
                head_outputs.append(z_k)

            # Concatenate all heads: (B, K * H)
            z = torch.cat(head_outputs, dim=-1)
            return z

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward propagation through GRU and attention.

        Args:
            x (torch.Tensor): Input tensor of shape
                [batch_size, seq_len, feature_size].
            mask (torch.Tensor, optional): Mask tensor of shape
                [batch_size, seq_len] where True indicates valid positions.
                Default is None.

        Returns:
            torch.Tensor: Output tensor of shape
                [batch_size, num_heads * feature_size].
        """
        x = self.dropout_layer(x)
        batch_size, sequence_len, feature_size = x.shape

        # GRU forward: (B, T, F)
        h_seq, _ = self.gru(x)

        # Multi-head attention: (B, K * F)
        z = self.attention(h_seq, mask)

        return z


class RNNAttention(BaseModel):
    """Recurrent Neural Network with Multi-headed Attention for EHR prediction.

    This model implements the RNN-based architecture with multi-headed attention
    from "Predicting utilization of healthcare services from individual disease
    trajectories using RNNs with multi-headed attention" (Kumar et al.).

    The model processes sequential patient visit data through an embedding layer,
    aggregates features, passes through a GRU with multi-headed attention, and
    produces predictions for healthcare resource allocation.

    Args:
        dataset (SampleDataset): The dataset containing samples with input and
            output schemas.
        embedding_dim (int): Dimension for embedding layer. Default is 128.
        h (int): Number of attention heads. Default is 4.
        **kwargs: Additional arguments passed to RNNAttentionLayer.

    Attributes:
        embedding_dim (int): Dimension of embeddings.
        num_heads (int): Number of attention heads.
        embedding_model (EmbeddingModel): Model for embedding inputs.
        rnn_attention (RNNAttentionLayer): GRU with attention layer.
        fc (nn.Linear): Final fully connected layer for predictions.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "conditions": [["A01", "B02"], ["C03"]],
        ...         "procedures": [["P01"], ["P02"]],
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "nested_sequence",
        ...         "procedures": "nested_sequence",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = RNNAttention(dataset=dataset, embedding_dim=128, h=4)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        h: int = 4,
        **kwargs
    ):
        super(RNNAttention, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim

        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Create RNN with attention layer
        self.rnn_attention = RNNAttentionLayer(
            feature_size=embedding_dim, h=h, **kwargs
        )

        output_size = self.get_output_size()
        self.num_heads = h
        self.fc = nn.Linear(self.num_heads * embedding_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            **kwargs: Input features and labels as defined in the dataset schema.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - loss: Computed loss value.
                - y_prob: Predicted probabilities.
                - y_true: Ground truth labels.
                - logit: Raw model outputs.
                - embed (optional): Patient embeddings if embed=True in kwargs.
        """
        embedded = self.embedding_model(kwargs)

        visit_reps = []

        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            if len(x.shape) == 4:
                x = torch.mean(x, dim=2)  # (B, T, E)
            elif len(x.shape) == 3:
                pass
            elif len(x.shape) == 2:
                x = x.unsqueeze(1)
            else:
                raise ValueError(
                    f"Unexpected tensor shape {x.shape} for feature {feature_key}"
                )

            visit_reps.append(x)

        # Average across features
        x = torch.stack(visit_reps, dim=0).mean(dim=0)
        mask = x.abs().sum(dim=-1) > 0

        # Apply RNN with attention
        z = self.rnn_attention(x, mask)

        # Final prediction layer
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
    from pyhealth.datasets import create_sample_dataset, get_dataloader

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

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
    model = RNNAttention(dataset=dataset)
    data_batch = next(iter(train_loader))

    ret = model(**data_batch)
    print(ret)

    ret["loss"].backward()
