from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


class RETAINLayer(nn.Module):
    """RETAIN layer.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    This layer is used in the RETAIN model. But it can also be used as a
    standalone layer.

    Args:
        feature_size: the hidden feature size.
        dropout: dropout rate. Default is 0.5.

    Examples:
        >>> from pyhealth.models import RETAINLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = RETAINLayer(64)
        >>> c = layer(input)
        >>> c.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        feature_size: int,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.feature_size = feature_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(feature_size, feature_size, batch_first=True)
        self.beta_gru = nn.GRU(feature_size, feature_size, batch_first=True)

        self.alpha_li = nn.Linear(feature_size, 1)
        self.beta_li = nn.Linear(feature_size, feature_size)

    @staticmethod
    def reverse_x(input, lengths):
        """Reverses the input."""
        reversed_input = input.new(input.size())
        for i, length in enumerate(lengths):
            reversed_input[i, :length] = input[i, :length].flip(dims=[0])
        return reversed_input

    def compute_alpha(self, rx, lengths):
        """Computes alpha attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        g, _ = self.alpha_gru(rx)
        g, _ = rnn_utils.pad_packed_sequence(g, batch_first=True)
        attn_alpha = torch.softmax(self.alpha_li(g), dim=1)
        return attn_alpha

    def compute_beta(self, rx, lengths):
        """Computes beta attention."""
        rx = rnn_utils.pack_padded_sequence(
            rx, lengths, batch_first=True, enforce_sorted=False
        )
        h, _ = self.beta_gru(rx)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)
        attn_beta = torch.tanh(self.beta_li(h))
        return attn_beta

    def forward(
        self,
        x: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, feature_size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            c: a tensor of shape [batch size, feature_size] representing the
                context vector.
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        rx = self.reverse_x(x, lengths)
        attn_alpha = self.compute_alpha(rx, lengths)
        attn_beta = self.compute_beta(rx, lengths)
        c = attn_alpha * attn_beta * x  # (patient, sequence len, feature_size)
        c = torch.sum(c, dim=1)  # (patient, feature_size)
        return c


class RETAIN(BaseModel):
    """RETAIN model.

    Paper: Edward Choi et al. RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention Mechanism. NIPS 2016.

    This model uses separate RETAIN layers for different features and applies
    reverse time attention to capture temporal dependencies. It now uses the
    unified EmbeddingModel for handling various input types.

    The model supports various input types through processors:
        - SequenceProcessor: Code sequences (e.g., diagnosis codes)
        - NestedSequenceProcessor: Nested code sequences (visit histories)
        - TimeseriesProcessor: Time series features
        - NestedSequenceFloatsProcessor: Nested numerical sequences

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters for the RETAIN layer.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": [["A", "B"], ["C"]],
        ...         "procedures": [["P1"], ["P2", "P3"]],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-1",
        ...         "conditions": [["D"], ["E", "F"]],
        ...         "procedures": [["P4"]],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "nested_sequence",
        ...         "procedures": "nested_sequence",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = RETAIN(dataset=dataset)
        >>>
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(RETAIN, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim

        # validate kwargs for RETAIN layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        # Use EmbeddingModel for unified embedding handling
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Create RETAIN layers for each feature
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(
                feature_size=embedding_dim, **kwargs
            )

        output_size = self.get_output_size()
        num_features = len(self.feature_keys)
        self.fc = nn.Linear(
            num_features * self.embedding_dim, output_size
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): patient embeddings if requested.
        """
        patient_emb = []
        embedded = self.embedding_model(kwargs)
        
        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            
            # Handle different input dimensions
            # Case 1: 4D tensor from NestedSequenceProcessor
            # (batch, visits, events, embedding_dim)
            # Need to sum across events to get (batch, visits, embedding_dim)
            if len(x.shape) == 4:
                x = torch.sum(x, dim=2)  # Sum across events within visit
            
            # Case 2: 3D tensor from SequenceProcessor or after summing
            # (batch, seq_len, embedding_dim) - already correct format
            elif len(x.shape) == 3:
                pass  # Already correct format
            
            # Case 3: 2D tensor - shouldn't happen for RETAIN but handle it
            elif len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add seq dim: (batch, 1, embedding_dim)
            
            else:
                raise ValueError(
                    f"Unexpected tensor shape {x.shape} for feature "
                    f"{feature_key}"
                )

            # Create mask: non-padding entries are valid
            # Check if all values in embedding dimension are zero (padding)
            # (batch_size, num_visits)
            mask = (x.abs().sum(dim=-1) > 0).float()

            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
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
            results["embed"] = patient_emb
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
    model = RETAIN(dataset=dataset)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
