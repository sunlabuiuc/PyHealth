# Contributor: Nikhita Shanker (nikhita2)
# Paper: "Robust Mortality Prediction in the ICU using Temporal Difference Learning"
#        https://arxiv.org/pdf/2411.04285

"""CNN-LSTM hybrid model for clinical prediction tasks.

This module implements the CNN-LSTM architecture from:
    "Robust Mortality Prediction in the Intensive Care Unit using
    Temporal Difference Learning" (Frost et al.)
    https://github.com/tdgfrost/td-icu-mortality

The model processes each input feature through an embedding layer,
a CNN encoder (Conv1d -> BatchNorm -> ReLU -> MaxPool), and an LSTM
encoder for sequential dependency modeling. Feature representations
are concatenated and passed through a dense decoder with batch
normalization to produce the final prediction.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class CNNLSTMPredictor(BaseModel):
    """CNN-LSTM hybrid model for clinical prediction tasks.

    Implements the architecture from "Robust Mortality Prediction in the
    Intensive Care Unit using Temporal Difference Learning" (Frost et al.).

    The model processes each input feature through three stages:

    1. Embedding: Maps discrete medical codes (e.g., ICD codes) to
       learned dense vectors of size embedding_dim.
    2. CNN encoder: A stack of num_cnn_layers blocks, each consisting
       of Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d. Captures
       short-range patterns between adjacent codes.
    3. LSTM encoder: A num_lstm_layers-layer LSTM that processes the
       CNN output sequentially. The final hidden state summarises
       long-range temporal dependencies across the full sequence.

    Per-feature LSTM outputs are concatenated and passed through a dense
    decoder (BatchNorm -> Linear -> ReLU -> Dropout -> BatchNorm ->
    Linear) to produce the final prediction.

    Paper reference: https://github.com/tdgfrost/td-icu-mortality

    Args:
        dataset: A SampleDataset from set_task(). The model reads
            input_schema and output_schema automatically.
        embedding_dim: Dimension of code embeddings. Default: 128.
        hidden_dim: Hidden dimension for CNN and LSTM layers. Default: 128.
        num_cnn_layers: Number of CNN encoder layers. Default: 2.
        num_lstm_layers: Number of LSTM encoder layers. Default: 2.
        dropout: Dropout rate for regularization. Default: 0.3.

    Examples:
        >>> from pyhealth.models import CNNLSTMPredictor
        >>> model = CNNLSTMPredictor(
        ...     dataset=samples,
        ...     embedding_dim=128,
        ...     hidden_dim=128,
        ...     num_cnn_layers=2,
        ...     num_lstm_layers=2,
        ...     dropout=0.3,
        ... )
        >>> # Forward pass returns dict with loss, y_prob, y_true, logit
        >>> output = model(**batch)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_cnn_layers: int = 2,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_cnn_layers = num_cnn_layers
        self.num_lstm_layers = num_lstm_layers

        # Embedding for coded features
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # CNN encoder per feature key
        # Each CNN layer consists of Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d
        self.cnn_layers = nn.ModuleDict()
        for key in self.feature_keys:
            layers: List[nn.Module] = []
            in_channels = embedding_dim
            for _ in range(num_cnn_layers):
                layers.extend(
                    [
                        nn.Conv1d(
                            in_channels,
                            hidden_dim,
                            kernel_size=2,
                            padding=1,
                        ),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2, stride=1),
                    ]
                )
                in_channels = hidden_dim
            self.cnn_layers[key] = nn.Sequential(*layers)

        # LSTM encoder per feature key
        self.lstm_layers = nn.ModuleDict()
        for key in self.feature_keys:
            self.lstm_layers[key] = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_lstm_layers,
                batch_first=True,
                dropout=dropout if num_lstm_layers > 1 else 0,
            )

        # Dense decoder:
        # BatchNorm -> Linear -> ReLU -> Dropout -> BatchNorm -> Linear)
        total_dim = hidden_dim * len(self.feature_keys)
        output_size = self.get_output_size()
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the CNN-LSTM model.

        Embeds each feature, passes through CNN and LSTM encoders,
        concatenates the hidden states, and decodes to a prediction.

        Args:
            **kwargs: Batch dict from a PyHealth DataLoader. Must contain
                keys matching self.feature_keys and self.label_keys.

                Feature keys (e.g., "conditions", "procedures") map to
                tensors of shape (batch, seq_len) containing integer
                code indices, where seq_len is the number of codes per
                visit (padded to equal length within a batch). Label
                keys (e.g., "mortality") map to tensors of shape
                (batch,) containing ground-truth targets.

        Returns:
            Dict with keys:
                - loss: Scalar loss tensor.
                - y_prob: Predicted probabilities (batch, 1).
                - y_true: Ground truth labels (batch,).
                - logit: Raw logits (batch, 1).
                - embed (optional): Patient embedding if embed=True.
        """
        embedded = self.embedding_model(kwargs)

        patient_embs: List[torch.Tensor] = []
        for key in self.feature_keys:
            x = embedded[key]  # (batch, seq_len, embedding_dim)

            # CNN expects (batch, channels, seq_len)
            # embedding_dim acts as Conv1d channels
            x = x.permute(0, 2, 1)
            x = self.cnn_layers[key](x)

            # LSTM expects (batch, seq_len, features)
            x = x.permute(0, 2, 1)
            _, (h_n, _) = self.lstm_layers[key](x)
            # Take last layer's hidden state
            patient_embs.append(h_n[-1])  # (batch, hidden_dim)

        # Concatenate and decode
        patient_emb = torch.cat(patient_embs, dim=1)
        logits = self.decoder(patient_emb)

        # Compute loss
        y_true = kwargs[self.label_keys[0]].to(self.device)
        loss_fn = self.get_loss_function()
        loss = loss_fn(logits, y_true)
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
