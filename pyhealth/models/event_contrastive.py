import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel


class EventContrastiveModel(BaseModel):
    """Event-Based Contrastive Learning Model for Medical Time Series.

    This model implements a simplified version of Event-Based Contrastive
    Learning (EBCL). It splits patient time-series into events, encodes each
    event, and applies a contrastive learning objective across events.

    Key components:
        1. Event segmentation (fixed-size windows)
        2. Event encoder (GRU)
        3. Projection head (MLP)
        4. Multi-event contrastive loss (InfoNCE)

    Args:
        dataset (Optional): PyHealth dataset object. Can be None for testing.
        input_dim (int): Number of input features per timestep.
        hidden_dim (int): Hidden size of the GRU encoder.
        projection_dim (int): Dimension of contrastive embedding space.
        temperature (float): Temperature parameter for contrastive loss.

    Example:
        >>> model = EventContrastiveModel(dataset=None, input_dim=8)
        >>> x = torch.randn(2, 10, 8)
        >>> embeddings = model(x)
        >>> loss = model.compute_loss(embeddings)
        >>> print(loss.item())
    """

    def __init__(
        self,
        dataset: Optional[object] = None,
        input_dim: int = 8,
        hidden_dim: int = 64,
        projection_dim: int = 32,
        temperature: float = 0.1,
    ) -> None:
        super().__init__(dataset=dataset)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # GRU encoder
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Projection head for contrastive space
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def split_events(self, x: torch.Tensor, window_size: int = 5) -> List[torch.Tensor]:
        """Splits time-series into fixed-size events.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).
            window_size (int): Length of each event window.

        Returns:
            List[torch.Tensor]: List of event tensors, each of shape
                (batch, window_size, features).
        """
        events: List[torch.Tensor] = []
        time_steps = x.shape[1]

        for i in range(0, time_steps - window_size + 1, window_size):
            events.append(x[:, i : i + window_size, :])

        return events

    def encode_event(self, event: torch.Tensor) -> torch.Tensor:
        """Encodes a single event sequence into an embedding.

        Args:
            event (torch.Tensor): Event tensor of shape
                (batch, window_size, features).

        Returns:
            torch.Tensor: Normalized embedding of shape (batch, projection_dim).
        """
        output, _ = self.encoder(event)
        hidden = output[:, -1, :]
        embedding = self.projection_head(hidden)
        embedding = F.normalize(embedding, dim=-1)
        return embedding

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Performs forward pass and returns event embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time, features).

        Returns:
            List[torch.Tensor]: List of embeddings, one per event.
        """
        events = self.split_events(x)

        # Random event sampling (EBCL-style augmentation)
        if len(events) > 2:
            events = random.sample(events, k=2)

        embeddings: List[torch.Tensor] = []
        for event in events:
            embeddings.append(self.encode_event(event))

        return embeddings

    def compute_loss(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Computes multi-event contrastive loss (InfoNCE).

        Args:
            embeddings (List[torch.Tensor]): List of embeddings, each of shape
                (batch, projection_dim).

        Returns:
            torch.Tensor: Scalar loss value.

        Raises:
            ValueError: If fewer than 2 events are provided.
        """
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 events for contrastive learning")

        # Concatenate embeddings across events
        z = torch.cat(embeddings, dim=0)

        # Compute similarity matrix
        similarity = torch.matmul(z, z.T) / self.temperature

        batch_size = embeddings[0].size(0)
        num_events = len(embeddings)

        # Labels: same patient across events are positives
        labels = torch.arange(batch_size).repeat(num_events)
        labels = labels.to(z.device)

        loss = F.cross_entropy(similarity, labels)

        return loss