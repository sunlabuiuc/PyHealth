import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import BaseModel


class EventContrastiveModel(BaseModel):
    """Event-Based Contrastive Learning Model for Medical Time Series.

    This model implements a simplified version of EBCL by:
    1. Splitting time-series into fixed-size events
    2. Encoding each event
    3. Applying multi-event contrastive learning (InfoNCE)

    Args:
        dataset: PyHealth dataset (can be None for testing)
        input_dim: number of input features
        hidden_dim: encoder hidden size
        projection_dim: dimension for contrastive space
        temperature: scaling factor for contrastive loss
    """

    def __init__(
        self,
        dataset=None,
        input_dim: int = 8,
        hidden_dim: int = 64,
        projection_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__(dataset=dataset)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Encoder (GRU)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def split_events(self, x, window_size=5):
        """Split time-series into fixed-size events.

        Args:
            x: tensor (batch, time, features)

        Returns:
            list of event tensors
        """
        events = []
        T = x.shape[1]

        for i in range(0, T - window_size + 1, window_size):
            events.append(x[:, i : i + window_size, :])

        return events

    def encode_event(self, event):
        """Encode a single event sequence."""
        output, _ = self.encoder(event)
        h = output[:, -1, :]
        z = self.projection_head(h)
        z = F.normalize(z, dim=-1)
        return z

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, time, features)

        Returns:
            list of embeddings (one per event)
        """
        events = self.split_events(x)

        # random sampling (EBCL-style)
        if len(events) > 2:
            import random
            events = random.sample(events, k=2)

        embeddings = []
        for event in events:
            z = self.encode_event(event)
            embeddings.append(z)

        return embeddings

    def compute_loss(self, embeddings):
        """Multi-event contrastive loss (InfoNCE-style)."""
        if len(embeddings) < 2:
            raise ValueError("Need at least 2 events for contrastive learning")

        # Concatenate all event embeddings
        z = torch.cat(embeddings, dim=0)

        # Similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature

        batch_size = embeddings[0].size(0)
        num_events = len(embeddings)

        # Labels: same patient across events are positives
        labels = torch.arange(batch_size).repeat(num_events)
        labels = labels.to(z.device)

        loss = F.cross_entropy(sim, labels)

        return loss