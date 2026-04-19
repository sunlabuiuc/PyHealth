"""DuETT: Dual Event Time Transformer for Electronic Health Records.

Author: Shubham Srivastava (ss253@illinois.edu)
Paper: DuETT: Dual Event Time Transformer for Electronic Health Records.
Paper Link: https://proceedings.mlr.press/v219/labach23a.html

Description:
    This module implements the DuETT model which treats EHR data as a
    two-dimensional event-type x time matrix and applies alternating
    Transformer attention over each axis. The event-axis attention captures
    inter-variable relationships at each timestep, while the time-axis
    attention captures temporal dynamics per variable. The model accepts
    pre-binned tensors where irregular observations have been aggregated
    into fixed time windows, with observation counts retained per cell to
    distinguish true zeros from missing entries.

    Reference: Labach, A.; Pokhrel, A.; Huang, X. S.; Zuberi, S.;
    Yi, S. E.; Volkovs, M.; Poutanen, T.; and Krishnan, R. G. 2023.
    DuETT: Dual Event Time Transformer for Electronic Health Records.
    In Proceedings of the 4th Machine Learning for Health Symposium,
    volume 219 of Proceedings of Machine Learning Research, 295-315.
"""

from typing import Dict

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class DuETTLayer(nn.Module):
    """Core DuETT encoder with dual-axis attention.

    Applies alternating Transformer attention over the event dimension
    (cross-variable relationships) and the time dimension (temporal
    dynamics). Accepts pre-binned event-by-time tensors with separate
    value and observation count inputs.

    Args:
        d_time_series: Number of event-type variables (V).
        d_static: Dimension of static patient features.
        d_embedding: Hidden dimension for Transformer layers.
            Default is 128.
        n_event_layers: Number of event-axis Transformer encoder
            layers. Default is 1.
        n_time_layers: Number of time-axis Transformer encoder
            layers. Default is 1.
        n_heads: Number of attention heads. Default is 4.
        dropout: Dropout rate. Default is 0.3.
        fusion_method: Method for pooling the final representation.
            One of "rep_token", "averaging", or "masked_embed".
            Default is "rep_token".

    Examples:
        >>> layer = DuETTLayer(d_time_series=10, d_static=2)
        >>> x_values = torch.randn(4, 24, 10)  # (B, T, V)
        >>> x_counts = torch.ones(4, 24, 10)   # (B, T, V)
        >>> static = torch.randn(4, 2)          # (B, S)
        >>> times = torch.linspace(0, 1, 24).unsqueeze(0).expand(4, -1)
        >>> emb = layer(x_values, x_counts, static, times)
        >>> emb.shape
        torch.Size([4, 128])
    """

    def __init__(
        self,
        d_time_series: int,
        d_static: int,
        d_embedding: int = 128,
        n_event_layers: int = 1,
        n_time_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.3,
        fusion_method: str = "rep_token",
    ):
        super().__init__()

        self.d_time_series = d_time_series
        self.d_static = d_static
        self.d_embedding = d_embedding
        self.n_event_layers = n_event_layers
        self.n_time_layers = n_time_layers
        self.fusion_method = fusion_method

        # Per-variable value embeddings: project each scalar to d_embedding
        self.value_embeddings = nn.ModuleList(
            [nn.Linear(1, d_embedding) for _ in range(d_time_series)]
        )

        # Per-variable count embeddings
        self.count_embeddings = nn.ModuleList(
            [nn.Linear(1, d_embedding) for _ in range(d_time_series)]
        )

        # Static feature encoder
        if d_static > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(d_static, d_embedding),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_embedding, d_embedding),
            )
        else:
            self.static_encoder = None

        # Time projection: project scalar bin times to d_embedding
        self.time_proj = nn.Linear(1, d_embedding)

        # Event-axis Transformer layers
        event_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embedding,
            nhead=n_heads,
            dim_feedforward=d_embedding * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.event_transformer = nn.TransformerEncoder(
            event_encoder_layer, num_layers=n_event_layers
        )

        # Time-axis Transformer layers
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embedding,
            nhead=n_heads,
            dim_feedforward=d_embedding * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.time_transformer = nn.TransformerEncoder(
            time_encoder_layer, num_layers=n_time_layers
        )

        # Representation token for pooling
        if fusion_method == "rep_token":
            self.rep_token = nn.Parameter(
                torch.randn(1, 1, d_embedding) * 0.02
            )

        # Layer norm before output
        self.output_norm = nn.LayerNorm(d_embedding)

    def forward(
        self,
        x_values: torch.Tensor,
        x_counts: torch.Tensor,
        static: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the DuETT encoder.

        Args:
            x_values: Binned time-series values of shape (B, T, V).
            x_counts: Observation counts of shape (B, T, V).
            static: Static patient features of shape (B, S).
            times: Bin endpoint times of shape (B, T).

        Returns:
            Patient embedding tensor of shape (B, d_embedding).
        """
        B, T, V = x_values.shape

        # Per-variable embedding: project each variable independently
        var_embeddings = []
        for v in range(V):
            val_emb = self.value_embeddings[v](
                x_values[:, :, v : v + 1]
            )  # (B, T, D)
            cnt_emb = self.count_embeddings[v](
                x_counts[:, :, v : v + 1]
            )  # (B, T, D)
            var_embeddings.append(val_emb + cnt_emb)

        # Stack: (B, T, V, D)
        x = torch.stack(var_embeddings, dim=2)

        # Add time-based positional encoding using actual bin times
        time_emb = self.time_proj(
            times.unsqueeze(-1)
        )  # (B, T, D)
        x = x + time_emb.unsqueeze(2)  # broadcast across V

        # Fuse static features (broadcast across T and V)
        if self.static_encoder is not None:
            static_emb = self.static_encoder(static)  # (B, D)
            x = x + static_emb.unsqueeze(1).unsqueeze(2)

        # Dual-axis attention: event then time
        # Event-axis attention: attend across variables at each timestep
        # Reshape (B, T, V, D) -> (B*T, V, D)
        x = x.reshape(B * T, V, -1)

        if self.fusion_method == "rep_token":
            # Prepend rep token along variable dimension
            rep = self.rep_token.expand(B * T, -1, -1)  # (B*T, 1, D)
            x = torch.cat([rep, x], dim=1)  # (B*T, V+1, D)

        x = self.event_transformer(x)  # (B*T, V(+1), D)

        if self.fusion_method == "rep_token":
            # Extract rep token and variable embeddings separately
            rep_out = x[:, 0, :]  # (B*T, D)
            x = x[:, 1:, :]  # (B*T, V, D)

        # Reshape back: (B, T, V, D)
        x = x.reshape(B, T, V, -1)

        # Time-axis attention: attend across timesteps for each variable
        # Reshape (B, T, V, D) -> (B*V, T, D)
        x = x.permute(0, 2, 1, 3).reshape(B * V, T, -1)
        x = self.time_transformer(x)  # (B*V, T, D)

        # Reshape back: (B, V, T, D)
        x = x.reshape(B, V, T, -1)

        # Pooling to (B, D)
        if self.fusion_method == "rep_token":
            # Use rep token output, averaged over time
            rep_out = rep_out.reshape(B, T, -1)  # (B, T, D)
            patient_emb = rep_out.mean(dim=1)  # (B, D)
        elif self.fusion_method == "averaging":
            # Average over both V and T
            patient_emb = x.mean(dim=(1, 2))  # (B, D)
        elif self.fusion_method == "masked_embed":
            # Weight by observation counts
            # Sum counts across variables for each timestep
            count_weights = x_counts.sum(dim=2)  # (B, T)
            count_weights = count_weights / (
                count_weights.sum(dim=1, keepdim=True) + 1e-8
            )
            # Average over variables, weighted average over time
            x_var_avg = x.mean(dim=1)  # (B, T, D)
            patient_emb = (
                x_var_avg * count_weights.unsqueeze(-1)
            ).sum(dim=1)  # (B, D)
        else:
            raise ValueError(
                f"Unknown fusion method: {self.fusion_method}"
            )

        patient_emb = self.output_norm(patient_emb)
        return patient_emb


class DuETT(BaseModel):
    """DuETT model for clinical prediction from EHR time series.

    DuETT (Dual Event Time Transformer) models electronic health records
    along two explicit axes: event type and time, using alternating
    attention over each. It accepts pre-binned event-by-time tensors where
    irregular observations have been aggregated into fixed time windows.

    This model does NOT use EmbeddingModel because DuETT requires
    per-variable linear projections, which is integral to its
    architecture.

    Args:
        dataset: The SampleDataset used to train the model.
        ts_values_key: Key for binned time-series values in the sample
            dict. Default is "ts_values".
        ts_counts_key: Key for observation counts in the sample dict.
            Default is "ts_counts".
        static_key: Key for static patient features. Default is "static".
        times_key: Key for bin endpoint times. Default is "times".
        d_embedding: Hidden dimension for Transformer layers.
            Default is 128.
        n_event_layers: Number of event-axis Transformer layers.
            Default is 1.
        n_time_layers: Number of time-axis Transformer layers.
            Default is 1.
        n_heads: Number of attention heads. Default is 4.
        dropout: Dropout rate. Default is 0.3.
        fusion_method: Pooling method. One of "rep_token", "averaging",
            or "masked_embed". Default is "rep_token".

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "ts_values": [[0.5, 0.3], [0.1, 0.0]],
        ...         "ts_counts": [[1.0, 1.0], [1.0, 0.0]],
        ...         "static": [0.65, 1.0],
        ...         "times": [0.5, 1.0],
        ...         "mortality": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "ts_values": [[0.8, 0.2], [0.4, 0.6]],
        ...         "ts_counts": [[1.0, 1.0], [1.0, 1.0]],
        ...         "static": [0.45, 0.0],
        ...         "times": [0.5, 1.0],
        ...         "mortality": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "ts_values": "tensor",
        ...         "ts_counts": "tensor",
        ...         "static": "tensor",
        ...         "times": "tensor",
        ...     },
        ...     output_schema={"mortality": "binary"},
        ...     dataset_name="test",
        ... )
        >>> model = DuETT(dataset=dataset, d_embedding=64)
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=2)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> output["y_prob"].shape
        torch.Size([2, 1])

    Note:
        Paper: Labach et al. 2023. DuETT: Dual Event Time Transformer
        for Electronic Health Records. ML4H 2023, PMLR 219:295-315.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        ts_values_key: str = "ts_values",
        ts_counts_key: str = "ts_counts",
        static_key: str = "static",
        times_key: str = "times",
        d_embedding: int = 128,
        n_event_layers: int = 1,
        n_time_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.3,
        fusion_method: str = "rep_token",
    ):
        super().__init__(dataset=dataset)

        self.ts_values_key = ts_values_key
        self.ts_counts_key = ts_counts_key
        self.static_key = static_key
        self.times_key = times_key
        self.d_embedding = d_embedding
        self.n_event_layers = n_event_layers
        self.n_time_layers = n_time_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.fusion_method = fusion_method

        assert (
            len(self.label_keys) == 1
        ), "DuETT supports a single label key"
        self.label_key = self.label_keys[0]

        # Determine dimensions from the dataset
        first_sample = dataset[0]
        ts_sample = first_sample[ts_values_key]
        if isinstance(ts_sample, torch.Tensor):
            d_time_series = ts_sample.shape[-1]
        else:
            d_time_series = len(ts_sample[0]) if ts_sample else 1

        static_sample = first_sample[static_key]
        if isinstance(static_sample, torch.Tensor):
            d_static = static_sample.shape[-1]
        else:
            d_static = len(static_sample) if static_sample else 0

        self.d_time_series = d_time_series
        self.d_static = d_static

        # Core DuETT encoder
        self.duett_layer = DuETTLayer(
            d_time_series=d_time_series,
            d_static=d_static,
            d_embedding=d_embedding,
            n_event_layers=n_event_layers,
            n_time_layers=n_time_layers,
            n_heads=n_heads,
            dropout=dropout,
            fusion_method=fusion_method,
        )

        # Classification head
        output_size = self.get_output_size()
        self.fc = nn.Sequential(
            nn.Linear(d_embedding, d_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embedding, output_size),
        )

    def forward(
        self, **kwargs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the DuETT model.

        Args:
            **kwargs: Keyword arguments containing input tensors keyed
                by ts_values_key, ts_counts_key, static_key, times_key,
                and the label key.

        Returns:
            Dict with keys "loss", "y_prob", "y_true", "logit", and
            optionally "embed".
        """
        ts_values = kwargs[self.ts_values_key].float().to(self.device)
        ts_counts = kwargs[self.ts_counts_key].float().to(self.device)
        static = kwargs[self.static_key].float().to(self.device)
        times = kwargs[self.times_key].float().to(self.device)

        # Ensure 2D tensors are expanded to 3D if needed
        if ts_values.dim() == 2:
            ts_values = ts_values.unsqueeze(-1)
            ts_counts = ts_counts.unsqueeze(-1)

        # Get patient embedding from DuETT encoder
        patient_emb = self.duett_layer(
            ts_values, ts_counts, static, times
        )

        # Classification
        logits = self.fc(patient_emb)

        # Compute loss and probabilities
        y_true = kwargs[self.label_key].to(self.device)
        loss_fn = self.get_loss_function()

        if self.mode == "multiclass":
            # cross_entropy expects (N,) long targets
            loss = loss_fn(logits, y_true.long())
        else:
            # binary/multilabel expect matching shapes
            if y_true.dim() == 1 and logits.dim() == 2:
                y_true = y_true.unsqueeze(-1).float()
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
