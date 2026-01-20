"""
MMTM: Multimodal Transfer Module for EHR Representation Learning.

Paper:
    Joze et al. "MMTM: Multimodal Transfer Module for CNN Fusion."
    CVPR 2020. (https://arxiv.org/abs/1911.08670)

This PyHealth implementation adapts the MMTM module for multimodal
Electronic Health Records (EHR) fusion. MMTM is a lightweight module
that exchanges channel-wise attention between two modalities, enabling
parameter-efficient cross-modal representation enhancement.

MMTM can be used independently (via `MMTMLayer`) or within the broader
PyHealth modeling ecosystem (via `MMTM`).

-----------------------------------------------------------------------
Examples (Layer):

    >>> layer = MMTMLayer(64, 128)
    >>> a = torch.randn(8, 64)
    >>> b = torch.randn(8, 128)
    >>> a_hat, b_hat = layer(a, b)

Examples (Model):

    >>> dataset = SampleDataset(...)
    >>> model = MMTM(dataset)
    >>> out = model(**batch)
    >>> out["loss"].backward()
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth.models import EmbeddingModel
from pyhealth.datasets import SampleDataset

__all__ = ["MMTMLayer", "MMTM"]

class MMTMLayer(nn.Module):
    """Multimodal Transfer Module (MMTM).

    Paper:
        Joze et al. "Multimodal Transfer Module for CNN Fusion."
        CVPR 2020.

    MMTM computes channel-wise importance weights for each modality and
    enhances each representation with modality-specific attention derived
    from a shared bottleneck.

    Args:
        dim_a: Feature dimension of modality A.
        dim_b: Feature dimension of modality B.
        reduction: Reduction factor for bottleneck (default: 4).

    Examples:
        >>> layer = MMTMLayer(64, 128)
        >>> a = torch.randn(4, 64)
        >>> b = torch.randn(4, 128)
        >>> a_out, b_out = layer(a, b)
    """

    def __init__(self, dim_a: int, dim_b: int, reduction: int = 4):
        super().__init__()
        bottleneck_size = max(8, (dim_a + dim_b) // reduction)

        # Squeeze step
        self.fc_squeeze = nn.Sequential(
            nn.Linear(dim_a + dim_b, bottleneck_size),
            nn.ReLU(inplace=True),
        )

        # Excitation for each modality
        self.fc_a = nn.Sequential(nn.Linear(bottleneck_size, dim_a), nn.Sigmoid())
        self.fc_b = nn.Sequential(nn.Linear(bottleneck_size, dim_b), nn.Sigmoid())

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([a, b], dim=-1)  # (B, Da+Db)
        z = self.fc_squeeze(joint)

        w_a = self.fc_a(z)  # (B, Da)
        w_b = self.fc_b(z)  # (B, Db)

        return a * w_a, b * w_b


class MMTM(BaseModel):
    """PyHealth multimodal fusion model using MMTM.

    This model:
        1. Embeds two modalities with PyHealth EmbeddingModel
        2. Pools patient-level representations with `get_last_visit`
        3. Applies MMTM fusion module
        4. Predicts labels via a classifier

    MMTM expects EXACTLY TWO feature modalities in the dataset.

    Args:
        dataset: PyHealth SampleDataset with two input modalities.
        embedding_dim: Embedding size for each modality (default: 128).
        reduction: Bottleneck reduction factor for MMTM.

    Examples:
        >>> model = MMTM(dataset, embedding_dim=128)
        >>> results = model(**batch)
        >>> results["loss"].backward()
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        reduction: int = 4,
    ):
        super().__init__(dataset=dataset)

        assert len(self.feature_keys) == 2, \
            f"MMTM requires exactly 2 input modalities, got {self.feature_keys}"

        self.feature_a, self.feature_b = self.feature_keys
        self.label_key = self.label_keys[0]

        # Embedding model shared by both modalities
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # MMTM fusion layer
        self.mmtm = MMTMLayer(
            dim_a=embedding_dim,
            dim_b=embedding_dim,
            reduction=reduction,
        )

        # Classifier after fusion
        output_dim = self.get_output_size()
        self.fc = nn.Linear(embedding_dim * 2, output_dim)

    def _mask_and_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Applies mask for padded timesteps and returns last valid visit."""
        mask = (x.sum(dim=-1) != 0).int()
        return get_last_visit(x, mask)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # ----- 1. Run through embedding model -----
        embedded = self.embedding_model({
            self.feature_a: kwargs[self.feature_a],
            self.feature_b: kwargs[self.feature_b],
        })

        # Extract embeddings
        a = embedded[self.feature_a]  # (B, T, D)
        b = embedded[self.feature_b]  # (B, T, D)

        # ----- 2. Pool last valid visit -----
        a_last = self._mask_and_pool(a)  # (B, D)
        b_last = self._mask_and_pool(b)  # (B, D)

        # ----- 3. Fuse using MMTM -----
        a_fused, b_fused = self.mmtm(a_last, b_last)
        patient_emb = torch.cat([a_fused, b_fused], dim=-1)

        # ----- 4. Classifier -----
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss_fn = self.get_loss_function()
        loss = loss_fn(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }

if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset, get_dataloader

    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "codes": [
                "250.01",     
                "414.01",     
                "272.4",      
            ],
            "procedures": [
                "36.15",      
                "99.04",      
                "88.72",      
            ],
            "label": 1,
        },
        {
            "patient_id": "p0",
            "visit_id": "v1",
            "codes": [
                "518.81",     
                "038.9",      
            ],
            "procedures": [
                "96.71",      
                "93.90",     
            ],
            "label": 0,
        },
    ]


    # Use simple processors for clarity
    dataset = SampleDataset(
        samples=samples,
        input_schema={
            "codes": "sequence",
            "procedures": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="mmtm_example",
    )

    loader = get_dataloader(dataset, batch_size=2)

    model = MMTM(dataset, embedding_dim=64)

    batch = next(iter(loader))
    results = model(**batch)

    print("y_prob:", results["y_prob"])
    print("loss:", results["loss"])

    results["loss"].backward()
