from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.utils import get_last_visit
from pyhealth.processors import (
    MultiHotProcessor,
    SequenceProcessor,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)
from pyhealth.processors.base_processor import FeatureProcessor


class RMSNorm(nn.Module):
    """Root mean square layer normalization (paper ref 62)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * rms * self.weight


class MambaBlock(nn.Module):
    """Single Mamba (SSM) block: RMSNorm -> expand -> conv -> SiLU -> SSM, gate -> residual.

    Paper Appendix C.1: input normalized, two branches (SSM path and gate), residual.
    """

    def __init__(
        self,
        d_model: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        d_inner: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.d_inner = d_inner if d_inner is not None else 2 * d_model

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, conv_kernel, padding=conv_kernel - 1, groups=1)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # SSM parameters (diagonal, per channel; fixed for stable training)
        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, state_size) * 0.5 + 0.5))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        self.B_param = nn.Parameter(torch.ones(self.d_inner, state_size) * 0.5)
        self.C_param = nn.Parameter(torch.randn(self.d_inner, state_size) * 0.1)

    def _ssm_step(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SSM output via causal convolution with learned kernel (parallel scan)."""
        B, L, D = x.shape
        N = self.state_size
        device = x.device

        A = -torch.exp(self.A_log.float())
        dt = torch.sigmoid(self.dt).unsqueeze(-1)
        A_bar = torch.exp(dt * A)
        B_bar = (torch.exp(dt * A) - 1) / (A + 1e-8) * self.B_param
        C = self.C_param

        # Kernel K[d, l] = sum_n C[d,n] * A_bar[d,n]^l * B_bar[d,n]
        arange_L = torch.arange(L, device=device, dtype=x.dtype)
        A_pow = A_bar.unsqueeze(-1).pow(arange_L.view(1, 1, -1))
        K = (C.unsqueeze(-1) * B_bar.unsqueeze(-1) * A_pow).sum(1)
        K = torch.flip(K, dims=[1])
        weight = K.unsqueeze(1)
        x_conv = x.permute(0, 2, 1)
        x_padded = torch.nn.functional.pad(x_conv, (L - 1, 0), value=0)
        out = torch.nn.functional.conv1d(x_padded, weight, groups=D)
        out = out[:, :, :L].permute(0, 2, 1)
        out = out + x * self.D.unsqueeze(0).unsqueeze(0)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_ssm, gate = xz.chunk(2, dim=-1)
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = self.conv1d(x_ssm)
        if x_ssm.size(-1) > L:
            x_ssm = x_ssm[:, :, :L]
        x_ssm = x_ssm.permute(0, 2, 1)
        x_ssm = torch.nn.functional.silu(x_ssm)
        x_ssm = self._ssm_step(x_ssm)
        out = x_ssm * torch.nn.functional.silu(gate)
        out = self.out_proj(out)
        return residual + out


class EHRMamba(BaseModel):
    """EHRMAMBA: Mamba-based foundation model for EHR (clinical prediction).

    Paper: EHRMAMBA: Towards Generalizable and Scalable Foundation Models for
    Electronic Health Records (arxiv 2405.14567). Uses Mamba (SSM) for linear
    complexity in sequence length; supports long EHR sequences.

    Args:
        dataset: SampleDataset for token/embedding setup.
        embedding_dim: Embedding and hidden dimension. Default 128.
        num_layers: Number of Mamba blocks. Default 2.
        state_size: SSM state size per channel. Default 16.
        conv_kernel: Causal conv kernel size in block. Default 4.
        dropout: Dropout before classification head. Default 0.1.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        num_layers: int = 2,
        state_size: int = 16,
        conv_kernel: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.dropout_rate = dropout

        assert len(self.label_keys) == 1, "EHRMamba supports single label key only"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            k: self.dataset.input_processors[k] for k in self.feature_keys
        }

        self.blocks = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.blocks[feature_key] = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=embedding_dim,
                        state_size=state_size,
                        conv_kernel=conv_kernel,
                    )
                    for _ in range(num_layers)
                ]
            )

        output_size = self.get_output_size()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(self.feature_keys) * embedding_dim, output_size)

    @staticmethod
    def _split_temporal(feature: Any) -> Tuple[Optional[torch.Tensor], Any]:
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, (SequenceProcessor, StageNetProcessor)):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _create_mask(self, feature_key: str, value: torch.Tensor) -> torch.Tensor:
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            mask = value != 0
        elif isinstance(processor, StageNetProcessor):
            mask = torch.any(value != 0, dim=-1) if value.dim() >= 3 else value != 0
        elif isinstance(processor, (TimeseriesProcessor, StageNetTensorProcessor)):
            if value.dim() >= 3:
                mask = torch.any(torch.abs(value) > 0, dim=-1)
            elif value.dim() == 2:
                mask = torch.any(torch.abs(value) > 0, dim=-1, keepdim=True)
            else:
                mask = torch.ones(value.size(0), 1, dtype=torch.bool, device=value.device)
        elif isinstance(processor, (TensorProcessor, MultiHotProcessor)):
            mask = torch.ones(value.size(0), 1, dtype=torch.bool, device=value.device)
        else:
            mask = torch.any(value != 0, dim=-1) if value.dim() >= 2 else torch.ones(value.size(0), 1, dtype=torch.bool, device=value.device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(1)
        mask = mask.bool()
        if mask.dim() == 2:
            invalid = ~mask.any(dim=1)
            if invalid.any():
                mask[invalid, 0] = True
        return mask

    @staticmethod
    def _pool_embedding(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_emb = []
        embedding_inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor
            masks[feature_key] = self._create_mask(feature_key, value_tensor)

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key].to(self.device)
            mask = masks[feature_key].to(self.device)
            x = self._pool_embedding(x)
            for blk in self.blocks[feature_key]:
                x = blk(x)
            last_h = get_last_visit(x, mask)
            patient_emb.append(last_h)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(self.dropout(patient_emb))
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {"patient_id": "p0", "visit_id": "v0", "diagnoses": ["A", "B"], "procedures": ["X"], "label": 1},
        {"patient_id": "p1", "visit_id": "v0", "diagnoses": ["C"], "procedures": ["Y", "Z"], "label": 0},
    ]
    input_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
        "diagnoses": "sequence",
        "procedures": "sequence",
    }
    output_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {"label": "binary"}
    dataset = create_sample_dataset(samples=samples, input_schema=input_schema, output_schema=output_schema, dataset_name="test")
    model = EHRMamba(dataset=dataset, embedding_dim=64, num_layers=2)
    loader = get_dataloader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    out = model(**batch)
    print("keys:", sorted(out.keys()))
    out["loss"].backward()
