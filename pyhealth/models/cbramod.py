"""CBraMod criss-cross EEG foundation model and fine-tuning heads.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC

References:
    Wang, J. et al. CBraMod: A Criss-Cross Brain Foundation Model for EEG
    Decoding. ICLR 2025. https://arxiv.org/abs/2412.07236
    https://openreview.net/forum?id=NPNUHgHF2w

    Code (Apache-2.0): https://github.com/wjq-learning/CBraMod

    Pretrained weights (TUEG, ~27K h):
    https://huggingface.co/weighting666/CBraMod ``pretrained_weights.pth``.

Architecture ported from ``models/cbramod.py`` and ``models/criss_cross_transformer.py``.
Fine-tuning heads follow ``models/model_for_{tuev,tuab,chb}.py``.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "weighting666/CBraMod"
DEFAULT_HF_FILENAME = "pretrained_weights.pth"


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerEncoder(nn.Module):
    """Stack of criss-cross encoder layers (from CBraMod ``criss_cross_transformer``)."""

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
    ) -> Tensor:
        del src_key_padding_mask, is_causal
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    """Criss-cross self-attention over channels and patches."""

    __constants__: ClassVar[list[str]] = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {}
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        del is_causal
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, : patch_size // 2]
        xt = x[:, :, :, patch_size // 2 :]
        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz * ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(
            xs,
            xs,
            xs,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size // 2).transpose(1, 2)
        xt = self.self_attn_t(
            xt,
            xt,
            xt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size // 2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class PatchEmbedding(nn.Module):
    """1 s patch embedding with spectral and depthwise positional branches."""

    def __init__(self, in_dim: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 49),
                stride=(1, 25),
                padding=(0, 24),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(
            bz, ch_num, patch_num, self.d_model
        )

        flat = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(flat, dim=-1, norm="forward")
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        positional = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional = positional.permute(0, 2, 3, 1)
        return patch_emb + positional


class CBraModBackbone(nn.Module):
    """CBraMod encoder (``models/cbramod.py``)."""

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.proj_out: nn.Module = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(_weights_init)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        return self.proj_out(feats)


def build_classifier_head(
    classifier: str,
    n_channels: int,
    n_patches: int,
    d_model: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """Build a downstream head matching the CBraMod reference scripts."""
    flat = n_channels * n_patches * d_model
    if classifier == "avgpooling_patch_reps":
        return nn.Sequential(
            Rearrange("b c s d -> b d c s"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(d_model, num_classes),
        )
    if classifier == "all_patch_reps_onelayer":
        return nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(flat, num_classes),
        )
    if classifier == "all_patch_reps_twolayer":
        return nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(flat, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
    if classifier == "all_patch_reps":
        return nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(flat, n_patches * d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(n_patches * d_model, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
    raise ValueError(
        f"Unknown classifier {classifier!r}. Expected one of "
        "'all_patch_reps', 'all_patch_reps_twolayer', "
        "'all_patch_reps_onelayer', 'avgpooling_patch_reps'."
    )


class CBraMod(BaseModel):
    """CBraMod foundation model with paper fine-tuning heads for EEG decoding.

    Expects ``signal`` shaped ``(batch, n_channels, n_time)`` at 200 Hz with
    ``n_time`` divisible by ``patch_size`` (default 200, i.e. 1 s patches).
    Inputs are divided by ``input_scale`` (default 100, matching raw µV scaling).

    Args:
        dataset: Training dataset defining label schema and processors.
        n_channels: Number of EEG channels (16 for TUEV/TUAB/CHB-MIT).
        patch_size: Samples per patch (200 at 200 Hz).
        n_patches: Optional fixed patch count (5 for TUEV, 10 for TUAB/CHB).
            Inferred from the first batch when omitted.
        classifier: Downstream head style from the CBraMod code release.
        dropout: Dropout on MLP heads.
        label_smoothing: Used for multiclass cross-entropy when > 0.
        input_scale: Divide inputs by this value (use ``1.0`` when the task
            already applied ``div_by_100`` normalization).
        n_layer: Transformer depth (12 in the paper checkpoint).
        d_model: Token dimension (200).
        dim_feedforward: FFN width (800).
        nhead: Attention heads (8).
        pretrained: ``True`` to load Hugging Face weights, a path string for a
            local checkpoint, or ``False`` to train from scratch.
        hf_repo_id: Hugging Face repository id.
        hf_filename: Checkpoint file name in the repository.
        frozen: If ``True``, freeze the backbone for linear probing.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import CBraMod
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "signal": [[0.0] * 1000 for _ in range(16)], "label": 0},
        ...     {"patient_id": "p1", "visit_id": "v0",
        ...      "signal": [[0.0] * 1000 for _ in range(16)], "label": 1},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     {"signal": "tensor"},
        ...     {"label": "multiclass"},
        ... )
        >>> model = CBraMod(
        ...     dataset=dataset,
        ...     n_patches=5,
        ...     n_layer=2,
        ...     input_scale=1.0,
        ...     pretrained=False,
        ... )
        >>> batch = {"signal": torch.randn(2, 16, 1000), "label": torch.tensor([0, 1])}
        >>> out = model(**batch)
        >>> out["logit"].shape[1] == dataset.output_processors["label"].size()
        True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_channels: int = 16,
        patch_size: int = 200,
        n_patches: int | None = None,
        classifier: str = "all_patch_reps",
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        input_scale: float = 100.0,
        n_layer: int = 12,
        d_model: int = 200,
        dim_feedforward: int = 800,
        nhead: int = 8,
        pretrained: bool | str = False,
        hf_repo_id: str = DEFAULT_HF_REPO,
        hf_filename: str = DEFAULT_HF_FILENAME,
        frozen: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset=dataset, **kwargs)
        self.n_channels = n_channels
        self.patch_size = patch_size
        self._n_patches = n_patches
        self.classifier_name = classifier
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.input_scale = input_scale
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename

        self.backbone = CBraModBackbone(
            in_dim=patch_size,
            out_dim=d_model,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            n_layer=n_layer,
            nhead=nhead,
        )
        self.backbone.proj_out = nn.Identity()

        self.classifier: nn.Module | None = None
        if n_patches is not None:
            self._build_classifier(n_patches)

        if pretrained:
            path = None if pretrained is True else str(pretrained)
            self.load_pretrained_weights(checkpoint_path=path)

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _output_classes(self) -> int:
        return self.get_output_size()

    def _build_classifier(self, n_patches: int) -> None:
        self._n_patches = n_patches
        self.classifier = build_classifier_head(
            self.classifier_name,
            self.n_channels,
            n_patches,
            200,
            self._output_classes(),
            self.dropout,
        )

    def _prepare_patches(self, signal: Tensor) -> Tensor:
        x = signal.to(self.device).float()
        if self.input_scale != 1.0:
            x = x / self.input_scale
        batch, channels, length = x.shape
        if length % self.patch_size != 0:
            raise ValueError(
                f"signal length {length} is not divisible by patch_size "
                f"{self.patch_size}"
            )
        if channels != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {channels}"
            )
        n_patches = length // self.patch_size
        if self.classifier is None:
            self._build_classifier(n_patches)
        elif self._n_patches != n_patches:
            raise ValueError(
                f"expected {self._n_patches} patches from constructor, "
                f"got {n_patches} from signal length"
            )
        return x.view(batch, channels, n_patches, self.patch_size)

    def encode(self, signal: Tensor) -> Tensor:
        """Return patch tokens ``(batch, channels, patches, d_model)``.

        Args:
            signal: ``(batch, n_channels, n_time)`` EEG window.

        Returns:
            Backbone token tensor for CELM-style aggregation.
        """
        patches = self._prepare_patches(signal)
        return self.backbone(patches)

    @staticmethod
    def pool(tokens: Tensor) -> Tensor:
        """Mean-pool patch tokens over channels and time patches."""
        return tokens.mean(dim=(1, 2))

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Parameters of the criss-cross encoder (for multi-lr fine-tuning)."""
        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the downstream classifier head."""
        if self.classifier is None:
            return []
        return list(self.classifier.parameters())

    def load_pretrained_weights(
        self,
        checkpoint_path: str | None = None,
        strict: bool = False,
        map_location: str | None = None,
    ) -> None:
        """Load CBraMod backbone weights from disk or Hugging Face.

        Args:
            checkpoint_path: Local ``.pth`` path. When ``None``, downloads
                ``hf_filename`` from ``hf_repo_id``.
            strict: Passed to ``load_state_dict``.
            map_location: Device for ``torch.load``.

        Examples:
            >>> # model = CBraMod(dataset=dataset, pretrained=False, n_patches=5, n_layer=2)
            >>> # model.load_pretrained_weights("/path/to/pretrained_weights.pth")
        """
        if map_location is None:
            map_location = str(self.device)

        if checkpoint_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required to download CBraMod weights. "
                    "Install it with: pip install huggingface_hub"
                ) from exc
            checkpoint_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.hf_filename,
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        cleaned: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            name = key.removeprefix("module.")
            name = name.removeprefix("backbone.")
            cleaned[name] = value

        incompatible = self.backbone.load_state_dict(cleaned, strict=strict)
        if incompatible.missing_keys:
            logger.info("Missing keys when loading CBraMod: %s", incompatible.missing_keys)
        if incompatible.unexpected_keys:
            logger.info(
                "Unexpected keys when loading CBraMod: %s",
                incompatible.unexpected_keys,
            )
        logger.info("Loaded CBraMod weights from %s", checkpoint_path)

    def _compute_loss(
        self,
        logits: Tensor,
        y_true: Tensor,
        label_key: str,
    ) -> Tensor:
        mode = self._resolve_mode(self.dataset.output_schema[label_key])
        target = y_true
        if target.dim() > 1 and target.size(-1) == 1:
            target = target.squeeze(-1)
        if mode == "multiclass" and self.label_smoothing > 0:
            return F.cross_entropy(
                logits,
                target.long(),
                label_smoothing=self.label_smoothing,
            )
        loss_fn = self.get_loss_function()
        return loss_fn(logits, y_true)

    def forward(self, **kwargs: Any) -> dict[str, Tensor]:
        """Forward pass returning loss, probabilities, and logits."""
        signal = kwargs.get("signal")
        if signal is None:
            raise ValueError("'signal' must be provided in inputs")

        tokens = self.encode(signal)
        assert self.classifier is not None
        logits = self.classifier(tokens)

        label_key = self.label_keys[0]
        y_true = kwargs[label_key].to(self.device)
        loss = self._compute_loss(logits, y_true, label_key)
        y_prob = self.prepare_y_prob(logits)

        results: dict[str, Tensor] = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed"):
            results["patch_tokens"] = tokens
            results["embed"] = self.pool(tokens)
        return results
