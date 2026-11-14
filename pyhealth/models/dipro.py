"""DiPro: disease progression from serial chest X-rays and the EHR around them.

Paper:
    Chen Liu, Wenfang Yao, Kejing Yin, William K. Cheung, and Jing Qin.
    "Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement
    and Multiscale Alignment." NeurIPS 2025. https://arxiv.org/abs/2510.11112

Adapted from DiPro (commit 64b12ec, ``models/model.py`` and
``models/Component.py``, https://github.com/Chenliu-svg/DiPro), MIT License,
Copyright (c) 2025 LIU Chen @ The Hong Kong Polytechnic University.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

_NUM_STATUSES = 3  # worsened, improved, no change
_NOT_MENTIONED = _NUM_STATUSES  # DiPro's FILL_VALUE: ignored by the losses
_NUM_HEADS = 8
_BACKBONES = ("resnet18", "resnet34", "resnet50")


def _with_first_if_empty(mask: torch.Tensor) -> torch.Tensor:
    """``mask`` ``[B, N]``, with position 0 set for a row that has none set.

    Attention keys are the real positions (True). A sample without one keeps
    its first position, so its attention has a key, and no padding beyond the
    first position, which depends on the rest of the batch, can change it.
    """
    first = torch.zeros_like(mask)
    first[:, 0] = True
    return mask | (first & ~mask.any(dim=1, keepdim=True))


def _attended_pairs(pair_mask: torch.Tensor) -> torch.Tensor:
    """The pairs that attention may read, ``[B, P]``: the real pairs, or the
    first (learned padding) pair of a sample without one."""
    return _with_first_if_empty(pair_mask)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` where ``mask`` (broadcast to them) is True, or 0."""
    mask = mask.expand_as(values).to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1)


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added to a sequence, then dropout.

    The same encoding as ``PositionalEncoding`` in ``pyhealth/models/biot.py``,
    copied so this model is self-contained.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class _OrderAwareProjection(nn.Module):
    """Projects pairs of region features, in their order and reversed.

    Args:
        feature_size: Size of one region's feature.
        dropout_rate: Dropout rate.
        concat_reversed_pair: See :class:`DiPro`.
    """

    def __init__(
        self, feature_size: int, dropout_rate: float, concat_reversed_pair: bool
    ) -> None:
        super().__init__()
        self.concat_reversed_pair = concat_reversed_pair
        self.layers = nn.Sequential(
            nn.Linear(2 * feature_size, 4 * feature_size),
            nn.GELU(),
            nn.Linear(4 * feature_size, feature_size),
        )
        self.norm = nn.BatchNorm1d(feature_size)
        self.dropout = nn.Dropout(dropout_rate)

    def _project(self, pairs: torch.Tensor) -> torch.Tensor:
        """Projects ``[B, P, R, 2D]`` to ``[B, P, R, D]``."""
        output = self.layers(pairs.flatten(0, 2))
        # BatchNorm cannot train on one row: a last batch of a single sample
        # with one region per image.
        if output.size(0) > 1 or not self.training:
            output = self.norm(output)
        return self.dropout(output).view(*pairs.shape[:3], -1)

    def forward(self, pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the projections of ``pairs`` and of the reversed pairs."""
        first, second = pairs.chunk(2, dim=-1)
        if self.concat_reversed_pair:
            reversed_pairs = torch.cat([second, first], dim=-1)
        else:
            # DiPro's layout: the two features interleaved, entry by entry.
            reversed_pairs = torch.stack([second, first], dim=-1).flatten(-2)
        return self._project(pairs), self._project(reversed_pairs)


class _LongitudinalCXREncoder(nn.Module):
    """The STD and PAE modules: splits region pairs into static and dynamic.

    Args:
        feature_size: Size of the static and dynamic features.
        num_diseases: Number of diseases.
        dropout_rate: Dropout rate.
        backbone: ResNet name.
        pretrained: Whether to load ImageNet weights.
        freeze_backbone: Whether to freeze the ResNet.
        concat_reversed_pair: See :class:`DiPro`.
    """

    def __init__(
        self,
        feature_size: int,
        num_diseases: int,
        dropout_rate: float,
        backbone: str,
        pretrained: bool,
        freeze_backbone: bool,
        concat_reversed_pair: bool,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models

        weights = None
        if pretrained:
            weights = tv_models.get_model_weights(backbone).IMAGENET1K_V1
        self.resnet = getattr(tv_models, backbone)(weights=weights)
        if freeze_backbone:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
        backbone_size = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.feature_adapter = nn.Sequential(
            nn.Linear(backbone_size, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.static_proj = _OrderAwareProjection(
            feature_size, dropout_rate, concat_reversed_pair
        )
        self.dynamic_proj = _OrderAwareProjection(
            feature_size, dropout_rate, concat_reversed_pair
        )
        self.progression_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.ReLU(),
                nn.Linear(feature_size, _NUM_STATUSES),
            )
            for _ in range(num_diseases)
        )

    def _classify(self, dynamic: torch.Tensor) -> torch.Tensor:
        """Returns the logits of each disease, ``[B, P, R, K, 3]``."""
        return torch.stack([head(dynamic) for head in self.progression_heads], dim=3)

    def forward(
        self, regions: torch.Tensor, pair_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Encodes the regions of a series of images.

        Args:
            regions: ``[B, T, R, C, H, W]``.
            pair_mask: True for real pairs, ``[B, T - 1]``. The losses ignore
                the other pairs. None means every pair is real.

        Returns:
            ``static`` and ``dynamic`` features ``[B, P, R, D]`` with ``P = T -
            1``, the region logits ``forward_logits`` and ``reversed_logits``,
            and the unweighted ``orthogonal``, ``consistency`` and
            ``static_order`` losses.
        """
        B, T, R = regions.shape[:3]
        features = self.feature_adapter(self.resnet(regions.flatten(0, 2)))
        features = features.view(B, T, R, -1)
        pairs = torch.cat([features[:, :-1], features[:, 1:]], dim=-1)

        static, static_reversed = self.static_proj(pairs)
        dynamic, dynamic_reversed = self.dynamic_proj(pairs)

        if pair_mask is None:
            pair_mask = torch.ones(B, T - 1, dtype=torch.bool, device=regions.device)
        real = pair_mask[:, :, None, None]  # [B, P, 1, 1]
        similarity = F.cosine_similarity(static, dynamic, dim=-1, eps=1e-12)
        consistency = static.sum() * 0.0
        if T > 2:  # needs two pairs, so it is zero for a pair of images
            consistency = _masked_mean(
                (static[:, :-1] - static[:, 1:]) ** 2, real[:, :-1] & real[:, 1:]
            )
        return {
            "static": static,
            "dynamic": dynamic,
            "forward_logits": self._classify(dynamic),
            "reversed_logits": self._classify(dynamic_reversed),
            "orthogonal": _masked_mean(similarity**2, real[..., 0]),
            "consistency": consistency,
            "static_order": _masked_mean(
                (static - static_reversed.detach()) ** 2, real
            ),
        }


class _EHRIntervalEncoder(nn.Module):
    """Pools the EHR of each image interval with time-aware cross-attention."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.attention = nn.MultiheadAttention(d_model, _NUM_HEADS, batch_first=True)

    def forward(
        self,
        ehr: torch.Tensor,
        ehr_times: torch.Tensor,
        ehr_mask: torch.Tensor,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        """Attends to the EHR inside each interval, favoring its center.

        Args:
            ehr: ``[B, L, D]``.
            ehr_times: Hours of the rows, ``[B, L]``.
            ehr_mask: True for real rows, ``[B, L]``.
            intervals: Start and end hours, ``[B, P, 2]``.

        Returns:
            ``[B, P, L, D]``.
        """
        B, L, _ = ehr.shape
        P = intervals.size(1)
        times = ehr_times[:, None].expand(B, P, L)
        start = intervals[..., 0, None].expand(B, P, L)
        end = intervals[..., 1, None].expand(B, P, L)
        time_features = torch.stack(
            [times - start, end - times, torch.sigmoid((times - start) * (end - times))],
            dim=-1,
        )
        query = self.time_encoder(time_features).flatten(0, 1)

        inside = (times >= start) & (times <= end) & ehr_mask[:, None]
        # Added to the attention scores: -inf removes rows outside the interval.
        scores = torch.where(
            inside, -(times - (start + end) / 2).abs(), float("-inf")
        )
        # With no row inside an interval, -inf everywhere would give NaN, so
        # the interval attends uniformly to the sample's real rows instead, or
        # to its first row if it has none. Padding rows never get a weight.
        fallback = torch.where(
            _with_first_if_empty(ehr_mask)[:, None], 0.0, float("-inf")
        )
        scores = torch.where(inside.any(-1, keepdim=True), scores, fallback)
        attention_mask = scores[:, :, None].expand(B, P, L, L).flatten(0, 1)
        attention_mask = attention_mask.repeat_interleave(_NUM_HEADS, dim=0)

        repeated_ehr = ehr.repeat_interleave(P, dim=0)
        output, _ = self.attention(
            query, repeated_ehr, repeated_ehr, attn_mask=attention_mask
        )
        return output.view(B, P, L, -1)


class _RegionAwareAttention(nn.Module):
    """Lets each region attend to the other regions and to its interval's EHR."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.region_proj = nn.Linear(d_model, d_model * 3)
        self.ehr_proj = nn.Linear(d_model, d_model * 2)
        self.attention = nn.MultiheadAttention(d_model, _NUM_HEADS, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, regions: torch.Tensor, ehr: torch.Tensor, ehr_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fuses ``[N, R, D]`` regions with ``[N, L, D]`` EHR; True is real."""
        region_query, region_key, region_value = self.region_proj(regions).chunk(
            3, dim=-1
        )
        ehr_key, ehr_value = self.ehr_proj(ehr).chunk(2, dim=-1)
        padding = torch.cat(
            [torch.zeros_like(regions[..., 0], dtype=torch.bool), ~ehr_mask], dim=1
        )
        output, _ = self.attention(
            region_query,
            torch.cat([region_key, ehr_key], dim=1),
            torch.cat([region_value, ehr_value], dim=1),
            key_padding_mask=padding,
        )
        return self.norm(output + regions)


class _MultiscaleFusion(nn.Module):
    """The MMF module: local, then global fusion of dynamic images and EHR.

    Args:
        num_tokens: Number of class tokens, one per disease.
        ehr_dim: Features per EHR row.
        d_model: Feature size.
        num_transformer_layers: Twice the number of region attention layers,
            as in DiPro, where ``num_layers // 2`` of them are built.
        num_fusion_layers: Number of self-attention layers.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        num_tokens: int,
        ehr_dim: int,
        d_model: int,
        num_transformer_layers: int,
        num_fusion_layers: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, d_model))
        self.ehr_cls_token = nn.Parameter(torch.randn(1, d_model))
        self.ehr_input = nn.Sequential(
            nn.Linear(ehr_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_rate),
        )
        self.ehr_position = _PositionalEncoding(d_model, dropout_rate)
        self.ehr_global_encoder = self._transformer(d_model, 1, dropout_rate)
        self.ehr_interval_encoder = _EHRIntervalEncoder(d_model)
        self.region_attention = nn.ModuleList(
            _RegionAwareAttention(d_model) for _ in range(num_transformer_layers // 2)
        )
        self.ehr_cross_attention = nn.MultiheadAttention(
            d_model, _NUM_HEADS, batch_first=True
        )
        self.ehr_cross_norm = nn.LayerNorm(d_model)
        self.fusion = self._transformer(d_model, num_fusion_layers, dropout_rate)

    @staticmethod
    def _transformer(
        d_model: int, num_layers: int, dropout_rate: float
    ) -> nn.TransformerEncoder:
        """Builds a Transformer that always returns its input's length."""
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=_NUM_HEADS,
            dim_feedforward=4 * d_model,
            dropout=dropout_rate,
            batch_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def _encode_ehr(self, ehr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encodes the whole series, ``[B, L, ehr_dim]`` to ``[B, L, D]``."""
        tokens = self.ehr_position(self.ehr_input(ehr))
        cls = self.ehr_cls_token.expand(ehr.size(0), -1, -1)
        padding = torch.cat(
            [torch.zeros_like(mask[:, :1]), ~mask], dim=1
        )
        return self.ehr_global_encoder(
            torch.cat([cls, tokens], dim=1), src_key_padding_mask=padding
        )[:, 1:]

    def forward(
        self,
        dynamic: torch.Tensor,
        pair_mask: torch.Tensor,
        ehr: torch.Tensor,
        ehr_mask: torch.Tensor,
        ehr_times: torch.Tensor,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        """Fuses the dynamic features with the EHR.

        Args:
            dynamic: ``[B, P, R, D]``.
            pair_mask: True for real pairs, ``[B, P]``.
            ehr: ``[B, L, ehr_dim]``.
            ehr_mask: True for real rows, ``[B, L]``.
            ehr_times: Hours of the rows, ``[B, L]``.
            intervals: Start and end hours, ``[B, P, 2]``.

        Returns:
            The disease tokens, the regions and the EHR, ``[B, K + P * R + L, D]``.
        """
        B, P, R, _ = dynamic.shape
        ehr = self._encode_ehr(ehr, ehr_mask)

        # Local: each pair's regions meet the EHR of its own interval.
        interval_ehr = self.ehr_interval_encoder(ehr, ehr_times, ehr_mask, intervals)
        regions = dynamic.flatten(0, 1)
        repeated_ehr = interval_ehr.flatten(0, 1)
        repeated_mask = ehr_mask.repeat_interleave(P, dim=0)
        for layer in self.region_attention:
            regions = layer(regions, repeated_ehr, repeated_mask)
        regions = rearrange(regions, "(b p) r d -> b (p r) d", b=B)

        # Global: the whole EHR attends to the fused regions of real pairs.
        region_padding = ~_attended_pairs(pair_mask).repeat_interleave(R, dim=1)
        attended, _ = self.ehr_cross_attention(
            ehr, regions, regions, key_padding_mask=region_padding
        )
        ehr = self.ehr_cross_norm(ehr + attended)

        tokens = self.tokens.expand(B, -1, -1)
        padding = torch.cat(
            [
                torch.zeros_like(tokens[..., 0], dtype=torch.bool),
                ~pair_mask.repeat_interleave(R, dim=1),
                ~ehr_mask,
            ],
            dim=1,
        )
        return self.fusion(
            torch.cat([tokens, regions, ehr], dim=1), src_key_padding_mask=padding
        )


class _GroupEmbedding(nn.Module):
    """Group embedding ``softmax(XW)G`` of the static tokens."""

    def __init__(self, d_model: int, num_groups: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_model, num_groups))
        self.G = nn.Parameter(torch.empty(num_groups, d_model))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.G)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps ``[B, N, D]`` to ``[B, N, D]``."""
        return F.softmax(x @ self.W, dim=-1) @ self.G


class DiPro(BaseModel):
    """DiPro model for disease progression between chest X-rays (Liu et al., 2025).

    Each image pair is reduced to anatomical regions. The STD module splits
    each pair of regions into a static and a dynamic feature, the PAE module
    makes them react properly to reversing the pair, and the MMF module fuses
    the dynamic features with the EHR inside the image interval, then with the
    whole series, then with the static features and demographics. One head per
    disease predicts worsened, improved or no change.

    Use it with :class:`~pyhealth.tasks.DiseaseProgressionMIMIC4`. The
    defaults are those of DiPro's ``scripts/experiment.sh`` for this task,
    which agree with the paper's final loss weights: ``lambda_pred`` 2,
    ``lambda_orthogonal`` 1, ``lambda_dynamic_order`` (the paper's
    :math:`\\lambda_{PAE}`) 2, and no consistency term, since that needs three
    images. DiPro trains them with AdamW at ``1.6e-4``, batches of 8, 4
    accumulated steps, and cosine decay over 100 epochs;
    :class:`~pyhealth.trainer.Trainer` has neither of the last two. Call
    :meth:`set_class_weights` on the training split to use DiPro's weighted
    losses.

    Inputs, from the dataset's fields:

    - ``ehr``: ``[B, L, ehr_dim]``. The dataset must have this field, since
      ``ehr_dim`` comes from its processor. Rows after a sample's last nonzero
      row are padding, and so is a sample that is all zeros, so a sample
      without EHR is given as all-zero rows. A batch may also leave ``ehr``
      out, and the model then works from the images alone.
    - ``cxr``: a list with one ``(images, hours, tag)`` per sample, as the
      ``time_image`` (``images`` ``[T, C, H, W]``, one region per image) and
      ``time_region_image`` (``[T, R, C, H, W]``) processors give it. An
      entry that is None, or a missing image at the end of a series, is
      replaced by learned padding. A pair with a padded image is masked out
      of every attention and loss, so a sample's predictions do not depend
      on how many images the other samples in its batch have. A sample with
      no real pair keeps its first, padded, pair as an attention key.
    - ``first_ehr_hour``: the hour of each sample's first EHR row.
    - ``demographics``: ``[B, n]``, optional.

    Labels:

    - ``progression``: ``[B, 3 * num_diseases]``, three per disease. A disease
      with none of its three set is not mentioned, and the losses ignore it.
    - ``region_progression``: the region labels of the PAE loss, ``[B, R,
      num_diseases]`` with 3 for not mentioned. It is only read for that loss,
      and only with region crops. With whole images, ``progression`` is used
      for that loss instead, as the single region is the whole image.

    Whole images (``time_image``, one region per image) are a PyHealth
    addition. Results in that mode are not those of DiPro as published,
    which always uses the region crops (``region_crops=True`` on the task).

    ``forward`` returns ``logit`` ``[B, 3 * num_diseases]``, and ``y_prob``,
    a softmax over the three statuses of each disease, so it is not a
    multilabel sigmoid. With labels, it also returns ``y_true``, ``loss``,
    and the unweighted ``prediction_loss``, ``orthogonal_loss``,
    ``consistency_loss``, ``static_order_loss`` and ``dynamic_order_loss``.

    Args:
        dataset: Sample dataset with ``ehr``, ``cxr`` and ``progression``.
        num_diseases: Number of diseases.
        d_model: Feature size; a multiple of 8. DiPro uses 256 here.
        num_transformer_layers: Twice the number of region attention layers.
        group_num: Number of groups of the group embedding, below ``d_model``.
        num_fusion_layers: Number of self-attention layers of the fusion.
        dropout_rate: Dropout rate.
        backbone: ``"resnet50"`` as in DiPro, or ``"resnet18"`` or
            ``"resnet34"``.
        pretrained: Whether to load ImageNet weights, which downloads them.
        freeze_backbone: Whether to freeze the ResNet.
        lambda_pred: Weight of the prediction loss.
        lambda_orthogonal: Weight of the orthogonality loss.
        lambda_consistency: Weight of the consistency of static features.
        lambda_static_order: Weight of the static reversal loss.
        lambda_dynamic_order: Weight of the region progression loss.
        concat_reversed_pair: The default False reproduces DiPro's code, in
            which a reversed pair is the two features interleaved entry by
            entry. True swaps the two halves instead, which is what the paper
            describes.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import torch
        >>> from PIL import Image
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import DiPro
        >>> folder = tempfile.TemporaryDirectory()
        >>> path = str(Path(folder.name) / "cxr.png")
        >>> Image.new("L", (32, 32), 128).save(path)
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{index}",
        ...         "ehr": torch.rand(9, 4).tolist(),
        ...         "first_ehr_hour": 1,
        ...         "cxr": ([path, path], [1.5, 9.5]),
        ...         "progression": ["a: worsened", "b: no change"],
        ...     }
        ...     for index in range(2)
        ... ]
        >>> labels = [f"{d}: {s}" for d in "ab" for s in
        ...           ("worsened", "improved", "no change")]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "ehr": "tensor",
        ...         "cxr": ("time_image", {"image_size": 32, "mode": "RGB"}),
        ...     },
        ...     output_schema={"progression": ("multilabel", {"labels": labels})},
        ... )
        >>> model = DiPro(
        ...     dataset,
        ...     num_diseases=2,
        ...     d_model=16,
        ...     backbone="resnet18",
        ...     pretrained=False,
        ... )
        >>> output = model(**next(iter(get_dataloader(dataset, batch_size=2))))
        >>> output["y_prob"].shape, output["loss"].requires_grad
        (torch.Size([2, 6]), True)
        >>> folder.cleanup()
    """

    def __init__(
        self,
        dataset: SampleDataset,
        num_diseases: int = 7,
        d_model: int = 256,
        num_transformer_layers: int = 2,
        group_num: int = 8,
        num_fusion_layers: int = 1,
        dropout_rate: float = 0.1,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        lambda_pred: float = 2.0,
        lambda_orthogonal: float = 1.0,
        lambda_consistency: float = 0.0,
        lambda_static_order: float = 0.5,
        lambda_dynamic_order: float = 2.0,
        concat_reversed_pair: bool = False,
    ) -> None:
        super().__init__(dataset=dataset)
        for field in ("ehr", "cxr"):
            if field not in self.feature_keys:
                raise ValueError(f"DiPro needs the input field '{field}'.")
        if self.label_keys != ["progression"] or self.mode != "multilabel":
            raise ValueError("DiPro needs the single multilabel label 'progression'.")
        output_size = self.label_processor.output_size()
        if output_size != _NUM_STATUSES * num_diseases:
            raise ValueError(
                f"Expected {_NUM_STATUSES} labels for each of {num_diseases} "
                f"diseases, got {output_size} labels."
            )
        if backbone not in _BACKBONES:
            raise ValueError(f"backbone must be one of {_BACKBONES}.")
        if d_model % _NUM_HEADS:
            raise ValueError(f"d_model must be a multiple of {_NUM_HEADS}.")
        if group_num >= d_model:
            raise ValueError("group_num must be smaller than d_model.")
        cxr_processor = self.dataset.input_processors["cxr"]
        if cxr_processor.size() != 3:
            raise ValueError("DiPro's ResNet needs images with three channels.")

        self.num_diseases = num_diseases
        self.num_regions = getattr(cxr_processor, "num_regions", 1)
        self.ehr_dim = self.dataset.input_processors["ehr"].size()
        self.lambda_pred = lambda_pred
        self.lambda_orthogonal = lambda_orthogonal
        self.lambda_consistency = lambda_consistency
        self.lambda_static_order = lambda_static_order
        self.lambda_dynamic_order = lambda_dynamic_order
        self.register_buffer("class_weights", torch.ones(num_diseases, _NUM_STATUSES))

        size = cxr_processor.image_size
        self.region_shape = (self.num_regions, 3, size, size)
        self.region_padding = nn.Parameter(torch.randn(self.region_shape))
        self.cxr_encoder = _LongitudinalCXREncoder(
            feature_size=d_model,
            num_diseases=num_diseases,
            dropout_rate=dropout_rate,
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            concat_reversed_pair=concat_reversed_pair,
        )
        self.fusion = _MultiscaleFusion(
            num_tokens=num_diseases,
            ehr_dim=self.ehr_dim,
            d_model=d_model,
            num_transformer_layers=num_transformer_layers,
            num_fusion_layers=num_fusion_layers,
            dropout_rate=dropout_rate,
        )
        self.group_embedding = _GroupEmbedding(d_model, group_num)
        self.demographic_encoder = None
        if "demographics" in self.feature_keys:
            self.demographic_encoder = nn.Sequential(
                nn.Linear(self.dataset.input_processors["demographics"].size(), d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )
        self.static_attention = nn.MultiheadAttention(
            d_model, _NUM_HEADS, batch_first=True
        )
        self.static_norm = nn.LayerNorm(d_model)
        self.output_layers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_model, _NUM_STATUSES),
            )
            for _ in range(num_diseases)
        )

    def set_class_weights(self, dataset: Iterable[dict[str, Any]]) -> None:
        """Weights each disease's statuses inversely to their frequency.

        Reproduces DiPro's weights, labeled samples of the disease / (3 *
        samples with the status). They apply to the prediction loss and the
        region loss, and are saved with the model. A status that the samples
        never show gets a weight of 1. Without this call, all weights are 1.

        This reads every sample, images included, so it can take a while.

        Args:
            dataset: The training samples. Pass the training split, so that
                validation and test labels do not set the weights.

        Raises:
            ValueError: If ``dataset`` is empty.

        Examples:
            >>> model.set_class_weights(train_dataset)  # doctest: +SKIP
        """
        counts, total = torch.zeros_like(self.class_weights), 0
        for sample in dataset:
            counts += (
                torch.as_tensor(sample["progression"], dtype=counts.dtype)
                .to(counts.device)
                .view_as(counts)
            )
            total += 1
        if total == 0:
            raise ValueError("set_class_weights got an empty dataset.")
        labeled = counts.sum(dim=-1, keepdim=True)
        weights = labeled / (_NUM_STATUSES * counts.clamp(min=1))
        self.class_weights.copy_(torch.where(counts > 0, weights, 1.0))

    def _stack_cxr(
        self, cxr: Sequence[Any] | None, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stacks the images, which the default collate leaves as a list.

        Returns:
            The regions ``[B, T, R, C, H, W]``, with learned padding in place
            of missing images, their hours ``[B, T]``, and a mask ``[B, T]``
            that is True for real images.
        """
        cxr = [None] * batch_size if cxr is None else cxr
        lengths = [0 if sample is None else len(sample[0]) for sample in cxr]
        length = max(2, *lengths)
        regions = self.region_padding.expand(batch_size, length, *self.region_shape)
        regions = regions.clone()
        hours = torch.zeros(batch_size, length, device=self.device)
        mask = torch.zeros(batch_size, length, dtype=torch.bool, device=self.device)
        for row, (sample, count) in enumerate(zip(cxr, lengths)):
            if count == 0:
                continue
            images = sample[0].to(self.device).float()
            if images.dim() == 4:  # whole images: a single region
                images = images.unsqueeze(1)
            regions[row, :count] = images
            hours[row, :count] = sample[1].to(self.device).float()
            mask[row, :count] = True
        return regions, hours, mask

    def _ehr_inputs(
        self, ehr: torch.Tensor | None, first_hour: Any, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the EHR, its mask of real rows, and the hour of each row."""
        if ehr is None:
            ehr = torch.zeros(batch_size, 1, self.ehr_dim)
        ehr = ehr.to(self.device).float()
        nonzero = (ehr != 0).any(dim=-1)
        # Padding comes last; an all-zero row before real ones is a real row.
        mask = nonzero.flip(1).cumsum(1).flip(1) > 0
        first_hour = (
            torch.zeros(batch_size)
            if first_hour is None
            else torch.as_tensor([float(hour) for hour in first_hour])
        )
        rows = torch.arange(ehr.size(1), device=self.device, dtype=ehr.dtype)
        return ehr, mask, first_hour.to(self.device)[:, None] + rows

    def _disease_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Sums the weighted cross-entropy of the diseases.

        Args:
            logits: ``[N, K, 3]``.
            target: ``[N, K]``, with 3 for not mentioned.
        """
        loss = logits.sum() * 0.0
        for disease in range(self.num_diseases):
            mentioned = target[:, disease] != _NOT_MENTIONED
            if mentioned.any():
                loss = loss + F.cross_entropy(
                    logits[mentioned, disease],
                    target[mentioned, disease],
                    weight=self.class_weights[disease],
                )
        return loss

    def _region_loss(
        self,
        encoded: dict[str, torch.Tensor],
        target: torch.Tensor,
        region_labels: Any,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the PAE loss on the pairs and on the reversed pairs.

        Args:
            encoded: Output of the CXR encoder.
            target: Status per disease, ``[B, K]``.
            region_labels: ``region_progression`` of the batch, or None.
            pair_mask: True for real pairs, ``[B, P]``.
        """
        logits = encoded["forward_logits"]
        B, P, R, K, _ = logits.shape
        if R == 1:
            labels = target[:, None, None].expand(B, P, 1, K)
        elif region_labels is None:
            return logits.sum() * 0.0
        else:
            labels = torch.as_tensor(region_labels, device=self.device).long()
            labels = labels.view(B, P, R, K)
        labels = labels.masked_fill(~pair_mask[:, :, None, None], _NOT_MENTIONED)
        # Reversing a pair swaps worsened (0) and improved (1).
        reversed_labels = torch.where(labels < 2, 1 - labels, labels)
        return self._disease_loss(
            torch.cat(
                [logits.flatten(0, 2), encoded["reversed_logits"].flatten(0, 2)]
            ),
            torch.cat([labels.flatten(0, 2), reversed_labels.flatten(0, 2)]),
        )

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Predicts the status of each disease.

        Args:
            **kwargs: A batch, as described in the class docstring.

        Returns:
            ``logit`` and ``y_prob``, and with labels ``y_true``, ``loss`` and
            the unweighted losses.
        """
        ehr, cxr = kwargs.get("ehr"), kwargs.get("cxr")
        if ehr is None and cxr is None:
            raise ValueError("DiPro needs at least one of 'ehr' and 'cxr'.")
        batch_size = len(ehr if cxr is None else cxr)
        regions, hours, image_mask = self._stack_cxr(cxr, batch_size)
        ehr, ehr_mask, ehr_times = self._ehr_inputs(
            ehr, kwargs.get("first_ehr_hour"), batch_size
        )

        pair_mask = image_mask[:, 1:]
        encoded = self.cxr_encoder(regions, pair_mask)
        intervals = torch.stack([hours[:, :-1], hours[:, 1:]], dim=-1)
        fused = self.fusion(
            encoded["dynamic"], pair_mask, ehr, ehr_mask, ehr_times, intervals
        )

        # Static tokens of padded pairs are left out, as in the fusion.
        static = encoded["static"].flatten(1, 2)
        num_regions = encoded["static"].size(2)
        static_padding = ~_attended_pairs(pair_mask).repeat_interleave(
            num_regions, dim=1
        )
        demographics = kwargs.get("demographics")
        if self.demographic_encoder is not None and demographics is not None:
            encoded_demographics = self.demographic_encoder(
                demographics.to(self.device).float()
            )
            static = torch.cat([static, encoded_demographics[:, None]], dim=1)
            static_padding = F.pad(static_padding, (0, 1), value=False)
        static = self.group_embedding(static)
        attended, _ = self.static_attention(
            fused, static, static, key_padding_mask=static_padding
        )
        tokens = self.static_norm(fused + attended)[:, : self.num_diseases]

        logits = torch.stack(
            [
                layer(tokens[:, disease])
                for disease, layer in enumerate(self.output_layers)
            ],
            dim=1,
        )
        results = {
            "logit": logits.flatten(1),
            "y_prob": F.softmax(logits, dim=-1).flatten(1),
        }

        labels = kwargs.get("progression")
        if labels is not None:
            y_true = labels.to(self.device).float()
            per_disease = y_true.view(batch_size, self.num_diseases, _NUM_STATUSES)
            target = per_disease.argmax(dim=-1).masked_fill(
                per_disease.sum(dim=-1) == 0, _NOT_MENTIONED
            )
            prediction_loss = self._disease_loss(logits, target)
            dynamic_order_loss = self._region_loss(
                encoded, target, kwargs.get("region_progression"), pair_mask
            )
            results.update(
                y_true=y_true,
                loss=self.lambda_pred * prediction_loss
                + self.lambda_orthogonal * encoded["orthogonal"]
                + self.lambda_consistency * encoded["consistency"]
                + self.lambda_static_order * encoded["static_order"]
                + self.lambda_dynamic_order * dynamic_order_loss,
                prediction_loss=prediction_loss.detach(),
                orthogonal_loss=encoded["orthogonal"].detach(),
                consistency_loss=encoded["consistency"].detach(),
                static_order_loss=encoded["static_order"].detach(),
                dynamic_order_loss=dynamic_order_loss.detach(),
            )
        return results
