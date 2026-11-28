"""GLIP-T: grounded language-image pretraining for prompt-based detection.

Paper: Li et al., "Grounded Language-Image Pre-training", CVPR 2022,
https://arxiv.org/abs/2112.03857. Code: https://github.com/microsoft/GLIP
(MIT licence), from which this pure-PyTorch port of the GLIP-T (Swin-T,
O365 + GoldG) detector is adapted. The official code builds on
``maskrcnn_benchmark`` and needs compiled CUDA kernels; this port uses
``torchvision.ops`` instead (deformable convolution, NMS), so it runs on CPU,
CUDA and Apple MPS, and loads the official checkpoint unchanged.

GLIP detects the phrases of a text prompt: every anchor box is scored
against every prompt token, and a phrase's score is the mean over its
tokens. MIU-VL (Qin et al., ICLR 2023) uses it zero-shot on medical images
with prompts that describe the target's colour, shape and location; see
:mod:`pyhealth.models.miu_vl`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import batched_nms, deform_conv2d

from ..datasets import SampleDataset
from .base_model import BaseModel

# GLIP-T (configs/pretrain/glip_Swin_T_O365_GoldG.yaml).
_SWIN_DEPTHS = (2, 2, 6, 2)
_SWIN_HEADS = (3, 6, 12, 24)
_SWIN_DIM = 96
_SWIN_WINDOW = 7
_CHANNELS = 256
_LANG_DIM = 768
_NUM_CONVS = 6
_ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_ANCHOR_SIZES = (64, 128, 256, 512, 1024)
# Inputs are BGR in [0, 255] (INPUT.TO_BGR255), normalised per channel.
_PIXEL_MEAN = (103.530, 116.280, 123.675)
_PIXEL_STD = (57.375, 57.120, 58.395)
_MIN_SIZE, _MAX_SIZE, _SIZE_DIVISIBILITY = 800, 1333, 32
_MAX_QUERY_LEN = 256
# Post-processing (MODEL.ATSS defaults).
_PRE_NMS_THRESH = 0.05
_PRE_NMS_TOP_N = 1000
_NMS_THRESH = 0.6
_DETECTIONS_PER_IMG = 100
_CLAMP = 50000.0

HF_REPO_ID = "GLIPModel/GLIP"
HF_REVISION = "66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1"
HF_FILENAME = "glip_tiny_model_o365_goldg.pth"
"""The official GLIP-T (O365 + GoldG) checkpoint (3.7 GB with optimizer state)."""


# ---------------------------------------------------------------------------
# Swin-T backbone (maskrcnn_benchmark/modeling/backbone/swint.py)
# ---------------------------------------------------------------------------


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _window_partition(x: torch.Tensor, size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // size, size, w // size, size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, size, size, c)


def _window_reverse(windows: torch.Tensor, size: int, h: int, w: int) -> torch.Tensor:
    b = windows.shape[0] // (h * w // size // size)
    x = windows.view(b, h // size, w // size, size, size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class _WindowAttention(nn.Module):
    def __init__(self, dim: int, window: int, heads: int) -> None:
        super().__init__()
        self.window = window
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, heads)
        )
        coords = torch.stack(
            torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")
        ).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        attn = attn + bias.view(n, n, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.heads, n, n) + mask[None, :, None]
            attn = attn.view(-1, self.heads, n, n)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class _SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, shift: int) -> None:
        super().__init__()
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, _SWIN_WINDOW, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, 4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int, mask: torch.Tensor) -> torch.Tensor:
        b, _, c = x.shape
        ws = _SWIN_WINDOW
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        pad_r, pad_b = (ws - w % ws) % ws, (ws - h % ws) % ws
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        hp, wp = x.shape[1:3]
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        windows = _window_partition(x, ws).view(-1, ws * ws, c)
        windows = self.attn(windows, mask if self.shift else None)
        x = _window_reverse(windows.view(-1, ws, ws, c), ws, hp, wp)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = x[:, :h, :w].reshape(b, h * w, c)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class _PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        x = F.pad(x.view(b, h, w, c), (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1
        )
        return self.reduction(self.norm(x.view(b, -1, 4 * c)))


class _SwinStage(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, downsample: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            _SwinBlock(dim, heads, 0 if i % 2 == 0 else _SWIN_WINDOW // 2)
            for i in range(depth)
        )
        self.downsample = _PatchMerging(dim) if downsample else None

    def _shift_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        ws, shift = _SWIN_WINDOW, _SWIN_WINDOW // 2
        hp, wp = math.ceil(h / ws) * ws, math.ceil(w / ws) * ws
        img = torch.zeros(1, hp, wp, 1, device=device)
        cuts = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        for i, hs in enumerate(cuts):
            for j, wsl in enumerate(cuts):
                img[:, hs, wsl, :] = 3 * i + j
        win = _window_partition(img, ws).view(-1, ws * ws)
        diff = win.unsqueeze(1) - win.unsqueeze(2)
        return diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)

    def forward(self, x: torch.Tensor, h: int, w: int):
        mask = self._shift_mask(h, w, x.device)
        for block in self.blocks:
            x = block(x, h, w, mask)
        if self.downsample is None:
            return x, x, h, w
        return x, self.downsample(x, h, w), (h + 1) // 2, (w + 1) // 2


class _PatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, _SWIN_DIM, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(_SWIN_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (4 - w % 4) % 4, 0, (4 - h % 4) % 4))
        x = self.proj(x)
        return self.norm(x.flatten(2).transpose(1, 2)), x.shape[2], x.shape[3]


class _SwinT(nn.Module):
    """Swin-T returning stages 3-5 (strides 8, 16, 32), each layer-normed."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.layers = nn.ModuleList(
            _SwinStage(_SWIN_DIM * 2**i, d, h, downsample=i < 3)
            for i, (d, h) in enumerate(zip(_SWIN_DEPTHS, _SWIN_HEADS))
        )
        # norm0 (stride 4) is an identity for the RetinaNet-style FPN.
        for i in range(1, 4):
            self.add_module(f"norm{i}", nn.LayerNorm(_SWIN_DIM * 2**i))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x, h, w = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            out, x, nh, nw = layer(x, h, w)
            if i > 0:
                out = getattr(self, f"norm{i}")(out)
                outs.append(out.view(-1, h, w, out.shape[-1]).permute(0, 3, 1, 2))
            h, w = nh, nw
        return outs


class _FPN(nn.Module):
    """FPN over Swin stages 3-5 plus P6/P7 from P5 (5 levels, 256 channels)."""

    def __init__(self) -> None:
        super().__init__()
        for idx, ch in zip((2, 3, 4), (192, 384, 768)):
            self.add_module(f"fpn_inner{idx}", nn.Conv2d(ch, _CHANNELS, 1))
            self.add_module(f"fpn_layer{idx}", nn.Conv2d(_CHANNELS, _CHANNELS, 3, padding=1))
        self.top_blocks = nn.Module()
        self.top_blocks.p6 = nn.Conv2d(_CHANNELS, _CHANNELS, 3, 2, 1)
        self.top_blocks.p7 = nn.Conv2d(_CHANNELS, _CHANNELS, 3, 2, 1)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        last = self.fpn_inner4(feats[2])
        results = [self.fpn_layer4(last)]
        for feat, idx in ((feats[1], 3), (feats[0], 2)):
            lateral = getattr(self, f"fpn_inner{idx}")(feat)
            last = lateral + F.interpolate(last, size=lateral.shape[-2:], mode="nearest")
            results.insert(0, getattr(self, f"fpn_layer{idx}")(last))
        p6 = self.top_blocks.p6(results[-1])
        return [*results, p6, self.top_blocks.p7(F.relu(p6))]


# ---------------------------------------------------------------------------
# Vision-language fusion head (maskrcnn_benchmark/modeling/rpn/vldyhead.py)
# ---------------------------------------------------------------------------


class _BiMultiHeadAttention(nn.Module):
    """Image-to-text and text-to-image attention (utils/fuse_helper.py)."""

    def __init__(self, v_dim: int, l_dim: int, embed_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim**-0.5
        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)
        self.dropout = nn.Dropout(0.1)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2).reshape(
            b * self.heads, n, self.head_dim
        )

    def _merge(self, x: torch.Tensor, b: int) -> torch.Tensor:
        n = x.shape[1]
        return x.view(b, self.heads, n, self.head_dim).transpose(1, 2).reshape(b, n, -1)

    def forward(self, v: torch.Tensor, lang: torch.Tensor, mask: torch.Tensor):
        b, n_v, _ = v.shape
        n_l = lang.shape[1]
        q = self._heads(self.v_proj(v) * self.scale)
        k = self._heads(self.l_proj(lang))
        attn = torch.bmm(q, k.transpose(1, 2)).clamp(-_CLAMP, _CLAMP)
        attn_l = attn.transpose(1, 2)
        attn_l = (attn_l - attn_l.max(-1, keepdim=True)[0]).clamp(-_CLAMP, _CLAMP)
        attn_l = attn_l.softmax(-1)
        pad = (mask == 0)[:, None, None, :]
        attn = attn.view(b, self.heads, n_v, n_l).masked_fill(pad, -9e15)
        attn_v = attn.view(b * self.heads, n_v, n_l).softmax(-1)
        out_v = torch.bmm(self.dropout(attn_v), self._heads(self.values_l_proj(lang)))
        out_l = torch.bmm(self.dropout(attn_l), self._heads(self.values_v_proj(v)))
        return self.out_v_proj(self._merge(out_v, b)), self.out_l_proj(self._merge(out_l, b))


class _BiAttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(_CHANNELS)
        self.layer_norm_l = nn.LayerNorm(_LANG_DIM)
        self.attn = _BiMultiHeadAttention(_CHANNELS, _LANG_DIM, 2048, 8)
        self.gamma_v = nn.Parameter(torch.full((_CHANNELS,), 1.0 / _NUM_CONVS))
        self.gamma_l = nn.Parameter(torch.full((_LANG_DIM,), 1.0 / _NUM_CONVS))

    def forward(self, feats: list[torch.Tensor], lang: torch.Tensor, mask: torch.Tensor):
        sizes = [f.shape[-2:] for f in feats]
        v = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], 1)
        v, lang = self.layer_norm_v(v), self.layer_norm_l(lang)
        dv, dl = self.attn(v, lang, mask)
        v = (v + self.gamma_v * dv).transpose(1, 2)
        out, start = [], 0
        for h, w in sizes:
            out.append(v[:, :, start:start + h * w].reshape(v.shape[0], -1, h, w))
            start += h * w
        return out, lang + self.gamma_l * dl


class _VLFuse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b_attn = _BiAttentionBlock()

    def forward(self, state: dict[str, Any]) -> dict[str, Any]:
        visual, lang = self.b_attn(state["visual"], state["hidden"], state["masks"])
        return {**state, "visual": visual, "hidden": lang}


class _BertSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(_LANG_DIM, _LANG_DIM)
        self.key = nn.Linear(_LANG_DIM, _LANG_DIM)
        self.value = nn.Linear(_LANG_DIM, _LANG_DIM)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, 12, 64).transpose(1, 2)

        scores = heads(self.query(x)) @ heads(self.key(x)).transpose(-1, -2) / 8.0
        scores = scores.clamp(-_CLAMP, _CLAMP) + mask
        out = self.dropout(scores.softmax(-1)) @ heads(self.value(x))
        return out.transpose(1, 2).reshape(b, n, _LANG_DIM)


class _BertDenseNorm(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, _LANG_DIM)
        self.LayerNorm = nn.LayerNorm(_LANG_DIM, eps=1e-12)
        self.dropout = nn.Dropout(0.1)


class _BertAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = _BertSelfAttention()
        self.output = _BertDenseNorm(_LANG_DIM)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.output
        return out.LayerNorm(out.dropout(out.dense(self.self(x, mask))) + x)


class _BertEncoderLayer(nn.Module):
    """A BERT layer with GLIP's clamps (rpn/modeling_bert.py)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = _BertAttention()
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(_LANG_DIM, 4 * _LANG_DIM)
        self.output = _BertDenseNorm(4 * _LANG_DIM)

    def forward(self, state: dict[str, Any]) -> dict[str, Any]:
        x, mask = state["hidden"], state["masks"]
        ext = (1.0 - mask[:, None, None, :].to(x.dtype)) * -10000.0
        x = self.attention(x, ext)
        h = F.gelu(self.intermediate.dense(x).clamp(-_CLAMP, _CLAMP)).clamp(-_CLAMP, _CLAMP)
        out = self.output
        h = out.dropout(out.dense(h)).clamp(-_CLAMP, _CLAMP)
        h = out.LayerNorm(h + x).clamp(-_CLAMP, _CLAMP)
        return {**state, "hidden": h}


class _DeformConvGN(nn.Module):
    """3x3 modulated deformable convolution + GroupNorm(16)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Module()
        self.conv.weight = nn.Parameter(torch.empty(out_ch, in_ch, 3, 3))
        self.conv.bias = nn.Parameter(torch.zeros(out_ch))
        nn.init.normal_(self.conv.weight, 0, 0.01)
        self.bn = nn.GroupNorm(16, out_ch)

    def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, w = ((s + 2 - 3) // self.stride + 1 for s in x.shape[-2:])
        out = deform_conv2d(
            x.float(), _fit(offset, h, w).float(), self.conv.weight.float(),
            self.conv.bias.float(), stride=self.stride, padding=1,
            mask=_fit(mask, h, w).float(),
        )
        return self.bn(out.to(x.dtype))


def _fit(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Reads ``t`` (``[B, C, H, W]``) as a ``[B, C, h, w]`` buffer.

    DyHead computes one set of offsets per FPN level and also applies it to
    the convolution of the next, smaller level. GLIP's CUDA kernel indexes
    each image's offset buffer with the output size, so it reads the first
    ``C * h * w`` values in memory order; the checkpoint was trained that way.
    """
    if t.shape[-2:] == (h, w):
        return t
    b, c = t.shape[:2]
    return t.reshape(b, -1)[:, : c * h * w].reshape(b, c, h, w)


class _DyReLU(nn.Module):
    """Dynamic ReLU (layers/dyrelu.py, K2 with bias)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // 4), nn.ReLU(), nn.Linear(ch // 4, 4 * ch), nn.Hardsigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = self.fc(x.mean((2, 3)))[:, :, None, None]
        a1, b1, a2, b2 = coeff.split(x.shape[1], dim=1)
        return torch.max(x * ((a1 - 0.5) * 2 + 1.0) + b1 - 0.5, x * ((a2 - 0.5) * 2) + b2 - 0.5)


class _DyConv(nn.Module):
    """DyHead block: scale-aware fusion of neighbouring FPN levels."""

    def __init__(self) -> None:
        super().__init__()
        self.DyConv = nn.ModuleList(
            [_DeformConvGN(_CHANNELS, _CHANNELS, s) for s in (1, 1, 2)]
        )
        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(_CHANNELS, 1, 1), nn.ReLU()
        )
        self.relu = _DyReLU(_CHANNELS)
        self.offset = nn.Conv2d(_CHANNELS, 27, 3, padding=1)

    def forward(self, state: dict[str, Any]) -> dict[str, Any]:
        feats = state["visual"]
        out = []
        for level, feat in enumerate(feats):
            om = self.offset(feat)
            args = (om[:, :18], om[:, 18:].sigmoid())
            parts = [self.DyConv[1](feat, *args)]
            if level > 0:
                parts.append(self.DyConv[2](feats[level - 1], *args))
            if level < len(feats) - 1:
                up = self.DyConv[0](feats[level + 1], *args)
                parts.append(
                    F.interpolate(up, size=feat.shape[-2:], mode="bilinear", align_corners=True)
                )
            stacked = torch.stack(parts)
            gate = F.hardsigmoid(torch.stack([self.AttnConv(p) for p in parts]))
            out.append(self.relu((stacked * gate).mean(0)))
        return {**state, "visual": out}


class _Scale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _VLDyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        tower: list[nn.Module] = []
        for _ in range(_NUM_CONVS):
            tower += [_VLFuse(), _BertEncoderLayer(), _DyConv()]
        self.dyhead_tower = nn.Sequential(*tower)
        # cls_logits (80 COCO classes) is unused by the grounding head but
        # kept so that the checkpoint loads strictly.
        self.cls_logits = nn.Conv2d(_CHANNELS, 80, 1)
        self.bbox_pred = nn.Conv2d(_CHANNELS, 4, 1)
        self.centerness = nn.Conv2d(_CHANNELS, 1, 1)
        self.scales = nn.ModuleList(_Scale() for _ in _ANCHOR_STRIDES)
        self.dot_product_projection_text = nn.Linear(_LANG_DIM, _CHANNELS)
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias_lang = nn.Parameter(torch.zeros(_LANG_DIM))
        self.bias0 = nn.Parameter(torch.full((1,), -math.log(99.0)))

    def forward(self, feats: list[torch.Tensor], lang: dict[str, torch.Tensor]):
        state = self.dyhead_tower({"visual": feats, **lang})
        emb = F.normalize(state["hidden"], p=2, dim=-1)
        tokens = self.dot_product_projection_text(emb / 2.0)
        token_bias = emb @ self.bias_lang + self.bias0
        boxes, centers, logits = [], [], []
        for level, feat in enumerate(state["visual"]):
            boxes.append(self.scales[level](self.bbox_pred(feat)))
            centers.append(self.centerness(feat))
            queries = feat.flatten(2).transpose(1, 2)
            logit = queries @ tokens.transpose(-1, -2) / self.log_scale.exp()
            logits.append((logit + token_bias[:, None, :]).clamp(-_CLAMP, _CLAMP))
        return boxes, centers, logits


# ---------------------------------------------------------------------------
# Anchors, box decoding and post-processing (rpn/anchor_generator.py,
# rpn/vldyhead.py BoxCoder, rpn/inference.py ATSSPostProcessor)
# ---------------------------------------------------------------------------


def _anchors(sizes: Sequence[tuple[int, int]], device: torch.device) -> list[torch.Tensor]:
    """One square anchor per location and level, Detectron-style."""
    out = []
    for (h, w), stride, size in zip(sizes, _ANCHOR_STRIDES, _ANCHOR_SIZES):
        ctr = (stride - 1) / 2
        half = (size - 1) / 2
        base = torch.tensor([ctr - half, ctr - half, ctr + half, ctr + half], device=device)
        ys = torch.arange(0, h * stride, stride, dtype=torch.float32, device=device)
        xs = torch.arange(0, w * stride, stride, dtype=torch.float32, device=device)
        sy, sx = torch.meshgrid(ys, xs, indexing="ij")
        shifts = torch.stack([sx, sy, sx, sy], -1).reshape(-1, 4)
        out.append(shifts + base)
    return out


def _decode(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    widths = anchors[:, 2] - anchors[:, 0] + 1
    heights = anchors[:, 3] - anchors[:, 1] + 1
    cx = anchors[:, 0] + 0.5 * (widths - 1)
    cy = anchors[:, 1] + 0.5 * (heights - 1)
    dx, dy = deltas[:, 0] / 10.0, deltas[:, 1] / 10.0
    dw = (deltas[:, 2] / 5.0).clamp(max=math.log(1000.0 / 16))
    dh = (deltas[:, 3] / 5.0).clamp(max=math.log(1000.0 / 16))
    pcx, pcy = dx * widths + cx, dy * heights + cy
    pw, ph = dw.exp() * widths, dh.exp() * heights
    return torch.stack(
        [pcx - 0.5 * (pw - 1), pcy - 0.5 * (ph - 1), pcx + 0.5 * (pw - 1), pcy + 0.5 * (ph - 1)],
        dim=1,
    )


def _postprocess(
    boxes: list[torch.Tensor],
    centers: list[torch.Tensor],
    logits: list[torch.Tensor],
    anchors: list[torch.Tensor],
    phrase_tokens: list[list[int]],
    image_size: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Detections of one image (batch index 0 of each level's outputs)."""
    height, width = image_size
    all_boxes, all_scores, all_labels = [], [], []
    for box, center, logit, anchor in zip(boxes, centers, logits, anchors):
        prob = logit[0].sigmoid()
        scores = torch.stack([prob[:, t].mean(-1) for t in phrase_tokens], dim=1)
        candidates = scores > _PRE_NMS_THRESH
        top_n = min(int(candidates.sum()), _PRE_NMS_TOP_N)
        scores = scores * center[0].reshape(-1).sigmoid()[:, None]
        values, order = scores[candidates].topk(top_n, sorted=False)
        loc, label = candidates.nonzero()[order].unbind(1)
        det = _decode(box[0].permute(1, 2, 0).reshape(-1, 4)[loc], anchor[loc])
        det[:, 0::2] = det[:, 0::2].clamp(0, width - 1)
        det[:, 1::2] = det[:, 1::2].clamp(0, height - 1)
        all_boxes.append(det)
        all_scores.append(values.sqrt())
        all_labels.append(label)
    det, scores, labels = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
    # GLIP's NMS measures boxes in pixels (width = x2 - x1 + 1).
    keep = batched_nms(det + det.new_tensor([0, 0, 1, 1]), scores, labels, _NMS_THRESH)
    det, scores, labels = det[keep], scores[keep], labels[keep]
    if len(scores) > _DETECTIONS_PER_IMG:
        thresh = torch.kthvalue(scores.cpu(), len(scores) - _DETECTIONS_PER_IMG + 1).values
        keep = scores >= thresh.item()
        det, scores, labels = det[keep], scores[keep], labels[keep]
    return {"boxes": det, "scores": scores, "labels": labels}


# ---------------------------------------------------------------------------
# Prompts and images
# ---------------------------------------------------------------------------


def _phrase_tokens(encoding: Any, spans: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Token indices of each character span (engine/inference.py)."""
    out = []
    for beg, end in spans:
        first = next(
            (t for c in (beg, beg + 1, beg + 2) if (t := encoding.char_to_token(c)) is not None),
            None,
        )
        last = next(
            (t for c in (end - 1, end - 2, end - 3)
             if c >= 0 and (t := encoding.char_to_token(c)) is not None),
            None,
        )
        if first is None or last is None:
            raise ValueError(f"Phrase at characters {beg}-{end} has no tokens.")
        out.append(list(range(first, last + 1)))
    return out


def _resize_shape(width: int, height: int) -> tuple[int, int]:
    """Test-time size (data/transforms/transforms.py Resize.get_size)."""
    size = _MIN_SIZE
    short, long = min(width, height), max(width, height)
    if long / short * size > _MAX_SIZE:
        size = round(_MAX_SIZE * short / long)
    if (width <= height and width == size) or (height <= width and height == size):
        return height, width
    if width < height:
        return int(size * height / width), size
    return size, int(size * width / height)


def build_caption(phrases: Sequence[dict[str, str] | str]) -> tuple[str, list[tuple[int, int]]]:
    """Joins phrases into one GLIP caption.

    Each phrase is a class name or a ``{"prefix", "name", "suffix"}`` dict, as
    in MIU-VL's prompt files; only ``name`` is scored, the prefix and suffix
    are context. Phrases are separated by ``". "``.

    Returns:
        The caption and the character span of each phrase's ``name``.

    Examples:
        >>> build_caption([{"name": "pink oval bump", "suffix": " in rectum"}])
        ('pink oval bump in rectum', [(0, 14)])
    """
    caption, spans = "", []
    for i, phrase in enumerate(phrases):
        if isinstance(phrase, str):
            phrase = {"name": phrase}
        if i:
            caption += ". "
        caption += phrase.get("prefix", "")
        start = len(caption)
        caption += phrase["name"]
        spans.append((start, len(caption)))
        caption += phrase.get("suffix", "")
    return caption, spans


class GLIPDetector(nn.Module):
    """The GLIP-T network: Swin-T + FPN, BERT and the fusion head.

    Its submodules are named as in the official model, so its
    ``state_dict`` is the official checkpoint without the ``module.`` prefix.

    Args:
        tokenizer: A Hugging Face tokenizer name, or a fast tokenizer
            (``char_to_token`` is needed to map phrases to tokens).

    Examples:
        >>> from pyhealth.models import GLIPDetector
        >>> detector = GLIPDetector()  # doctest: +SKIP
        >>> detector.load_checkpoint("glip_tiny_model_o365_goldg.pth")  # doctest: +SKIP
        >>> detector.eval()(["polyp.png"], [["polyp"]])[0]["scores"]  # doctest: +SKIP
        tensor([0.3360, ...])
    """

    def __init__(self, tokenizer: Any = "bert-base-uncased") -> None:
        super().__init__()
        from transformers import AutoTokenizer, BertConfig, BertModel

        self.backbone = nn.Module()
        self.backbone.body = _SwinT()
        self.backbone.fpn = _FPN()
        self.language_backbone = nn.Module()
        self.language_backbone.body = nn.Module()
        # BertConfig() is bert-base-uncased; GLIP's checkpoint holds its weights.
        self.language_backbone.body.model = BertModel(BertConfig(), add_pooling_layer=False)
        self.rpn = nn.Module()
        self.rpn.head = _VLDyHead()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        )
        self.register_buffer("pixel_mean", torch.tensor(_PIXEL_MEAN).view(3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(_PIXEL_STD).view(3, 1, 1), persistent=False)

    def load_checkpoint(self, path: str) -> None:
        """Loads an official GLIP checkpoint (or a saved ``state_dict``)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        state = {
            k.removeprefix("module."): v
            for k, v in state.items()
            if "anchor_generator" not in k and not k.endswith("embeddings.position_ids")
        }
        self.load_state_dict(state, strict=True)

    def preprocess(self, image: Any) -> torch.Tensor:
        """An RGB image (PIL, ``[H, W, 3]`` uint8 array or path) as GLIP input."""
        from PIL import Image

        if isinstance(image, str):
            with Image.open(image) as f:
                image = f.convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        h, w = _resize_shape(*image.size)
        image = image.resize((w, h), Image.BILINEAR)
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = x.view(h, w, 3).permute(2, 0, 1).flip(0).float()
        return (x.to(self.pixel_mean.device) - self.pixel_mean) / self.pixel_std

    def encode_text(self, captions: Sequence[str]):
        tokens = self.tokenizer(
            list(captions), max_length=_MAX_QUERY_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.pixel_mean.device)
        out = self.language_backbone.body.model(
            input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
        )
        return tokens, {"hidden": out.last_hidden_state, "masks": tokens.attention_mask}

    def forward(
        self, images: Sequence[Any], phrases: Sequence[Sequence[dict[str, str] | str]]
    ) -> list[dict[str, torch.Tensor]]:
        """Detects each image's phrases.

        Images are processed one at a time, as padding a batch changes the
        features near the borders.

        Args:
            images: RGB images (PIL images, uint8 arrays or file paths).
            phrases: For each image, the phrases to detect (see
                :func:`build_caption`).

        Returns:
            For each image, ``boxes`` (``[N, 4]``, ``x1, y1, x2, y2`` in pixels
            of the original image), ``scores`` (``[N]``) and ``labels``
            (``[N]``, the index of the phrase).
        """
        results = []
        for image, image_phrases in zip(images, phrases):
            x = self.preprocess(image)
            h, w = x.shape[-2:]
            pad_h = (-h) % _SIZE_DIVISIBILITY
            pad_w = (-w) % _SIZE_DIVISIBILITY
            x = F.pad(x, (0, pad_w, 0, pad_h))[None]
            caption, spans = build_caption(image_phrases)
            tokens, lang = self.encode_text([caption])
            phrase_tokens = _phrase_tokens(tokens[0], spans)
            feats = self.backbone.fpn(self.backbone.body(x))
            boxes, centers, logits = self.rpn.head(feats, lang)
            anchors = _anchors([f.shape[-2:] for f in feats], x.device)
            det = _postprocess(boxes, centers, logits, anchors, phrase_tokens, (h, w))
            orig_w, orig_h = _image_size(image)
            det["boxes"] = det["boxes"] * det["boxes"].new_tensor(
                [orig_w / w, orig_h / h, orig_w / w, orig_h / h]
            )
            results.append(det)
        return results


def _image_size(image: Any) -> tuple[int, int]:
    from PIL import Image

    if isinstance(image, str):
        with Image.open(image) as f:
            return f.size
    if isinstance(image, Image.Image):
        return image.size
    return image.shape[1], image.shape[0]


class GLIP(BaseModel):
    """GLIP-T for zero-shot, prompt-based detection (MIU-VL).

    Wraps :class:`GLIPDetector` for PyHealth samples. Each sample's image
    (a path, PIL image or RGB array) is searched for the phrases of a
    prompt; a prompt is a list of phrases, each a string or a
    ``{"prefix", "name", "suffix"}`` dict (see :func:`build_caption` and
    :mod:`pyhealth.models.miu_vl`). Phrase ``i`` predicts label ``i``.

    The model is used zero-shot, as in MIU-VL's zero-shot results: it has
    no loss, and ``forward`` returns detections only. Fine-tuning is not
    supported. The official checkpoint is GLIP-T (Swin-T) pretrained on
    Objects365 and GoldG (Li et al., CVPR 2022); this port matches the
    official code's outputs within float error.

    Args:
        dataset: Optional :class:`~pyhealth.datasets.SampleDataset`, e.g.
            from :class:`~pyhealth.tasks.PolypDetection`. Its first output
            key (e.g. ``"boxes"``, COCO ``[x, y, w, h]`` lists) is returned
            as ``y_true``.
        prompt: The prompt used for every image, unless a batch carries
            ``prompt_key``.
        pretrained: ``True`` downloads the official checkpoint
            (:data:`HF_REPO_ID` at :data:`HF_REVISION`, 3.7 GB); a string is
            a local checkpoint path; ``False`` keeps random weights.
        image_key: Batch key of the images.
        prompt_key: Batch key of per-sample prompts (MIU-VL's image-specific
            VQA and hybrid prompts), used instead of ``prompt`` when present.
        tokenizer: Text tokenizer, a Hugging Face name or a fast tokenizer;
            GLIP-T uses ``bert-base-uncased``.

    Examples:
        >>> from pyhealth.models import GLIP
        >>> from pyhealth.models.miu_vl import POLYP_MANUAL_PROMPT
        >>> model = GLIP(prompt=[POLYP_MANUAL_PROMPT], pretrained=True)  # doctest: +SKIP
        >>> out = model(image=["polyp.png"])  # doctest: +SKIP
        >>> out["y_pred"][0]["boxes"]  # doctest: +SKIP
        [[189.4, 304.7, 193.5, 172.5], ...]
    """

    def __init__(
        self,
        dataset: SampleDataset | None = None,
        prompt: Sequence[dict[str, str] | str] | None = None,
        pretrained: bool | str = False,
        image_key: str = "image",
        prompt_key: str = "prompt",
        tokenizer: Any = "bert-base-uncased",
    ) -> None:
        super().__init__(dataset)
        self.prompt = list(prompt) if prompt is not None else None
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.detector = GLIPDetector(tokenizer)
        if pretrained:
            path = pretrained
            if pretrained is True:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(HF_REPO_ID, HF_FILENAME, revision=HF_REVISION)
            self.detector.load_checkpoint(str(path))

    def forward(self, **batch: Any) -> dict[str, Any]:
        """Detects the prompt in each image of the batch.

        Returns:
            ``y_pred``: for each image, ``boxes`` (COCO ``[x, y, w, h]`` in
            pixels, as GLIP's evaluation writes them), ``scores`` and
            ``labels`` (phrase indices), as lists, best first. ``y_true``:
            the dataset's boxes, when the batch has them.

        Raises:
            ValueError: If neither ``prompt`` nor the batch gives a prompt.
        """
        images = batch[self.image_key]
        prompts = batch.get(self.prompt_key)
        if prompts is None:
            if self.prompt is None:
                raise ValueError(f"Give GLIP a prompt or a batch with {self.prompt_key!r}.")
            prompts = [self.prompt] * len(images)
        y_pred = []
        for det in self.detector(images, prompts):
            boxes = det["boxes"].detach().cpu()
            order = det["scores"].argsort(descending=True).cpu()
            xywh = torch.cat([boxes[:, :2], boxes[:, 2:] - boxes[:, :2] + 1], dim=1)[order]
            y_pred.append({
                "boxes": xywh.tolist(),
                "scores": det["scores"].detach().cpu()[order].tolist(),
                "labels": det["labels"].cpu()[order].tolist(),
            })
        out: dict[str, Any] = {"y_pred": y_pred}
        target = self.label_keys[0] if self.label_keys else None
        if target is not None and target in batch:
            out["y_true"] = [{"boxes": boxes} for boxes in batch[target]]
        return out
