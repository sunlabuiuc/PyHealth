from __future__ import annotations

import os
import sys
import types
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TissueAwareSimCLR(BaseModel):
    """Multiple instance learning classifier with a SimCLR-initialized ResNet-18 encoder.

    This model is designed for TCGA-CRCk slide-level prediction from bags of image
    tiles. Each bag is encoded tile by tile using a ResNet-18 backbone, projected
    into a hidden space, pooled into a single bag representation, and passed to a
    classification head.

    The model supports either attention-based pooling or mean pooling over tile
    embeddings and can optionally freeze the encoder during downstream training.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        checkpoint_path: Optional[str] = None,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        freeze_encoder: bool = False,
        pooling: str = "attention",
        tile_chunk_size: int = 1024,
        use_bf16: bool = False,
    ) -> None:
        """Initializes the tissue-aware SimCLR classifier.

        Args:
            dataset: PyHealth sample dataset used to infer feature and label metadata.
            checkpoint_path: Optional path to a pretrained encoder checkpoint.
            hidden_dim: Output dimension of the projection layer before pooling.
            dropout: Dropout probability applied before the final classifier.
            freeze_encoder: Whether to freeze encoder weights during training.
            pooling: Bag pooling strategy. Must be either "attention" or "mean".
            tile_chunk_size: Number of tiles to encode at once to control memory usage.
            use_bf16: Whether to use bfloat16 autocast on CUDA during encoding and
                classification.

        Raises:
            ValueError: If the dataset does not expose exactly one feature key.
            ValueError: If the dataset does not expose exactly one label key.
            ValueError: If `pooling` is not one of the supported strategies.
            ValueError: If `tile_chunk_size` is less than 1.
        """
        super().__init__(dataset=dataset)

        if len(self.feature_keys) != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects exactly one feature key, "
                f"but got {self.feature_keys}."
            )
        if len(self.label_keys) != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects exactly one label key, "
                f"but got {self.label_keys}."
            )
        if pooling not in {"attention", "mean"}:
            raise ValueError("pooling must be either 'attention' or 'mean'.")
        if tile_chunk_size < 1:
            raise ValueError("tile_chunk_size must be at least 1.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.freeze_encoder = freeze_encoder
        self.tile_chunk_size = tile_chunk_size
        self.use_bf16 = use_bf16

        backbone = models.resnet18(weights=None)
        self.encoder_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        if checkpoint_path is not None:
            self._load_encoder_checkpoint(checkpoint_path)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_v = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, self.get_output_size())

        train_labels = [
            int(dataset[i]["label"])
            for i in range(len(dataset))
            if str(dataset[i].get("data_split", "")).strip().lower() in {"train", "training", "tr"}
        ]
        
        num_pos = sum(train_labels)
        num_neg = len(train_labels) - num_pos
        pos_weight = num_neg / max(num_pos, 1)
        
        self.register_buffer(
            "pos_weight_tensor",
            torch.tensor([pos_weight], dtype=torch.float32),
        )
        
    def _load_encoder_checkpoint(self, checkpoint_path: str) -> None:
        """Loads encoder weights from a plain PyTorch checkpoint.

        The loader accepts multiple checkpoint layouts and removes common wrapper
        prefixes before applying the weights to the ResNet encoder.

        Args:
            checkpoint_path: Path to the checkpoint file on disk.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
            ValueError: If the checkpoint does not contain a usable state dict.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "encoder" in checkpoint:
                state_dict = checkpoint["encoder"]

        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint did not contain a usable state dict.")

        cleaned_state_dict = {}
        removable_prefixes = [
            "model.resnet.",
            "resnet.",
            "model.",
            "module.",
            "encoder.",
            "backbone.",
            "online_network.encoder.",
            "encoder_q.",
        ]

        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue

            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in removable_prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            cleaned_state_dict[new_key] = value

        missing, unexpected = self.encoder.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )

        if missing:
            print(f"[Warning] Missing keys: {missing}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {unexpected}")

    def _extract_images_and_mask(
        self,
        feature: Union[torch.Tensor, Tuple[torch.Tensor, ...], list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts a batch of bags into a padded image tensor and validity mask.

        This method supports either an already-batched image tensor or the
        list/tuple-based bag structure commonly returned by PyHealth processors.

        Args:
            feature: Input bag payload. Expected either as a tensor with shape
                [B, N, C, H, W] or [N, C, H, W], or as a list/tuple of per-sample
                bag tensors.

        Returns:
            A tuple containing:
                - images: Padded image tensor of shape [B, N, C, H, W].
                - mask: Boolean tensor of shape [B, N] indicating valid tiles.

        Raises:
            ValueError: If the feature payload type is unsupported.
            ValueError: If image tensors do not have the expected shape.
            ValueError: If a bag does not contain a tensor.
            ValueError: If the batch of bags is empty.
        """
        # Case 1: already a proper batched tensor
        if torch.is_tensor(feature):
            images = feature
            if images.dim() == 4:
                images = images.unsqueeze(0)
            if images.dim() != 5:
                raise ValueError(
                    f"Expected image tensor with shape [B, N, C, H, W], got {tuple(images.shape)}"
                )
            mask = images.abs().sum(dim=(2, 3, 4)) > 0
            return images, mask

        # Case 2: PyHealth often gives list/tuple per sample
        if not isinstance(feature, (list, tuple)):
            raise ValueError("Unsupported tile_bag payload type.")

        bag_tensors = []
        for item in feature:
            bag = item
            while isinstance(bag, (tuple, list)) and len(bag) > 0:
                bag = bag[0]

            if not torch.is_tensor(bag):
                raise ValueError("Expected each bag to contain a tensor.")

            # Sometimes processor may add a leading singleton batch dim
            if bag.dim() == 5 and bag.size(0) == 1:
                bag = bag.squeeze(0)

            if bag.dim() != 4:
                raise ValueError(
                    f"Expected each bag tensor to have shape [N, C, H, W], got {tuple(bag.shape)}"
                )

            bag_tensors.append(bag)

        if not bag_tensors:
            raise ValueError("Received empty batch of bags.")

        max_tiles = max(bag.shape[0] for bag in bag_tensors)
        c, h, w = bag_tensors[0].shape[1:]

        images = torch.zeros(
            (len(bag_tensors), max_tiles, c, h, w),
            dtype=bag_tensors[0].dtype,
        )
        mask = torch.zeros((len(bag_tensors), max_tiles), dtype=torch.bool)

        for i, bag in enumerate(bag_tensors):
            n = bag.shape[0]
            images[i, :n] = bag
            mask[i, :n] = True

        return images, mask

    def _encode_flat_images(self, flat_images: torch.Tensor) -> torch.Tensor:
        """Encodes flattened tile images into normalized projected features.

        Images are processed in chunks to reduce memory usage. If enabled, bfloat16
        autocast is used on CUDA. When the encoder is frozen, forward passes through
        the encoder are wrapped in `torch.no_grad()`.

        Args:
            flat_images: Tensor of tile images with shape [num_tiles, C, H, W].

        Returns:
            Tensor of L2-normalized tile features with shape
            [num_tiles, hidden_dim].
        """
        outputs = []
        use_amp = self.use_bf16 and self.device.type == "cuda"

        if self.freeze_encoder:
            self.encoder.eval()

        for start in range(0, flat_images.size(0), self.tile_chunk_size):
            chunk = flat_images[start : start + self.tile_chunk_size]
            chunk = chunk.to(self.device, non_blocking=True).float()
            chunk = chunk.contiguous(memory_format=torch.channels_last)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.freeze_encoder:
                        with torch.no_grad():
                            enc = self.encoder(chunk)
                    else:
                        enc = self.encoder(chunk)
                    enc = torch.flatten(enc, start_dim=1)
                    feat = self.proj(enc)
                    feat = F.normalize(feat, dim=-1)
            else:
                if self.freeze_encoder:
                    with torch.no_grad():
                        enc = self.encoder(chunk)
                    enc = torch.flatten(enc, start_dim=1)
                    feat = self.proj(enc)
                    feat = F.normalize(feat, dim=-1)
                else:
                    enc = self.encoder(chunk)
                    enc = torch.flatten(enc, start_dim=1)
                    feat = self.proj(enc)
                    feat = F.normalize(feat, dim=-1)

            outputs.append(feat.float())

        return torch.cat(outputs, dim=0)

    def _pool_bag(
        self,
        tile_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pools tile-level features into a single bag representation.

        Depending on the configured pooling strategy, this method applies either
        simple masked mean pooling or gated attention pooling.

        Args:
            tile_features: Tile embeddings of shape [B, N, hidden_dim].
            mask: Boolean validity mask of shape [B, N].

        Returns:
            A tuple containing:
                - bag_embeddings: Tensor of pooled bag representations with shape
                  [B, hidden_dim].
                - weights: Tensor of pooling weights with shape [B, N].
        """
        if self.pooling == "mean":
            weights = mask.float()
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = weights / denom
            bag_embeddings = torch.sum(weights.unsqueeze(-1) * tile_features, dim=1)
            return bag_embeddings, weights

        attn_v = torch.tanh(self.attn_v(tile_features))
        attn_u = torch.sigmoid(self.attn_u(tile_features))
        attn_logits = self.attn_w(attn_v * attn_u).squeeze(-1)

        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(attn_logits, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)

        bag_embeddings = torch.sum(weights.unsqueeze(-1) * tile_features, dim=1)
        return bag_embeddings, weights

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Runs the forward pass for slide-level prediction.

        Expected inputs are passed through `kwargs`, with image bags stored under
        `self.feature_key` and labels optionally stored under `self.label_key`.

        Args:
            **kwargs: Model inputs containing:
                - `self.feature_key`: Bagged image tiles.
                - `self.label_key`: Optional labels for supervised loss computation.

        Returns:
            A dictionary containing:
                - "loss": Training loss if labels are provided, otherwise `None`.
                - "y_prob": Predicted probabilities.
                - "y_true": Ground-truth labels if provided, otherwise `None`.
                - "logit": Raw classifier logits in float32.
                - "attention_weights": Tile-level pooling weights.

        Raises:
            ValueError: If the extracted image tensor does not have 3 channels.
        """
        images, mask = self._extract_images_and_mask(kwargs[self.feature_key])

        batch_size, num_tiles, channels, height, width = images.shape
        if channels != 3:
            raise ValueError("ResNet expects 3-channel images.")

        flat_images = images.reshape(batch_size * num_tiles, channels, height, width)
        tile_features = self._encode_flat_images(flat_images)
        tile_features = tile_features.reshape(batch_size, num_tiles, -1)

        mask = mask.to(self.device, non_blocking=True)
        bag_embeddings, weights = self._pool_bag(tile_features, mask)

        use_amp = self.use_bf16 and self.device.type == "cuda"
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bag_embeddings = self.dropout(bag_embeddings)
                logit = self.classifier(bag_embeddings)
        else:
            bag_embeddings = self.dropout(bag_embeddings)
            logit = self.classifier(bag_embeddings)

        logit_fp32 = logit.float()
        y_prob = self.prepare_y_prob(logit_fp32)

        y_true = None
        loss = None

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            mode = self.mode

            if mode == "multiclass":
                y_true = y_true.squeeze(-1).long()
                loss = nn.CrossEntropyLoss()(logit_fp32, y_true)

            elif mode in {"binary", "multilabel"}:
                y_true = y_true.float()
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logit_fp32,
                    y_true,
                    pos_weight=self.pos_weight_tensor,
                )

            elif mode == "regression":
                y_true = y_true.float()
                loss = nn.MSELoss()(logit_fp32, y_true)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit_fp32,
            "attention_weights": weights,
        }