import random
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


def _conv1d(
    in_planes: int,
    out_planes: int,
    kernel_size: int,
    stride: int = 1,
    padding: Optional[int] = None,
) -> nn.Conv1d:
    if padding is None:
        padding = kernel_size // 2
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )


class _BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = _conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv1d(planes, planes, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    @staticmethod
    def _align_time(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = min(left.shape[-1], right.shape[-1])
        return left[..., :length], right[..., :length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Trim to the same length before adding since strided convolutions can produce off-by-one mismatches
        residual, out = self._align_time(residual, out)
        out = out + residual
        out = self.relu(out)
        return out


class MultiScaleResNet1D(nn.Module):
    """Multi-scale 1D ResNet backbone used by the SPES models.

    Three parallel residual branches (kernel sizes 3, 5, and 7) each
    produce a 256-dimensional pooled feature; their concatenation yields a
    768-dimensional output embedding.

    Args:
        input_channel: Number of input channels (signal modes, e.g. 1 or 2).
        layers: Number of residual blocks per stage in each branch.
            Defaults to ``[1, 1, 1, 1]``.
        dropout_rate: Dropout probability applied after the final pooling.
            Default is 0.2.
    """

    output_dim = 256 * 3

    def __init__(
        self,
        input_channel: int,
        layers: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        layers = list(layers or [1, 1, 1, 1])

        # Track inplanes per kernel size separately so each branch can build its own downsample projections
        self.inplanes = {3: 64, 5: 64, 7: 64}

        self.conv1 = nn.Conv1d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer(3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer(3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer(3, 256, layers[2], stride=2)

        self.layer5x5_1 = self._make_layer(5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer(5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer(5, 256, layers[2], stride=2)

        self.layer7x7_1 = self._make_layer(7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer(7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer(7, 256, layers[2], stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=dropout_rate)

    def _make_layer(
        self,
        kernel_size: int,
        planes: int,
        blocks: int,
        stride: int = 2,
    ) -> nn.Sequential:
        downsample = None
        inplanes = self.inplanes[kernel_size]
        if stride != 1 or inplanes != planes * _BasicBlock1D.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    inplanes,
                    planes * _BasicBlock1D.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * _BasicBlock1D.expansion),
            )

        layers: List[nn.Module] = [
            _BasicBlock1D(inplanes, planes, kernel_size, stride, downsample)
        ]
        self.inplanes[kernel_size] = planes * _BasicBlock1D.expansion
        for _ in range(1, blocks):
            layers.append(
                _BasicBlock1D(self.inplanes[kernel_size], planes, kernel_size)
            )

        return nn.Sequential(*layers)

    def _forward_branch(self, x: torch.Tensor, layers: Iterable[nn.Module]) -> torch.Tensor:
        for layer in layers:
            x = layer(x)
        return self.pool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out3 = self._forward_branch(
            x, (self.layer3x3_1, self.layer3x3_2, self.layer3x3_3)
        )
        out5 = self._forward_branch(
            x, (self.layer5x5_1, self.layer5x5_2, self.layer5x5_3)
        )
        out7 = self._forward_branch(
            x, (self.layer7x7_1, self.layer7x7_2, self.layer7x7_3)
        )

        # Concatenate the three branch embeddings along the feature dimension
        out = torch.cat([out3, out5, out7], dim=1)
        # Squeeze the trailing dim from AdaptiveAvgPool1d output
        out = out[:, :, 0]
        return self.drop(out)


class SPESResponseEncoder(nn.Module):
    """Channel-wise CCEP response encoder combining a ResNet and Transformer.

    Each channel's response (mean and/or std across stimulation trials) is
    independently embedded by an optional multi-scale 1D ResNet and/or a
    flattened MLP prefix, then aggregated by a Transformer encoder whose
    class token serves as the output representation.

    Args:
        mean: If ``True``, include the mean CCEP response as an input mode.
        std: If ``True``, include the std CCEP response as an input mode.
            At least one of ``mean`` or ``std`` must be ``True``.
        conv_embedding: If ``True``, embed each channel via
            :class:`MultiScaleResNet1D`. Default is ``True``.
        mlp_embedding: If ``True`` (and ``conv_embedding`` is ``True``),
            prepend a flattened MLP prefix to the ResNet embedding.
            Default is ``True``.
        dropout_rate: Dropout probability. Default is 0.5.
        num_layers: Number of Transformer encoder layers. Default is 2.
        embedding_dim: Dimension of the per-channel embedding passed to the
            Transformer. Default is 64.
        random_channels: If set, randomly sub-sample this many channels per
            forward pass. ``None`` uses all channels. Default is ``None``.
        noise_std: Std of Gaussian noise injected during training. Default is 0.1.
        max_mlp_timesteps: Maximum number of timesteps kept for the MLP prefix.
            Default is 155.
        expected_timesteps: Expected signal length when ``conv_embedding=False``.
            Default is 509.
    """

    def __init__(
        self,
        mean: bool,
        std: bool,
        conv_embedding: bool = True,
        mlp_embedding: bool = True,
        dropout_rate: float = 0.5,
        num_layers: int = 2,
        embedding_dim: int = 64,
        random_channels: Optional[int] = None,
        noise_std: float = 0.1,
        max_mlp_timesteps: int = 155,
        expected_timesteps: int = 509,
    ):
        super().__init__()
        if not (mean or std):
            raise ValueError("Either mean or std, or both, must be enabled.")

        self.mean = mean
        self.std = std
        self.conv_embedding = conv_embedding
        self.mlp_embedding = mlp_embedding
        self.random_channels = random_channels
        self.noise_std = noise_std
        self.max_mlp_timesteps = max_mlp_timesteps
        self.expected_timesteps = expected_timesteps

        mode_count = int(self.mean) + int(self.std)
        if conv_embedding:
            self.msresnet = MultiScaleResNet1D(
                input_channel=mode_count, dropout_rate=dropout_rate
            )
            embedding_in = MultiScaleResNet1D.output_dim
            if mlp_embedding:
                embedding_in += mode_count * max_mlp_timesteps
        else:
            embedding_in = mode_count * expected_timesteps

        self.patch_to_embedding = nn.Linear(embedding_in, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.class_token = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(1, 1, embedding_dim))
        )

        # Pick the largest nhead that evenly divides embedding_dim, up to embedding_dim // 8
        nhead = max(
            head
            for head in range(1, max(1, embedding_dim // 8) + 1)
            if embedding_dim % head == 0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return x
        return x + torch.randn_like(x) * self.noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = self.apply_noise_and_zero_channels(x)

        if self.random_channels is not None:
            x = self.select_random_channels(x, self.random_channels)

        # Zero distance means the channel slot is padding and should be masked out
        distances = x[:, 0, :, 0]
        key_padding_mask = self.create_key_padding_mask(distances)

        channel_features = self.prepare_channels(x)
        x = self.dropout(self.patch_to_embedding(channel_features))

        # Prepend the learnable class token and aggregate via the Transformer
        class_token = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((class_token, x), dim=1)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        # Return only the class token output as the sequence-level embedding
        return x[:, 0]

    def apply_noise_and_zero_channels(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        # Identify non-padded channels by summing distances across the batch
        valid_columns = torch.where(x[:, 0, :, 0].sum(dim=0) != 0)[0]
        if len(valid_columns) > 0:
            # Randomly zero out up to half the valid channels for regularization
            sample_size = random.randint(0, len(valid_columns) // 2)
            if sample_size > 0:
                random_indices = torch.randperm(
                    len(valid_columns), device=x.device
                )[:sample_size]
                x[:, :, valid_columns[random_indices], :] = 0

        # Add noise only to the time series (positions 1:), not the distance (position 0)
        x[:, :, :, 1:] = self._add_noise(x[:, :, :, 1:])
        return x

    def select_random_channels(
        self, x: torch.Tensor, num_channels: int
    ) -> torch.Tensor:
        all_x = []
        distances = x[:, 0, :, 0]
        for single_sample, distance in zip(x, distances):
            valid_rows = torch.where(distance != 0)[0]
            if len(valid_rows) == 0:
                raise ValueError("SPES input contains a sample with no valid channels.")
            replacement = len(valid_rows) < num_channels
            p = torch.ones(len(valid_rows), device=x.device) / len(valid_rows)
            idx = p.multinomial(num_samples=num_channels, replacement=replacement)
            channels = valid_rows[idx].sort()[0]
            all_x.append(single_sample[:, channels])
        return torch.stack(all_x, dim=0)

    @staticmethod
    def create_key_padding_mask(distances: torch.Tensor) -> torch.Tensor:
        key_padding_mask = distances == 0
        # Prepend a False column for the class token, which is never masked
        false_column = torch.zeros(
            distances.size(0), 1, dtype=torch.bool, device=distances.device
        )
        return torch.cat([false_column, key_padding_mask], dim=1)

    def _selected_modes(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean and self.std:
            return x[:, :, :, 1:]
        if self.mean:
            return x[:, :1, :, 1:]
        return x[:, 1:, :, 1:]

    def _selected_modes_with_distance(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean and self.std:
            return x
        if self.mean:
            return x[:, :1]
        return x[:, 1:]

    def prepare_channels(self, x: torch.Tensor) -> torch.Tensor:
        mode_count = int(self.mean) + int(self.std)
        if self.conv_embedding:
            conv_input = self._selected_modes(x)
            batch_size, modes, channels, timesteps = conv_input.shape
            # Merge batch and channel dims so each channel is processed independently
            conv_input = conv_input.reshape(-1, modes, timesteps)

            late_output = self.msresnet(conv_input)
            late_output = late_output.reshape(batch_size, channels, -1)

            if not self.mlp_embedding:
                return late_output

            # Prepend a short flattened prefix to the ResNet embedding
            prefix = self._selected_modes_with_distance(x)[
                :, :, :, : self.max_mlp_timesteps
            ]
            if prefix.shape[-1] < self.max_mlp_timesteps:
                pad = self.max_mlp_timesteps - prefix.shape[-1]
                prefix = nn.functional.pad(prefix, (0, pad))
            prefix = prefix.swapaxes(1, 2).reshape(batch_size, channels, -1)
            return torch.cat([prefix, late_output], dim=-1)

        selected = self._selected_modes(x)
        batch_size, _, channels, timesteps = selected.shape
        if timesteps < self.expected_timesteps:
            selected = nn.functional.pad(selected, (0, self.expected_timesteps - timesteps))
        elif timesteps > self.expected_timesteps:
            selected = selected[:, :, :, : self.expected_timesteps]
        return selected.swapaxes(1, 2).reshape(
            batch_size, channels, mode_count * self.expected_timesteps
        )


class SPESResNet(BaseModel):
    """Multi-scale 1D CNN classifier for CCEP SPES SOZ localization.

    Randomly sub-samples a fixed number of channels from the input tensor and
    passes them through a :class:`MultiScaleResNet1D` followed by a linear
    classifier. ``input_type="divergent"`` uses the stimulation-channel view
    (``X_stim``) and ``input_type="convergent"`` uses the recording-channel
    view (``X_recording``).

    Args:
        dataset: The dataset to train the model on. Used to infer label keys
            and output size.
        input_type: Either ``"divergent"`` (stimulation view) or
            ``"convergent"`` (recording view). Default is ``"divergent"``.
        input_channels: Number of channels randomly sampled per forward pass.
            Default is 40.
        stim_key: Key in the sample batch for the stimulation tensor.
            Default is ``"X_stim"``.
        recording_key: Key in the sample batch for the recording tensor.
            Default is ``"X_recording"``.
        noise_std: Std of Gaussian noise added to non-distance features during
            training. Default is 0.1.
        dropout_rate: Dropout probability inside the ResNet. Default is 0.2.
        pos_weight: Optional positive-class weight for
            ``binary_cross_entropy_with_logits``. Default is ``None``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`MultiScaleResNet1D`.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> n_ch, n_t = 30, 509
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "visit_id": f"v{i}",
        ...         "X_stim": np.random.randn(2, n_ch, n_t).astype(np.float32),
        ...         "X_recording": np.random.randn(2, n_ch, n_t).astype(np.float32),
        ...         "electrode_lobes": np.array([i % 7], dtype=np.int64),
        ...         "electrode_coords": np.random.randn(3).astype(np.float32),
        ...         "soz": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"X_stim": "tensor", "X_recording": "tensor",
        ...                   "electrode_lobes": "tensor", "electrode_coords": "tensor"},
        ...     output_schema={"soz": "binary"},
        ...     dataset_name="test",
        ... )
        >>> from pyhealth.models import SPESResNet
        >>> model = SPESResNet(dataset=dataset, input_channels=10)
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> ret = model(**batch)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_type: str = "divergent",
        input_channels: int = 40,
        stim_key: str = "X_stim",
        recording_key: str = "X_recording",
        noise_std: float = 0.1,
        dropout_rate: float = 0.2,
        pos_weight: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(dataset=dataset)
        if input_type not in {"divergent", "convergent"}:
            raise ValueError("input_type must be 'divergent' or 'convergent'.")
        if len(self.label_keys) != 1:
            raise ValueError("SPESResNet supports exactly one label key.")

        self.input_type = input_type
        self.input_channels = input_channels
        self.stim_key = stim_key
        self.recording_key = recording_key
        self.noise_std = noise_std
        self.label_key = self.label_keys[0]
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([pos_weight], dtype=torch.float32),
            )

        self.msresnet = MultiScaleResNet1D(
            input_channel=input_channels,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.fc = nn.Linear(MultiScaleResNet1D.output_dim, self.get_output_size())

    @property
    def feature_key(self) -> str:
        return self.stim_key if self.input_type == "divergent" else self.recording_key

    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std <= 0:
            return x
        return x + torch.randn_like(x) * self.noise_std

    def _sample_channels(self, x: torch.Tensor) -> torch.Tensor:
        # Distance is stored at position 0 of the last dim
        distances = x[:, 0, :, 0]
        all_x = []

        for single_sample, distance in zip(x, distances):
            valid_rows = torch.where(distance != 0)[0]
            if len(valid_rows) == 0:
                raise ValueError("SPES input contains a sample with no valid channels.")
            # Sample with replacement when fewer valid channels exist than requested
            replacement = len(valid_rows) < self.input_channels
            p = torch.ones(len(valid_rows), device=x.device) / len(valid_rows)
            idx = p.multinomial(
                num_samples=self.input_channels, replacement=replacement
            )
            channels = valid_rows[idx].sort()[0]
            # Use only the mean mode (index 0) and skip the distance (last dim index 0)
            all_x.append(single_sample[0, channels, 1:])

        return torch.stack(all_x, dim=0)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        if self.pos_weight is None:
            return self.get_loss_function()(logits, y_true)
        return F.binary_cross_entropy_with_logits(
            logits,
            y_true,
            pos_weight=self.pos_weight.to(logits.device),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = kwargs[self.feature_key].to(self.device)
        if self.training:
            x = x.clone()
            x[:, :, :, 1:] = self._add_noise(x[:, :, :, 1:])

        x = self._sample_channels(x)
        emb = self.msresnet(x)
        logits = self.fc(emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self._compute_loss(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results


class SPESTransformer(BaseModel):
    """Transformer-based classifier over SPES CCEP response-channel embeddings.

    One or more :class:`SPESResponseEncoder` networks process the stimulation
    and/or recording tensors; their class-token outputs are concatenated and
    passed through a linear classifier.

    Args:
        dataset: The dataset to train the model on. Used to infer label keys
            and output size.
        net_configs: List of dicts, each specifying one encoder. Required keys:

            - ``"type"`` (``"divergent"`` or ``"convergent"``): selects
              ``X_stim`` or ``X_recording`` as input.
            - ``"mean"`` (bool): include the mean response mode.
            - ``"std"`` (bool): include the std response mode.

        dropout_rate: Dropout probability applied in each encoder and before
            the final linear layer. Default is 0.5.
        stim_key: Key in the sample batch for the stimulation tensor.
            Default is ``"X_stim"``.
        recording_key: Key in the sample batch for the recording tensor.
            Default is ``"X_recording"``.
        pos_weight: Optional positive-class weight for
            ``binary_cross_entropy_with_logits``. Default is ``None``.
        **kwargs: Additional keyword arguments forwarded to each
            :class:`SPESResponseEncoder` (e.g. ``embedding_dim``,
            ``num_layers``, ``random_channels``).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset
        >>> n_ch, n_t = 30, 509
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "visit_id": f"v{i}",
        ...         "X_stim": np.random.randn(2, n_ch, n_t).astype(np.float32),
        ...         "X_recording": np.random.randn(2, n_ch, n_t).astype(np.float32),
        ...         "electrode_lobes": np.array([i % 7], dtype=np.int64),
        ...         "electrode_coords": np.random.randn(3).astype(np.float32),
        ...         "soz": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"X_stim": "tensor", "X_recording": "tensor",
        ...                   "electrode_lobes": "tensor", "electrode_coords": "tensor"},
        ...     output_schema={"soz": "binary"},
        ...     dataset_name="test",
        ... )
        >>> from pyhealth.models import SPESTransformer
        >>> net_configs = [{"type": "divergent", "mean": True, "std": True}]
        >>> model = SPESTransformer(dataset=dataset, net_configs=net_configs,
        ...                         embedding_dim=32, random_channels=10)
        >>> from pyhealth.datasets import get_dataloader
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> ret = model(**batch)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        net_configs: list[dict],
        dropout_rate: float = 0.5,
        stim_key: str = "X_stim",
        recording_key: str = "X_recording",
        pos_weight: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("SPESTransformer supports exactly one label key.")

        self.net_configs = net_configs
        self.stim_key = stim_key
        self.recording_key = recording_key
        self.label_key = self.label_keys[0]
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor([pos_weight], dtype=torch.float32),
            )

        self.eegnets = nn.ModuleList(
            [
                SPESResponseEncoder(
                    mean=net_config["mean"],
                    std=net_config["std"],
                    dropout_rate=dropout_rate,
                    **kwargs,
                )
                for net_config in net_configs
            ]
        )

        embedding_dim = kwargs.get("embedding_dim", 64)
        total_feature_size = embedding_dim * len(net_configs)
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_feature_size, self.get_output_size()),
        )

        # Explicitly initialize the classifier head
        nn.init.xavier_uniform_(self.fc[1].weight)
        if self.fc[1].bias is not None:
            nn.init.zeros_(self.fc[1].bias)

    def _get_input(self, net_config: dict, kwargs: dict) -> torch.Tensor:
        input_type = net_config["type"]
        if input_type == "divergent":
            return kwargs[self.stim_key].to(self.device)
        if input_type == "convergent":
            return kwargs[self.recording_key].to(self.device)
        raise ValueError(
            f"Invalid type '{input_type}' in net_configs; expected 'convergent' or 'divergent'."
        )

    def _compute_loss(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        if self.pos_weight is None:
            return self.get_loss_function()(logits, y_true)
        return F.binary_cross_entropy_with_logits(
            logits,
            y_true,
            pos_weight=self.pos_weight.to(logits.device),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        processed_inputs = [
            eegnet(self._get_input(net_config, kwargs))
            for net_config, eegnet in zip(self.net_configs, self.eegnets)
        ]
        emb = torch.cat(processed_inputs, dim=1)
        logits = self.fc(emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self._compute_loss(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results
