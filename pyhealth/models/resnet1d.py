"""1-D ResNet for ECG classification.

Port of the ``ResNet1d`` backbone + ``HeadModule`` prediction head from
*In-depth Benchmarking of Deep Neural Network Architectures for ECG
Diagnosis* (Nonaka & Seita, 2021,
https://proceedings.mlr.press/v149/nonaka21a.html), whose reference code adapts
`torchvision's ResNet <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
by replacing every convolution with a 1-D convolution
(`seitalab/dnn_ecg_comparison <https://github.com/seitalab/dnn_ecg_comparison>`_,
``experiment/codes/architectures/resnet1d.py`` and ``head_module.py``).

Not to be confused with :class:`~pyhealth.models.ResNet1dBackbone` used by
:class:`~pyhealth.models.ProtoECGNet`, which is a feature extractor that
keeps the temporal axis; this model is a complete classifier: it pools to a
fixed-size feature vector and ends in the benchmark's prediction head.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..datasets import SampleDataset
from .base_model import BaseModel


def _conv3x1(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
) -> nn.Conv1d:
    """3x1 (kernel-size-3) convolution with same-length padding."""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
        bias=False,
    )


def _conv1x1(
    in_planes: int, out_planes: int, stride: int = 1
) -> nn.Conv1d:
    """1x1 convolution, used for residual projections."""
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock1D(nn.Module):
    """Residual block with two 3x1 convolutions (ResNet-18/34 layout).

    Args:
        in_planes: Number of input channels.
        planes: Number of middle/output channels.
        stride: Stride of the first convolution.
        downsample: Optional projection for the residual branch.
        norm_layer: Normalization layer constructor.

    Examples:
        >>> import torch
        >>> block = BasicBlock1D(64, 64)
        >>> block(torch.randn(2, 64, 100)).shape
        torch.Size([2, 64, 100])
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x1(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Adds a skip connection that is projected when stride > 1."""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Bottleneck1D(nn.Module):
    """Residual block with 1x1 - 3x1 - 1x1 convolutions (ResNet-50 layout).

    Args:
        in_planes: Number of input channels.
        planes: Number of middle channels; output has ``planes * 4``.
        stride: Stride of the 3x1 convolution.
        downsample: Optional projection for the residual branch.
        norm_layer: Normalization layer constructor.

    Examples:
        >>> import torch
        >>> block = Bottleneck1D(256, 64)
        >>> block(torch.randn(2, 256, 100)).shape
        torch.Size([2, 256, 100])
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = _conv3x1(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Adds a skip connection that is projected when stride > 1."""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet1D(BaseModel):
    """1-D ResNet ECG classifier from the Nonaka & Seita benchmark.

    A stem (``Conv1d(k=7, s=2)`` -> BatchNorm -> ReLU -> MaxPool) feeds four
    residual stages with strides 1, 2, 2, 2; the feature map is pooled with
    :class:`~torch.nn.AdaptiveAvgPool1d` to a single step, projected to
    ``hidden_dim`` (the benchmark's ``backbone_out_dim = 256``), and passed
    through the benchmark's prediction head
    (``Linear`` -> ReLU -> BatchNorm -> Dropout -> ``Linear``). Because the
    pooling is adaptive, any signal length works — a full 10 s record and a
    2.5 s crop both run unchanged.

    Compatible with tensor samples whose key is ``signal`` of shape
    ``(in_channels, T)`` — e.g. the PTB-XL tasks
    :class:`~pyhealth.tasks.PTBXLMultilabelClassification` and
    :class:`~pyhealth.tasks.PTBXLMulticlassClassification`. The number of
    output classes, the loss and the output activation all come from the
    dataset's label processor, so the same class serves binary, multiclass
    and multilabel tasks.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` with a
            ``signal`` input feature and exactly one label output.
        depth: ResNet depth: ``18`` or ``34`` use
            :class:`BasicBlock1D` with ``[2, 2, 2, 2]`` / ``[3, 4, 6, 3]``
            blocks; ``50`` uses :class:`Bottleneck1D` with ``[3, 4, 6, 3]``.
        in_channels: Number of ECG leads (PTB-XL: 12).
        hidden_dim: Backbone output dimension; the benchmark uses 256.
        head_hidden_dim: Prediction-hidden width; the benchmark uses 128.
        head_dropout: Dropout probability in the prediction head; the
            benchmark uses 0.25.
        training_crop_len: When set and training, each sample longer than
            this is cropped to it at a random offset (the benchmark uses
            1250 samples = 2.5 s at 500 Hz, resampled every epoch).
        eval_window_len: When set and evaluating, a signal longer than this
            is run through the network in overlapping windows of this
            length and the class logits are aggregated by element-wise
            maximum across windows (the benchmark's ``SubsampleEval`` +
            ``aggregator``, 1250 samples). The maximum is taken on the
            logits, before the label processor's activation.
        eval_window_stride: Stride between evaluation windows; defaults to
            ``eval_window_len // 2`` (benchmark: 625 samples = 1.25 s).
            Must not exceed ``eval_window_len``.
        class_weight: Per-class weights for the multiclass cross-entropy,
            in label-vocabulary order (the benchmark uses
            ``[1, n_target/n_normal, n_other/n_normal]`` computed from the
            training split, see
            :func:`~pyhealth.tasks.compute_multiclass_class_weights`).
            Only valid when the dataset's label processor is multiclass.

    Attributes:
        mode (str): Label kind inherited from
            :class:`~pyhealth.models.BaseModel` (e.g. ``"multilabel"``).

    Note:
        Convolution weights use Kaiming (fan_out, ReLU) initialization and
        normalization layers weight=1 / bias=0, matching the benchmark's
        released code.

    Note:
        Evaluation windowing pads the signal tail with zeros so the last
        window starting before the end still fits, exactly like the
        benchmark's ``SubsampleEval._pad_signal``; with the benchmark
        settings a 5000-sample record yields 8 windows of 1250 (stride
        625). The random training crop draws from torch's global RNG, so
        ``torch.manual_seed(...)`` makes runs reproducible (the original
        seeded NumPy instead).

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "signal": torch.zeros(12, 400),
        ...      "label": 0},
        ...     {"patient_id": "p1", "signal": torch.zeros(12, 400),
        ...      "label": 1},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = ResNet1D(dataset=dataset)
        >>> model.mode
        'binary'
        >>> out = model(signal=torch.zeros(2, 12, 400),
        ...             label=torch.tensor([[0.0], [1.0]]))
        >>> out["logit"].shape
        torch.Size([2, 1])
        >>> sorted(out)
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    #: depth -> (blocks per stage, block class), as in the benchmark's
    #: resnet1d18 / resnet1d34 / resnet1d50 constructors.
    DEPTHS: ClassVar[dict[int, tuple[list[int], type[nn.Module]]]] = {
        18: ([2, 2, 2, 2], BasicBlock1D),
        34: ([3, 4, 6, 3], BasicBlock1D),
        50: ([3, 4, 6, 3], Bottleneck1D),
    }

    def __init__(
        self,
        dataset: SampleDataset,
        depth: int = 18,
        in_channels: int = 12,
        hidden_dim: int = 256,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.25,
        training_crop_len: int | None = None,
        eval_window_len: int | None = None,
        eval_window_stride: int | None = None,
        class_weight: Sequence[float] | None = None,
    ) -> None:
        if depth not in self.DEPTHS:
            raise ValueError(
                f"depth must be one of {sorted(self.DEPTHS)}, got {depth!r}"
            )
        if "signal" not in self._feature_keys(dataset):
            raise ValueError(
                "ResNet1D expects a 'signal' input feature (e.g. from a "
                "PTB-XL task); this dataset has features "
                f"{self._feature_keys(dataset)}."
            )
        for name, value in (
            ("training_crop_len", training_crop_len),
            ("eval_window_len", eval_window_len),
            ("eval_window_stride", eval_window_stride),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            eval_window_len is not None
            and eval_window_stride is not None
            and int(eval_window_stride) > int(eval_window_len)
        ):
            raise ValueError(
                "eval_window_stride must not exceed eval_window_len"
            )
        super().__init__(dataset)
        if class_weight is not None:
            if self.mode != "multiclass":
                raise ValueError(
                    "class_weight weights the multiclass cross-entropy; "
                    f"this dataset's label mode is {self.mode!r}."
                )
            weights = torch.as_tensor(class_weight, dtype=torch.float32)
            if weights.numel() != self.label_processor.output_size():
                raise ValueError(
                    "class_weight must have one entry per class; expected "
                    f"{self.label_processor.output_size()}, got "
                    f"{weights.numel()}."
                )
            self.register_buffer("class_weight", weights)
        else:
            self.register_buffer("class_weight", None)
        self.signal_key = "signal"
        self.label_key = self.label_keys[0]
        self.depth = depth
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.training_crop_len = training_crop_len
        self.eval_window_len = eval_window_len
        self.eval_window_stride = eval_window_stride

        layers, block = self.DEPTHS[depth]
        self._planes = 64
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(512 * block.expansion, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(head_hidden_dim),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, self.label_processor.output_size()),
        )
        self._init_weights()

    @staticmethod
    def _feature_keys(dataset: SampleDataset) -> list[str]:
        """Input feature names of the dataset (empty when dataset is None)."""
        if dataset is None:
            return []
        return list(getattr(dataset, "input_schema", {}) or {})

    def _make_layer(
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Builds one residual stage with a projection downsample."""
        downsample: nn.Module | None = None
        if stride != 1 or self._planes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self._planes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self._planes, planes, stride, downsample)]
        self._planes = planes * block.expansion
        layers.extend(block(self._planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming fan_out init for convs, 1/0 for norm layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def backbone(self, signal: Tensor) -> Tensor:
        """Runs stem + stages + pooling + projection.

        Args:
            signal: Batch of ECG signals, shape ``(batch, in_channels, T)``.

        Returns:
            Feature tensor of shape ``(batch, hidden_dim)``.
        """
        x = self.maxpool(self.relu(self.bn1(self.conv1(signal))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.projection(torch.flatten(x, 1))

    @staticmethod
    def _random_crop(signal: Tensor, length: int) -> Tensor:
        """Crops each sample to ``length`` at an independent random offset."""
        total = signal.shape[-1]
        starts = torch.randint(
            0, total - length + 1, (signal.shape[0],), device=signal.device
        )
        indices = starts[:, None] + torch.arange(length, device=signal.device)
        indices = indices[:, None, :].expand(-1, signal.shape[1], -1)
        return torch.gather(signal, 2, indices)

    def _windowed_logits(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        """Runs ``eval_window_len`` windows and aggregates by max of logits.

        Mirrors the benchmark's ``SubsampleEval`` (window stride
        ``eval_window_stride``, default half the window; the tail is
        zero-padded so the last window starting before the end fits) and
        ``aggregator`` (element-wise maximum over windows, on the logits).

        Returns:
            ``(feature, logits)``: the mean backbone feature across
            windows and the aggregated logits, both ``(batch, ...)``.
        """
        length = int(self.eval_window_len)
        stride = int(self.eval_window_stride or length // 2)
        total = signal.shape[-1]
        n_windows = (total - 1) // stride + 1
        pad = (n_windows - 1) * stride + length - total
        if pad > 0:
            signal = F.pad(signal, (0, pad))
        windows = signal.unfold(2, length, stride)  # (B, C, W, length)
        batch, channels = windows.shape[0], windows.shape[1]
        windows = windows.permute(0, 2, 1, 3).reshape(-1, channels, length)
        feature = self.backbone(windows)
        logits = self.head(feature)
        feature = feature.view(batch, n_windows, -1).mean(dim=1)
        logits = logits.view(batch, n_windows, -1).max(dim=1).values
        return feature, logits

    def forward_from_embedding(self, feature: Tensor) -> dict[str, Tensor]:
        """Applies the prediction head to backbone features.

        ``feature`` is the raw ``(batch, in_channels, T)`` signal, matching
        :class:`~pyhealth.models.BaseModel`'s convention for models that
        take dense inputs.

        Args:
            feature: Batch of ECG signals.

        Returns:
            The same dict as :meth:`forward`.
        """
        return self.forward(signal=feature)

    def forward(
        self, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Forward pass on a batch of ECG signals.

        Args:
            **kwargs: Must contain ``signal``: a tensor of shape
                ``(batch, in_channels, T)``. The label (``label`` /
                ``labels``, i.e. ``self.label_key``) is optional; when
                present it also yields ``loss`` and ``y_true``. A truthy
                ``embed`` keyword adds the backbone features under
                ``embed`` (the mean over windows when evaluation windowing
                is active).

        Returns:
            Dict with ``logit`` and ``y_prob``, plus ``loss`` / ``y_true``
            (and optionally ``embed``) as described above.

        Raises:
            KeyError: If ``signal`` is not in ``kwargs``.
        """
        if self.signal_key not in kwargs:
            raise KeyError(
                f"ResNet1D.forward requires a '{self.signal_key}' input "
                f"tensor; got keys {sorted(kwargs)}."
            )
        signal = kwargs[self.signal_key]
        if not isinstance(signal, Tensor):
            signal = torch.as_tensor(signal)
        signal = signal.to(self.device).float()

        crop_len = self.training_crop_len
        if self.training and crop_len and signal.shape[-1] > crop_len:
            signal = self._random_crop(signal, crop_len)

        windowed = (
            not self.training
            and self.eval_window_len is not None
            and signal.shape[-1] > self.eval_window_len
        )
        if windowed:
            feature, logits = self._windowed_logits(signal)
        else:
            feature = self.backbone(signal)
            logits = self.head(feature)

        y_prob = self.label_processor.activation(logits)
        results = {"y_prob": y_prob, "logit": logits}
        # The label is optional, so the model can predict without it.
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            if self.class_weight is not None:
                results["loss"] = self.label_processor.loss(
                    logits, y_true, weight=self.class_weight
                )
            else:
                results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = feature
        return results
