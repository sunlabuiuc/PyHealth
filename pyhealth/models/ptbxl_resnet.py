"""1D ResNet models for PTB-XL ECG classification, with optional TaskAug.

Paper (TaskAug):
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag, and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms."
    Proceedings of the Conference on Health, Inference, and Learning (CHIL),
    PMLR 174:282-310, 2022. https://proceedings.mlr.press/v174/raghu22a.html

Author:
    Lisha Ranganath (GitHub: @LishaRanganath) and Meghaben D Dhalwadi

Description:
    ``PTBXLResNet`` is a small 1D ResNet that classifies a 12-lead ECG.
    ``TaskAugResNet`` is the same network preceded by the TaskAug augmentation
    policy (``pyhealth.models.taskaug``), applied only while training.
"""

import torch
from torch import nn

from .base_model import BaseModel
from .taskaug import aug_policy


def _to_taskaug_length(x: torch.Tensor) -> torch.Tensor:
    """Match the signal length used by the original TaskAug experiments.

    PTB-XL records at 500 Hz have 5000 samples. The original TaskAug code
    keeps every second sample (250 Hz, 2500 samples) and drops the last 4,
    giving 2496. Signals of any other length are returned unchanged.

    Args:
        x: ECG batch of shape ``(batch, 12, length)``.

    Returns:
        torch.Tensor: The batch at ``(batch, 12, 2496)`` if ``length`` was
        5000, otherwise ``x`` itself.

    Examples:
        >>> _to_taskaug_length(torch.zeros(2, 12, 5000)).shape
        torch.Size([2, 12, 2496])
    """
    if x.shape[-1] == 5000:
        x = x[:, :, ::2]
        x = x[:, :, :-4]
    return x


class ResBlock1D(nn.Module):
    """Residual block of two 1D convolutions with batch normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first convolution. Defaults to 1.

    Examples:
        >>> ResBlock1D(32, 64, stride=2)(torch.zeros(2, 32, 100)).shape
        torch.Size([2, 64, 50])
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7, stride=stride, padding=3
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        """Applies the residual block to ``x`` of shape ``(batch, channels, length)``."""
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class PTBXLResNet(BaseModel):
    """Small 1D ResNet for PTB-XL ECG classification.

    The model takes a 12-lead signal under the dataset's single input key and
    predicts the dataset's single label. Signals with 5000 samples (500 Hz)
    are first reduced to the 2496 samples used by the TaskAug experiments, so
    this baseline sees the same input as :class:`TaskAugResNet`. Use
    ``PTBXLDataset(sampling_rate=500)`` for a fair comparison.

    Args:
        dataset (SampleDataset): Dataset with one tensor input of shape
            ``(12, length)`` and one label, for example from a PTB-XL task such
            as :class:`~pyhealth.tasks.PTBXLHYP`.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset, get_dataloader
        >>> from pyhealth.tasks import PTBXLHYP
        >>> from pyhealth.models import PTBXLResNet
        >>> dataset = PTBXLDataset(root="/data/ptb-xl/1.0.3", sampling_rate=500)  # doctest: +SKIP
        >>> task = PTBXLHYP(dataset.scp_statements_path)  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> model = PTBXLResNet(dataset=samples)  # doctest: +SKIP
        >>> batch = next(iter(get_dataloader(samples, batch_size=8)))  # doctest: +SKIP
        >>> sorted(model(**batch))  # doctest: +SKIP
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(self, dataset):
        super().__init__(dataset=dataset)
        self.encoder = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ResBlock1D(32, 32),
            ResBlock1D(32, 64, stride=2),
            ResBlock1D(64, 128, stride=2),
            ResBlock1D(128, 256, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, self.label_processor.output_size())

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Runs a forward pass.

        Args:
            **kwargs: The batch from a PyHealth dataloader; it must hold the
                signal and the label.

        Returns:
            dict[str, torch.Tensor]: ``loss`` (scalar), ``y_prob`` and
            ``logit`` of shape ``(batch, 1)``, and ``y_true``.
        """
        x = kwargs[self.feature_keys[0]].to(self.device)
        x = _to_taskaug_length(x)
        emb = self.encoder(x).squeeze(-1)
        logits = self.fc(emb)
        y_true = kwargs[self.label_keys[0]].to(self.device)
        loss = self.label_processor.loss(logits, y_true)
        y_prob = self.label_processor.activation(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}


class TaskAugResNet(PTBXLResNet):
    """:class:`PTBXLResNet` with the TaskAug augmentation policy.

    During training each batch passes through ``k`` chained TaskAug stages
    (temporal warp, baseline wander, Gaussian noise, random crop, random
    displacement, magnitude scaling or no-op, mixed by learned weights) with
    class-conditional magnitudes. In evaluation mode the signal is not
    augmented.

    Note:
        This class applies the augmentation but does not reproduce the paper's
        bilevel optimization, where the augmentation magnitudes are updated
        from a validation-set hypergradient. Here the magnitudes are ordinary
        parameters trained jointly with the network on the training loss.
        Minimizing the training loss tends to shrink the magnitudes and favor
        the weakest operations, so the policy can drift toward little or no
        augmentation. In training mode the signal must be 500 Hz (5000
        samples) or already 2496 samples long, otherwise ``ValueError`` is
        raised.

    Args:
        dataset (SampleDataset): As for :class:`PTBXLResNet`.
        k (int): Number of chained augmentation stages. Defaults to 2.

    Examples:
        >>> from pyhealth.models import TaskAugResNet
        >>> model = TaskAugResNet(dataset=samples, k=3)  # doctest: +SKIP
        >>> len(model.taskaug.sub_policies[0].stages)  # doctest: +SKIP
        3
    """

    def __init__(self, dataset, k=2):
        super().__init__(dataset)
        self.k = k
        self.taskaug = aug_policy.full_policy(
            num_sub_policies=1,
            operation_count=k,
            num_chunks=1,
            learn_mag=True,
            learn_prob=False,
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Runs a forward pass, augmenting the signal first when training.

        Args:
            **kwargs: The batch from a PyHealth dataloader.

        Returns:
            dict[str, torch.Tensor]: Same keys as :meth:`PTBXLResNet.forward`.
        """
        x = kwargs[self.feature_keys[0]].to(self.device)
        x = _to_taskaug_length(x)
        if self.training and x.shape[-1] != 2496:
            raise ValueError(
                "TaskAugResNet needs 500 Hz PTB-XL signals (5000 samples, "
                "trimmed to 2496) in training mode, got length "
                f"{x.shape[-1]}. Load the dataset with sampling_rate=500."
            )
        y_true = kwargs[self.label_keys[0]].to(self.device)
        if self.training:
            # TaskAug conditions its magnitudes on integer class labels.
            x = self.taskaug(x, y_true.view(-1).long())
        emb = self.encoder(x).squeeze(-1)
        logits = self.fc(emb)
        loss = self.label_processor.loss(logits, y_true)
        y_prob = self.label_processor.activation(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
