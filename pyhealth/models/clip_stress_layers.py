"""Building blocks for the Yang et al. (CHIL 2025) reference architecture.

Independent implementations of channel attention, preactivation convolutional
residual blocks and residual projection heads. No reference repository imports.
"""

import torch
from torch import nn
from torch.nn import functional as F


class StressProjection(nn.Module):
    """Linear projection followed by a residual GELU MLP and LayerNorm.

    Examples:
        >>> layer = StressProjection(32, 128)
        >>> layer(torch.zeros(2, 32)).shape
        torch.Size([2, 128])
    """

    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Linear(input_dim, output_dim)
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        return self.layer_norm(projected + self.dropout(self.fc(F.gelu(projected))))


class StressConvBlock(nn.Module):
    """Two width-three convolutions; symmetric zero padding on the shortcut.

    Examples:
        >>> block = StressConvBlock(8, 16, first=False, dropout=0.5)
        >>> block(torch.zeros(2, 8, 21)).shape
        torch.Size([2, 16, 21])
    """

    def __init__(self, input_dim, output_dim, first, dropout):
        super().__init__()
        self.first = first
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.conv1 = nn.Conv1d(input_dim, output_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.extra = output_dim - input_dim

    def forward(self, x):
        residual = x
        if self.extra:
            residual = F.pad(
                x.transpose(1, 2), (self.extra // 2, self.extra - self.extra // 2)
            ).transpose(1, 2)
        if not self.first:
            x = self.dropout(F.relu(self.bn1(x)))
        x = self.conv1(x)
        x = self.conv2(self.dropout(F.relu(self.bn2(x))))
        return x + residual


class StressTabularResNet(nn.Module):
    """Reference tabular ResNet1D: stride one, filters doubled every four blocks.

    Examples:
        >>> encoder = StressTabularResNet(21, 64, num_layers=3)
        >>> encoder(torch.zeros(2, 21)).shape
        torch.Size([2, 64])
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        width = 2 * hidden_dim
        self.stem = nn.Conv1d(1, width, 3, padding=1)
        self.stem_bn = nn.BatchNorm1d(width)
        blocks = []
        for index in range(num_layers):
            next_width = width * 2 if index and index % 4 == 0 else width
            blocks.append(StressConvBlock(width, next_width, index == 0, dropout))
            width = next_width
        self.blocks = nn.Sequential(*blocks)
        self.final_bn = nn.BatchNorm1d(width)
        self.output = nn.Linear(width, hidden_dim)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x.unsqueeze(1))))
        x = self.blocks(x)
        return self.output(F.relu(self.final_bn(x)).mean(-1))


def stress_contrastive_loss(sensor, text, temperature, objective):
    """Reference soft targets or the paper's paired cosine objective.

    The reference targets intentionally remain differentiable and use tau as
    a multiplier, while cross-modal logits use tau as a divisor.

    Examples:
        >>> vectors = torch.eye(3)
        >>> logits, loss = stress_contrastive_loss(
        ...     vectors, vectors, 0.07, "reference_soft")
        >>> logits.shape, loss.ndim
        (torch.Size([3, 3]), 0)
    """
    if objective == "paper_cosine":
        sensor, text = F.normalize(sensor, dim=-1), F.normalize(text, dim=-1)
        logits = text @ sensor.T / temperature
        targets = torch.arange(len(sensor), device=sensor.device)
        loss = (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
        ) / 2
    else:
        logits = text @ sensor.T / temperature
        targets = ((sensor @ sensor.T + text @ text.T) * temperature / 2).softmax(-1)
        loss = (
            -(
                (targets * logits.log_softmax(-1)).sum(-1)
                + (targets.T * logits.T.log_softmax(-1)).sum(-1)
            ).mean()
            / 2
        )
    return logits, loss
