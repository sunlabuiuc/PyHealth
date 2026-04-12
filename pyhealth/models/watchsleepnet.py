import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class ResidualBlock(nn.Module):
    """Residual block with strided convolution for temporal downsampling
        and channel expansion.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size. Default is 3.
        stride: Stride for the first convolution. Default is 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x + residual


class TCNBlock(nn.Module):
    """A single TCN block with dilated causal convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size. Default is 3.
        dilation: Initial dilation factor, doubled after each layer. Default is 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        num_layers = 3
        self.layers = []
        current_channels = in_channels

        for _ in range(num_layers):
            conv = nn.Conv1d(
                current_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            )
            bn = nn.BatchNorm1d(out_channels)
            relu = nn.ReLU(inplace=True)
            dropout = nn.Dropout(0.2)
            self.layers.append(nn.Sequential(conv, bn, relu, dropout))
            current_channels = out_channels
            dilation *= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FeatureExtractor(nn.Module):
    """Feature extractor using residual blocks for each epoch independently.

    Args:
        in_channels: Number of input channels (e.g., 1 for a single IBI signal).
    """

    def __init__(
        self,
        in_channels: int
    ):
        super().__init__()

        blocks = []
        out_channels = 16
        stride = 1
        res_channels = [32, 64, 128, 256]

        blocks.append(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=7, stride=stride, padding=3)
        )

        blocks.append(nn.ReLU(inplace=True))
        current_channels = out_channels

        for out_ch in res_channels:
            blocks.append(ResidualBlock(current_channels, out_ch, stride=4))
            current_channels = out_ch

        self.res_blocks = nn.Sequential(*blocks)
        self.conv = nn.Conv1d(256, 256, kernel_size=3, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        return x


class WatchSleepNet(BaseModel):
    """WatchSleepNet model for sleep stage classification.

    Architecture:
        1. Residual convolutional blocks to extract multi-level spatial features
           while preserving info through skip connections.
        2. TCN to address longer-range dependencies.
        3. Bi-directional LSTM to capture temporal dependencies in both directions.
        4. Multi-head attention mechanism to focus on important time steps and features.
        5. Fully connected layer with softmax for classification.

    Args:
        dataset: A SampleDataset object.
        seq_sample_size: Number of samples per epoch (e.g., 750 for 30s at 25Hz).
            Default is 750.
        num_features: Number of input signal channels per epoch. Default is 1.
        num_classes: Number of output sleep stage classes. Default is 3.
        tcn_kernel_size: Kernel size for TCN layers. Default is 3.
        lstm_hidden_size: Hidden size for each LSTM direction. Default is 128.
        lstm_num_layers: Number of LSTM layers. Default is 2.
        num_heads: Number of multi-head attention heads. Default is 4.

    Note:
        Default hyperparameters are based on the original WatchSleepNet paper

    Examples:
        >>> from pyhealth.trainer import Trainer
        >>> from pyhealth.datasets import DREAMTDataset
        >>> from pyhealth.datasets import get_dataloader, split_by_patient
        >>> from pyhealth.models import WatchSleepNet
        >>> from pyhealth.tasks import SleepStagingDREAMT
        >>>
        >>> # Load DREAMT dataset
        >>> dataset = DREAMTDataset(root="/path/to/dreamt")
        >>> task = SleepStagingDREAMT()
        >>> sample_dataset = dataset.set_task(task=task)
        >>>
        >>> # Initialize model
        >>> model = WatchSleepNet(dataset=sample_dataset)
        >>>
        >>> # Create dataloaders
        >>> train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
        >>> val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
        >>> test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
        >>>
        >>> # Train model
        >>> trainer = Trainer(
        ...     model=model,
        ...     metrics=["cohen_kappa", "f1_macro", "f1_weighted", "accuracy"],
        ...     exp_name="watchsleepnet_sleep_staging"
        ... )

        >>> trainer.train(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     test_dataloader=test_loader,
        ...     epochs=_EPOCHS,
        ...     monitor="cohen_kappa",
        ...     monitor_criterion="max",
        ...     weight_decay=_DECAY_WEIGHT
        ... )
    """

    def __init__(
        self,
        dataset=None,
        seq_sample_size: int = 750,
        num_features: int = 1,
        num_classes: int = 3,
        tcn_kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        num_heads: int = 4
    ):
        super(WatchSleepNet, self).__init__(dataset)

        self.feature_extractor = FeatureExtractor(in_channels=num_features)
        feature_out_channels = 256

        self.tcn = TCNBlock(feature_out_channels,
                            seq_sample_size, tcn_kernel_size, dilation=1)
        tcn_out_channels = seq_sample_size

        self.lstm = nn.LSTM(
            input_size=tcn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_size = 2 * lstm_hidden_size  # bidirectional

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        self.classifier = nn.Linear(lstm_out_size, num_classes)

    def get_metrics(self, output: torch.FloatTensor, label: torch.LongTensor) -> dict:
        """Computes loss and prediction tensors from the classifier output.

        Args:
            output: Output tensor of shape (batch, seq_len, num_classes).
            label: Ground truth class indices of shape (batch,), one per sequence.

        Returns:
            A dictionary with the following keys:
                loss: Cross-entropy loss.
                y_prob: Softmax probabilities of shape (batch, num_classes).
                y_true: Ground truth labels of shape (batch,).
        """
        criterion = nn.CrossEntropyLoss()

        last_output = output[:, -1, :]
        loss = criterion(last_output, label)

        y_prob = torch.softmax(last_output, dim=-1)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": label
        }

    def forward(self, signal: torch.FloatTensor, label: torch.LongTensor, **_) -> dict:
        """Forward propagation.

        Args:
            signal: Input IBI tensor of shape (batch, seq_len, seq_sample_size).
            label: Ground truth class indices of shape (batch,), one per sequence.
            **kwargs: Additional keys from the batch (e.g., patient_id, record_id) that
                are ignored.

        Returns:
            A dictionary with the following keys:
                loss: Cross-entropy loss.
                y_prob: Softmax probabilities of shape (batch, num_classes).
                y_true: Ground truth labels of shape (batch,).
        """
        signal = signal.to(self.device)
        label = label.to(self.device)

        batch_size, seq_len, seq_sample_size = signal.shape

        x = signal.view(batch_size * seq_len, 1, seq_sample_size)
        x = self.feature_extractor(x)

        x = x.view(batch_size, seq_len, -1).permute(0, 2, 1)
        x = self.tcn(x)

        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        attn_out, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_out)

        output = self.classifier(x)

        return self.get_metrics(output, label)
