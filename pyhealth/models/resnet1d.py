import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset

class ResNet1D(BaseModel):
    """
    ResNet1D: A deep residual network for healthcare prediction tasks, specifically designed for time-series data such as ECG.
    Args:
        dataset: Dataset to train the model.
        num_classes: Number of output classes (e.g., binary or multiclass).
        in_channels: The number of input channels for the 1D convolution (e.g., 1 for univariate, more for multivariate signals).
        output_dim: The final dimension of the output before classification.
    """
    
    def __init__(self, dataset: SampleEHRDataset, num_classes: int = 2, in_channels: int = 1, output_dim: int = 256, **kwargs):
        super(ResNet1D, self).__init__(dataset=dataset, **kwargs)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # Define the architecture
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the ResNet1D model consisting of several residual blocks followed by a fully connected layer for classification.
        """
        layers = []
        layers.append(nn.Conv1d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(self._make_resnet_block(64, 64, num_blocks=2))
        layers.append(self._make_resnet_block(64, 128, num_blocks=2, stride=2))
        layers.append(self._make_resnet_block(128, 256, num_blocks=2, stride=2))
        layers.append(self._make_resnet_block(256, self.output_dim, num_blocks=2, stride=2))
        
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.output_dim, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def _make_resnet_block(self, in_channels, out_channels, num_blocks, stride=1):
        """
        Creates a series of residual blocks with the specified input/output channels and number of blocks.
        """
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResNet1DBlock(in_channels, out_channels, stride if i == 0 else 1))
            in_channels = out_channels
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        """
        Forward pass of the ResNet1D model.
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
        Returns:
            y_pred: Predicted output from the model.
        """
        return self.model(x)

class ResNet1DBlock(nn.Module):
    """
    A single residual block consisting of two 1D convolutional layers.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet1DBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

# Wrapper for ResNet1D model for compatibility with PyHealth
class ResNet1DModel(BaseModel):
    def __init__(self, dataset: SampleEHRDataset, num_classes: int = 2, in_channels: int = 1, output_dim: int = 256, **kwargs):
        super(ResNet1DModel, self).__init__(dataset=dataset, **kwargs)
        self.model = ResNet1D(dataset=dataset, num_classes=num_classes, in_channels=in_channels, output_dim=output_dim, **kwargs)
    
    def forward(self, **kwargs):
        """
        Process input data and return model predictions (loss, logits, probabilities, etc.).
        Args:
            **kwargs: Input batch data.
        """
        features, labels = self._prepare_input(**kwargs)
        outputs = self.model(features)
        
        if self.mode == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(outputs, labels.float())
        elif self.mode == "multiclass":
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels.long().squeeze())
        else:
            raise ValueError("Mode must be 'binary' or 'multiclass'")

        return {"loss": loss, "y_pred": outputs, "y_true": labels}
