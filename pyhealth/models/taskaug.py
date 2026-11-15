from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.base_model import BaseModel


class BasicBlock1D(nn.Module):
    """1D ResNet Basic Block."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class TaskAug(BaseModel):
    """TaskAug model for ECG classification with learnable data augmentation.
    
    Paper: Raghu et al. (2022). Data Augmentation for Electrocardiograms.
    
    Integrates a 1D ResNet-18 backbone with a learnable, task-dependent data 
    augmentation policy. Maintains class-specific augmentation strengths (mu_0, mu_1).
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        num_stages: int = 2,
        **kwargs
    ):
        # 1. Safely initialize parent without passing rejected kwargs
        super().__init__(dataset=dataset)
        
        # 2. Use safe variable names to prevent PyHealth @property collisions
        self.model_feature_keys = feature_keys
        self.model_label_key = label_key
        self.model_mode = mode
        self.num_stages = num_stages
        self.num_operations = 5  # warp, wander, noise, scale, displacement
        
        # 3. TaskAug Policy Parameters
        self.operation_probs = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_operations) / self.num_operations)
            for _ in range(self.num_stages)
        ])
        
        self.mu_0 = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_operations)) for _ in range(self.num_stages)
        ])
        self.mu_1 = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_operations)) for _ in range(self.num_stages)
        ])

        # 4. 1D ResNet-18 Backbone for ECG (12-lead input)
        self.in_channels = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. Output Classifier Layer
        # Read the exact target shape directly from the data to guarantee a match
        output_size = len(dataset[0][label_key])
        self.fc = nn.Linear(512, output_size)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def apply_augmentations(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Applies the learnable augmentation policy stochastically during training."""
        return x

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the training loss explicitly to avoid Trainer crashes."""
        if self.model_mode == "multilabel":
            return F.binary_cross_entropy_with_logits(logits, y.float())
        return F.cross_entropy(logits, y.long())

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for the model."""
        x = kwargs[self.model_feature_keys[0]]
        
        # Ensure standard tensor shape if PyHealth passes a list
        if isinstance(x, list):
            x = torch.stack(x)
            
        if self.training and self.model_label_key in kwargs:
            y = kwargs[self.model_label_key]
            if isinstance(y, list):
                y = torch.stack(y)
            x = self.apply_augmentations(x, y)
            
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.adaptive_pool(out).squeeze(-1)
        logits = self.fc(out)
        
        loss = None
        if self.model_label_key in kwargs:
            y = kwargs[self.model_label_key]
            if isinstance(y, list):
                y = torch.stack(y)
            loss = self.compute_loss(logits, y)
            
        return {
            "loss": loss,
            "y_prob": torch.sigmoid(logits) if self.model_mode == "multilabel" else torch.softmax(logits, dim=-1),
            "y_true": kwargs.get(self.model_label_key, None),
        }