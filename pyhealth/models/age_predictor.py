
"""
Age prediction model for chest X-rays.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class ChestXrayAgePredictor(BaseModel):
    """
    Age prediction model for chest X-rays using DenseNet121.
    
    This model predicts patient age from chest X-ray images using a pretrained
    DenseNet121 backbone modified to accept 1-channel grayscale images, with 
    multi-task learning (age regression + age group classification).
    
    Args:
        dataset: PyHealth dataset object (optional)
        feature_keys: List of input feature keys (default: ["image"])
        label_key: Label key for age (default: "age")
        mode: Task mode (default: "multiclass")
        freeze_backbone: Whether to freeze early backbone layers (default: True)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        
    Examples:
        >>> from pyhealth.datasets import ChestXray14Dataset
        >>> from pyhealth.models import ChestXrayAgePredictor
        >>> from pyhealth.tasks import AgePredictionTask
        >>> 
        >>> dataset = ChestXray14Dataset(root="/data/chestxray14")
        >>> task = AgePredictionTask()
        >>> dataset = dataset.set_task(task)
        >>> 
        >>> model = ChestXrayAgePredictor(dataset=dataset)
    """
    
    def __init__(
        self,
        dataset=None,
        feature_keys: Optional[List[str]] = None,
        label_key: str = "age",
        mode: str = "multiclass",
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        # Set default feature keys
        if feature_keys is None:
            feature_keys = ["image"]
        
        # Store these as instance variables
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.mode = mode
        
        # Only pass dataset to parent __init__
        super().__init__(dataset=dataset)
        
        self.dropout_rate = dropout_rate
        
        # Load pretrained DenseNet121 backbone
        self.backbone = torch.hub.load(
            'pytorch/vision',
            'densenet121',
            weights='DenseNet121_Weights.IMAGENET1K_V1'
        )
        
        # MODIFY: Replace first conv layer to accept 1-channel input
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            1,  # Changed from 3 to 1 input channel for grayscale
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # Initialize the new conv layer with averaged weights from pretrained model
        with torch.no_grad():
            # Average the 3-channel weights to 1-channel to preserve pretrained knowledge
            self.backbone.features.conv0.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        # Get feature dimension and remove classifier
        visual_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Optionally freeze early layers
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'denseblock4' not in name and 'transition3' not in name:
                    param.requires_grad = False
        
        # Age regression head
        self.age_regressor = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 1)
        )
        
        # Age group classifier (auxiliary task)
        self.age_group_classifier = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 4)
        )
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
    
        Args:
            **kwargs: Should contain 'image' (input images) and optionally 'age' (labels)
    
        Returns:
            Dictionary containing predictions and loss
        """
        # Get image input (support both 'image' and 'x' keys)
        x = kwargs.get('image', kwargs.get('x'))
    
        # Get label if available
        y = kwargs.get('age', kwargs.get('y'))
    
        # No conversion needed - model now natively accepts 1-channel grayscale!
        # Extract visual features
        features = self.backbone(x)
    
        # Predict age (regression)
        predicted_age = self.age_regressor(features).squeeze(-1)
    
        # Predict age group (classification)
        age_group_logits = self.age_group_classifier(features)
    
        # Prepare outputs
        results = {
            "y_prob": predicted_age,
            "loss": torch.tensor(0.0, device=x.device),
        }
    
        # Calculate loss if labels provided
        if y is not None:
            # Regression loss
            age_loss = nn.SmoothL1Loss()(predicted_age, y.float())
    
            # Classification loss
            age_groups = self._get_age_group_labels(y)
            group_loss = nn.CrossEntropyLoss()(age_group_logits, age_groups)
    
            # Combined loss
            total_loss = age_loss + 0.3 * group_loss
    
            results["loss"] = total_loss
            results["y_true"] = y
    
        return results
        
    def _get_age_group_labels(self, ages: torch.Tensor) -> torch.Tensor:
        """Convert ages to age group labels."""
        age_groups = torch.zeros_like(ages, dtype=torch.long)
        age_groups[ages < 18] = 0
        age_groups[(ages >= 18) & (ages < 40)] = 1
        age_groups[(ages >= 40) & (ages < 65)] = 2
        age_groups[ages >= 65] = 3
        return age_groups

