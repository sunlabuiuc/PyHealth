import torch
import torch.nn as nn
from torchvision import models
from pyhealth.models import BaseModel


class ChestXRayVGG16(BaseModel):
    """
    VGG-16 model adapted for chest X-ray binary classification.

    Author: Karan Thapar, Jonathan Bui
    NetID: kthapar2, jtbui3
    Course: CS598 Deep Learning for Healthcare, Spring 2025
    Title: ChestXRayVGG16 Model for MIMIC-CXR

    This class loads a VGG-16 architecture pretrained on ImageNet and adapts it
    for binary classification tasks (e.g., pneumonia detection) on chest X-ray data.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained)
        self.backbone.classifier[-1] = nn.Linear(self.backbone.classifier[-1].in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.backbone(x)
        return self.sigmoid(logits)

    def get_config(self):
        return {
            "model": "ChestXRayVGG16",
            "pretrained": True,
            "num_classes": 1
        }
