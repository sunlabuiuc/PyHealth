import torch
import torch.nn as nn
from typing import List


class CXRMultimodalModel(nn.Module):
    """
    Multimodal model for chest X-ray classification using image + text.

    Args:
        text_encoder (nn.Module): Module to encode text tokens.
        image_encoder (nn.Module): Module to encode image tokens.
        hidden_dim (int): Dimension of hidden combined features.
        num_classes (int): Number of output classes.
    """

    def __init__(self, text_encoder: nn.Module, image_encoder: nn.Module, hidden_dim: int, num_classes: int):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, text: torch.Tensor, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for multimodal fusion and classification.

        Args:
            text (torch.Tensor): Tensor of shape (B, T) for tokenized text.
            images (List[torch.Tensor]): List of image tensors [B, T_i] for each image modality.

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        text_features = self.text_encoder(text)
        image_features = [self.image_encoder(img) for img in images]
        image_mean = torch.stack(image_features, dim=0).mean(dim=0)
        combined = text_features + image_mean
        return self.classifier(combined)
