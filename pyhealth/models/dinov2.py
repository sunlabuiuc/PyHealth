# Contributor: [Your Name]
# NetID: [Your NetID]

import torch
import torch.nn as nn
from pyhealth.models.base_model import BaseModel

class DINOv2(BaseModel):
    """Self-supervised DINOv2 vision foundation model adapted for PyHealth.

    This model follows the PyHealth 2.0 Dataset-Aware pattern, automatically 
    configuring its input and output layers based on the provided dataset's schema.
    
    DINOv2 is a state-of-the-art vision transformer trained via self-supervised 
    learning to extract robust visual features without relying on image-text pairs 
    or labeled datasets. It has been shown to exhibit strong out-of-distribution 
    generalization, making it highly suitable for medical imaging tasks where 
    artifacts often confuse supervised models.

    Paper Reference:
        - Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
        - "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations" (CHIL 2025)

    Args:
        dataset (SampleDataset): The PyHealth dataset object to train the model.
        model_size (str, optional): The size of the DINOv2 backbone to pull from 
            TorchHub ("vits14", "vitb14", "vitl14", "vitg14"). Defaults to "vits14".
        **kwargs: Additional keyword arguments passed to BaseModel.

    Examples:
        >>> from pyhealth.models import DINOv2
        >>> model = DINOv2(
        ...     dataset=task_dataset, 
        ...     feature_keys=["image"], 
        ...     label_key="melanoma",
        ...     model_size="vits14"
        ... )
    """
    def __init__(self, dataset, model_size="vits14", **kwargs):
        # 1. Base initialization (automatically sets self.feature_keys and self.label_keys)
        super(DINOv2, self).__init__(dataset=dataset)
        
        self.model_size = model_size
        
        # 2. Load the backbone from Meta/Facebook's TorchHub
        self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}')
        embed_dim = 384 if model_size == "vits14" else 768
        
        # 3. Configure classification head based on the dataset's output schema
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        
        output_size = self.get_output_size()
        
        # Linear probing / fine-tuning classification head
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, **kwargs):
        """Forward pass for the DINOv2 model.

        Args:
            **kwargs: Keyword arguments containing the batch data. Must include 
                the keys specified in `feature_keys` and optionally `label_key`.

        Returns:
            dict: A dictionary containing:
                - "logit" (torch.Tensor): The raw model outputs.
                - "y_prob" (torch.Tensor): The predicted probabilities (sigmoid/softmax).
                - "loss" (torch.Tensor, optional): The computed loss, if `label_key` is present.
                - "y_true" (torch.Tensor, optional): The ground truth labels.
        """
        x = kwargs[self.feature_key]
        
        # Handle batching of image tensors
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
            
        x = x.to(self.device)

        # Backbone and Head
        features = self.backbone(x)
        logits = self.fc(features)

        # Use BaseModel's native helpers for evaluation consistency
        y_prob = self.prepare_y_prob(logits)
        res = {"logit": logits, "y_prob": y_prob}

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            loss_fn = self.get_loss_function()
            
            # BCEWithLogitsLoss requires Float tensors for binary mode
            y_true_float = y_true.float().unsqueeze(1) if y_true.dim() == 1 else y_true
            res["loss"] = loss_fn(logits, y_true_float)
            res["y_true"] = y_true

        return res