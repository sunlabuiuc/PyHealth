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

    Args:
        dataset (SampleDataset): The PyHealth dataset object to train the model.
        model_size (str, optional): The size of the DINOv2 backbone to pull from 
            TorchHub ("vits14", "vitb14", "vitl14", "vitg14"). Defaults to "vits14".
        **kwargs: Additional keyword arguments passed to BaseModel.
    
    Author:
        Mumme, Raymond Paul rmumme2@illinois.edu

    Examples:
        >>> from pyhealth.models import DINOv2
        >>> model = DINOv2(dataset=task_dataset, feature_keys=["image"], label_key="melanoma", mode="binary")
    """

    def __init__(self, dataset, model_size="vits14", **kwargs):
        # Extract keys BEFORE calling super to avoid TypeError
        self.feature_keys = kwargs.pop("feature_keys", ["image"])
        self.label_key = kwargs.pop("label_key", "melanoma")
        self.mode = kwargs.pop("mode", "binary")
        
        # Now super().__init__ only gets what it expects
        super().__init__(dataset=dataset, **kwargs)

        # Dimension Mapping for all sizes
        dim_map = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
        if model_size not in dim_map:
            raise ValueError(f"Invalid model_size: {model_size}. Choose from: {list(dim_map.keys())}")
        embed_dim = dim_map[model_size]

        # Guards for feature_keys and label_keys
        if not self.feature_keys or len(self.feature_keys) == 0:
            raise ValueError("DINOv2 requires at least one feature key (e.g., 'image').")
        self.feature_key = self.feature_keys[0]
        
        self.label_key = self.label_key if self.label_key else None

        # TorchHub Loading with Offline Fallback Safety
        repo = 'facebookresearch/dinov2'
        hub_model_name = f'dinov2_{model_size}'
        
        try:
            # Try to load from Hub (checks online for updates, then uses cache)
            self.backbone = torch.hub.load(repo, hub_model_name, trust_repo=True)
        except Exception as e:
            print(f"[!] Warning: Failed to load DINOv2 from TorchHub. Attempting offline cache load. Error: {e}")
            try:
                # Force PyTorch to look only in the local cache
                import os
                hub_dir = torch.hub.get_dir()
                repo_dir = os.path.join(hub_dir, repo.replace('/', '_'))
                if os.path.exists(repo_dir):
                     # Add repo to path temporarily to allow local import
                     import sys
                     sys.path.insert(0, repo_dir)
                     from hubconf import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
                     loaders = {"vits14": dinov2_vits14, "vitb14": dinov2_vitb14, "vitl14": dinov2_vitl14, "vitg14": dinov2_vitg14}
                     self.backbone = loaders[model_size]()
                     sys.path.pop(0)
                else:
                    raise FileNotFoundError(f"Local TorchHub cache not found for {repo}.")
            except Exception as inner_e:
                raise RuntimeError(f"Could not load DINOv2 from Hub or local cache: {inner_e}")

        # Freeze the transformer backbone (Linear Probing setup)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Output Classifier
        output_size = self.get_output_size()
        self.fc = nn.Linear(embed_dim, output_size)


    def forward(self, embed=False, **kwargs):
        """Forward propagation for the DINOv2 model.

        Args:
            embed (bool): If True, returns the latent embeddings before the linear classifier.
            **kwargs: Keyword arguments containing the batch data. Must include 
                the keys specified in `feature_keys` and optionally `label_key`.

        Returns:
            dict: A dictionary containing:
                - "logit" (torch.Tensor): The raw model outputs.
                - "y_prob" (torch.Tensor): The predicted probabilities (sigmoid/softmax).
                - "loss" (torch.Tensor, optional): The computed loss, if `label_key` is present.
                - "y_true" (torch.Tensor, optional): The ground truth labels.
                - "embed" (torch.Tensor, optional): The latent features, if `embed=True`.
        """
        x = kwargs[self.feature_key]
        
        # Handle batching of image tensors
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
            
        x = x.to(self.device)

        # Interpretability support: Extract embeddings
        features = self.backbone(x)
        
        if embed:
            return {"embed": features}

        # Classification Head
        logits = self.fc(features)

        # Use BaseModel's native helpers for evaluation consistency
        y_prob = self.prepare_y_prob(logits)
        res = {"logit": logits, "y_prob": y_prob}

        if self.label_key and self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            loss_fn = self.get_loss_function()
            
            # BCEWithLogitsLoss requires Float tensors for binary mode and matching shapes
            if self.mode == "binary":
                y_true_float = y_true.float().view_as(logits)
                loss = loss_fn(logits, y_true_float)
            else:
                loss = loss_fn(logits, y_true)
                
            res["loss"] = loss
            res["y_true"] = y_true

        return res

    def forward_from_embedding(self, embed):
        """Passes pre-computed embeddings through the final classification layer.
        
        Useful for interpretability tasks (like TCAV or TSNE) where latent 
        features need to be quickly classified without re-running the backbone.
        """
        logits = self.fc(embed)
        y_prob = self.prepare_y_prob(logits)
        return {"logit": logits, "y_prob": y_prob}