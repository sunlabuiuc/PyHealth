from typing import Dict

import torch
import torch.nn.functional as F

from pyhealth.models import Transformer
from pyhealth.models.base_model import BaseModel

from .base_interpreter import BaseInterpreter

# Import TorchvisionModel conditionally to avoid circular imports
try:
    from pyhealth.models import TorchvisionModel
    HAS_TORCHVISION_MODEL = True
except ImportError:
    HAS_TORCHVISION_MODEL = False
    TorchvisionModel = None


def apply_self_attention_rules(R_ss, cam_ss):
    """Apply Chefer's self-attention rules for relevance propagation.

    Args:
        R_ss: Relevance matrix [batch, seq_len, seq_len].
        cam_ss: Attention weight matrix [batch, seq_len, seq_len].

    Returns:
        Updated relevance matrix after propagating through attention layer.
    """
    return torch.matmul(cam_ss, R_ss)


def avg_heads(cam, grad):
    """Average attention scores weighted by gradients across heads.

    Args:
        cam: Attention weights [batch, heads, seq_len, seq_len] or [batch, seq_len, seq_len].
        grad: Gradients w.r.t. attention weights. Same shape as cam.

    Returns:
        Gradient-weighted attention [batch, seq_len, seq_len].
    """
    if len(cam.size()) < 4 and len(grad.size()) < 4:
        return (grad * cam).clamp(min=0)
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=1)
    return cam.clone()


class CheferRelevance(BaseInterpreter):
    """Chefer's gradient-weighted attention method for transformer interpretability.

    This class implements the relevance propagation method from Chefer et al. for
    explaining transformer model predictions. It computes relevance scores for each
    input token (for text/EHR transformers) or patch (for Vision Transformers) by
    combining attention weights with their gradients.

    The method works by:
    1. Performing a forward pass to capture attention maps from each layer
    2. Computing gradients of the target class w.r.t. attention weights
    3. Combining attention and gradients using element-wise multiplication
    4. Propagating relevance through layers using attention rollout rules

    This approach provides more faithful explanations than raw attention weights
    alone, as it accounts for how attention contributes to the final prediction.

    Paper:
        Chefer, Hila, Shir Gur, and Lior Wolf.
        "Generic Attention-model Explainability for Interpreting Bi-Modal and
        Encoder-Decoder Transformers."
        Proceedings of the IEEE/CVF International Conference on Computer Vision
        (ICCV), 2021.

    Supported Models:
        - PyHealth Transformer: For sequential/EHR data with multiple feature keys
        - TorchvisionModel (ViT variants): vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

    Args:
        model (BaseModel): A trained PyHealth model to interpret. Must be either:
            - A ``Transformer`` model for sequential/EHR data
            - A ``TorchvisionModel`` with a ViT architecture for image data

    Example:
        >>> # Example 1: PyHealth Transformer for EHR data
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods import CheferRelevance
        >>>
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A05B", "A05C", "A06A"],
        ...         "procedures": ["P01", "P02"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v1",
        ...         "conditions": ["A05B"],
        ...         "procedures": ["P01"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ehr_example",
        ... )
        >>> model = Transformer(dataset=dataset)
        >>> # ... train the model ...
        >>>
        >>> # Create interpreter and compute attribution
        >>> interpreter = CheferRelevance(model)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>>
        >>> # Default: attribute to predicted class
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns dict: {"conditions": tensor, "procedures": tensor}
        >>> print(attributions["conditions"].shape)  # [batch, num_tokens]
        >>>
        >>> # Optional: attribute to a specific class (e.g., class 1)
        >>> attributions = interpreter.attribute(class_index=1, **batch)
        >>>
        >>> # Example 2: TorchvisionModel ViT for image data
        >>> from pyhealth.datasets import COVID19CXRDataset
        >>> from pyhealth.models import TorchvisionModel
        >>> from pyhealth.interpret.utils import visualize_image_attr
        >>>
        >>> base_dataset = COVID19CXRDataset(root="/path/to/data")
        >>> sample_dataset = base_dataset.set_task()
        >>> model = TorchvisionModel(
        ...     dataset=sample_dataset,
        ...     model_name="vit_b_16",
        ...     model_config={"weights": "DEFAULT"},
        ... )
        >>> # ... train the model ...
        >>>
        >>> # Create interpreter and compute attribution
        >>> # Task schema: input_schema={"image": "image"}, so feature_key="image"
        >>> interpreter = CheferRelevance(model)
        >>>
        >>> # Default: attribute to predicted class
        >>> result = interpreter.attribute(**batch)
        >>> # Returns dict keyed by feature_key: {"image": tensor}
        >>> attr_map = result["image"]  # Shape: [batch, 1, H, W]
        >>>
        >>> # Optional: attribute to a specific class (e.g., predicted class)
        >>> pred_class = model(**batch)["y_prob"].argmax().item()
        >>> result = interpreter.attribute(class_index=pred_class, **batch)
        >>>
        >>> # Visualize
        >>> img, attr, overlay = visualize_image_attr(
        ...     image=batch["image"][0],
        ...     attribution=result["image"][0, 0],
        ... )
    """

    def __init__(self, model: BaseModel):
        super().__init__(model)
        
        # Determine model type
        self._is_transformer = isinstance(model, Transformer)
        self._is_vit = False
        
        if HAS_TORCHVISION_MODEL and TorchvisionModel is not None:
            if isinstance(model, TorchvisionModel):
                self._is_vit = model.is_vit_model()
        
        if not self._is_transformer and not self._is_vit:
            raise ValueError(
                f"CheferRelevance requires a Transformer or TorchvisionModel (ViT), "
                f"got {type(model).__name__}. For TorchvisionModel, only ViT variants "
                f"(vit_b_16, vit_b_32, etc.) are supported."
            )

    def attribute(
        self,
        interpolate: bool = True,
        class_index: int = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute relevance scores for each token/patch.

        This is the primary method for computing attributions. Returns a
        dictionary keyed by the model's feature keys (from the task schema).

        Args:
            interpolate: For ViT models, if True interpolate attribution to image size.
            class_index: Target class index to compute attribution for. If None
                (default), uses the model's predicted class. This is useful when
                you want to explain why a specific class was predicted or to
                compare attributions across different classes.
            **data: Input data from dataloader batch containing:
                - For Transformer: feature keys (conditions, procedures, etc.) + label
                - For ViT: image feature key (e.g., "image") + label

        Returns:
            Dict[str, torch.Tensor]: Dictionary keyed by feature keys from the task schema.

            - For Transformer: ``{"conditions": tensor, "procedures": tensor, ...}``
              where each tensor has shape ``[batch, num_tokens]``.
            - For ViT: ``{"image": tensor}`` (or whatever the task's image key is)
              where tensor has shape ``[batch, 1, H, W]``.
        """
        if self._is_vit:
            return self._attribute_vit(
                interpolate=interpolate,
                class_index=class_index,
                **data
            )
        return self._attribute_transformer(class_index=class_index, **data)

    def _attribute_transformer(
        self,
        class_index: int = None,
        **data
    ) -> Dict[str, torch.Tensor]:
        """Compute relevance for PyHealth Transformer models.
        
        Args:
            class_index: Target class for attribution. If None, uses predicted class.
            **data: Input data from dataloader batch.
        """
        data["register_hook"] = True

        logits = self.model(**data)["logit"]
        if class_index is None:
            class_index = torch.argmax(logits, dim=-1)

        one_hot = F.one_hot(torch.tensor(class_index), logits.size()[1]).float()
        one_hot = one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.to(logits.device) * logits)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        feature_keys = self.model.feature_keys
        num_tokens = {}
        for key in feature_keys:
            feature_transformer = self.model.transformer[key].transformer
            for block in feature_transformer:
                num_tokens[key] = block.attention.get_attn_map().shape[-1]

        attn = {}
        for key in feature_keys:
            R = (
                torch.eye(num_tokens[key])
                .unsqueeze(0)
                .repeat(len(data[key]), 1, 1)
                .to(logits.device)
            )
            for blk in self.model.transformer[key].transformer:
                grad = blk.attention.get_attn_grad()
                cam = blk.attention.get_attn_map()
                cam = avg_heads(cam, grad)
                R += apply_self_attention_rules(R, cam).detach()
            attn[key] = R[:, 0]

        return attn

    def _attribute_vit(
        self,
        interpolate: bool = True,
        class_index: int = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute ViT attribution and return spatial attribution map.
        
        Args:
            interpolate: If True, interpolate to full image size.
            class_index: Target class for attribution. If None, uses predicted class.
            **data: Must contain the image feature key.
            
        Returns:
            Dict keyed by the model's feature_key (e.g., "image") with spatial
            attribution map of shape [batch, 1, H, W].
        """
        # Get the feature key (first element of feature_keys list)
        feature_key = self.model.feature_keys[0]
        x = data.get(feature_key)
        if x is None:
            raise ValueError(
                f"Expected feature key '{feature_key}' in data. "
                f"Available keys: {list(data.keys())}"
            )
        
        x = x.to(self.model.device)
        
        # Infer input size from image dimensions (assumes square images)
        input_size = x.shape[-1]
        
        # Forward pass with attention capture
        self.model.zero_grad()
        logits, attention_maps = self.model.forward_with_attention(x, register_hook=True)
        
        # Use predicted class if not specified
        target_class = class_index
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        if isinstance(target_class, int):
            one_hot[:, target_class] = 1
        else:
            if target_class.dim() == 0:
                target_class = target_class.unsqueeze(0)
            one_hot.scatter_(1, target_class.unsqueeze(1), 1)
        
        one_hot = one_hot.requires_grad_(True)
        (logits * one_hot).sum().backward(retain_graph=True)
        
        # Compute gradient-weighted attention
        attention_gradients = self.model.get_attention_gradients()
        batch_size = attention_maps[0].shape[0]
        num_tokens = attention_maps[0].shape[-1]
        device = attention_maps[0].device
        
        R = torch.eye(num_tokens, device=device)
        R = R.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        for attn, grad in zip(attention_maps, attention_gradients):
            cam = avg_heads(attn, grad)
            R = R + apply_self_attention_rules(R.detach(), cam.detach())
        
        # CLS token's relevance to patches (excluding CLS itself)
        patches_attr = R[:, 0, 1:]
        
        # Reshape to spatial layout
        h_patches, w_patches = self.model.get_num_patches(input_size)
        attr_map = patches_attr.reshape(batch_size, 1, h_patches, w_patches)
        
        if interpolate:
            attr_map = F.interpolate(
                attr_map,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
        
        # Return keyed by the model's feature key (e.g., "image")
        return {feature_key: attr_map}

    # Backwards compatibility aliases
    def get_relevance_matrix(self, **data):
        """Alias for _attribute_transformer. Use attribute() instead."""
        return self._attribute_transformer(**data)

    def get_vit_attribution_map(
        self,
        interpolate: bool = True,
        class_index: int = None,
        **data
    ):
        """Alias for attribute() for ViT. Use attribute() instead.
        
        Returns the attribution tensor directly (not wrapped in a dict).
        """
        result = self._attribute_vit(
            interpolate=interpolate,
            class_index=class_index,
            **data
        )
        # Return the attribution tensor directly (get the first/only value)
        feature_key = self.model.feature_keys[0]
        return result[feature_key]
