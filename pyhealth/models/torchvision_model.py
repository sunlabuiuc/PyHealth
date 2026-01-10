from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..datasets import SampleDataset
from .base_model import BaseModel

SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
    "swin_t",
    "swin_s",
    "swin_b",
]

SUPPORTED_MODELS_FINAL_LAYER = {}
for model in SUPPORTED_MODELS:
    if "resnet" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "fc"
    elif "densenet" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "classifier"
    elif "vit" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "heads.head"
    elif "swin" in model:
        SUPPORTED_MODELS_FINAL_LAYER[model] = "head"
    else:
        raise NotImplementedError


class TorchvisionModel(BaseModel):
    """Models from PyTorch's torchvision package for image classification.

    This class is a wrapper for pretrained models from torchvision. It will 
    automatically load the corresponding model and weights from torchvision. 
    The final classification layer is replaced with a linear layer matching 
    the dataset's output size, enabling transfer learning on custom datasets.

    The model supports:
    - Standard forward pass for training/inference
    - Embedding extraction for interpretability methods
    - Attention map capture for ViT models (used by CheferRelevance)

    Supported Models:
    -----------------
    ResNet (resnet18, resnet34, resnet50, resnet101, resnet152):
        Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        "Deep Residual Learning for Image Recognition."
        IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

    DenseNet (densenet121, densenet161, densenet169, densenet201):
        Paper: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
        "Densely Connected Convolutional Networks."
        IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

    Vision Transformer (vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14):
        Paper: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
        "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
        International Conference on Learning Representations (ICLR), 2021.

    Swin Transformer (swin_t, swin_s, swin_b):
        Paper: Ze Liu, Yutong Lin, Yue Cao, et al.
        "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows."
        IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

        Paper: Ze Liu, Han Hu, Yutong Lin, et al.
        "Swin Transformer V2: Scaling Up Capacity and Resolution."
        IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

    Args:
        dataset (SampleDataset): The dataset to train the model. Used to query
            information such as the number of output classes. Must have exactly
            one feature key (the image) and one label key.
        model_name (str): Name of the model to use. Must be one of:
            resnet18, resnet34, resnet50, resnet101, resnet152,
            densenet121, densenet161, densenet169, densenet201,
            vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14,
            swin_t, swin_s, swin_b.
        model_config (Dict[str, Any]): Dictionary of kwargs to pass to the model
            constructor. Common options include:
            - ``{"weights": "DEFAULT"}``: Use pretrained ImageNet weights
            - ``{"weights": None}``: Random initialization
            See torchvision documentation for all supported kwargs.

    Example:
        >>> from pyhealth.datasets import COVID19CXRDataset, get_dataloader
        >>> from pyhealth.models import TorchvisionModel
        >>> from pyhealth.trainer import Trainer
        >>>
        >>> # Load a medical imaging dataset
        >>> base_dataset = COVID19CXRDataset(root="/path/to/COVID-19_Radiography_Dataset")
        >>> sample_dataset = base_dataset.set_task()
        >>>
        >>> # Create a ViT model with pretrained weights
        >>> model = TorchvisionModel(
        ...     dataset=sample_dataset,
        ...     model_name="vit_b_16",
        ...     model_config={"weights": "DEFAULT"},
        ... )
        >>>
        >>> # Train the model
        >>> train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
        >>> trainer = Trainer(model=model, device="cuda:0")
        >>> trainer.train(train_dataloader=train_loader, epochs=10)
        >>>
        >>> # Inference
        >>> test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)
        >>> y_true, y_prob, _ = trainer.inference(test_loader)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str,
        model_config: Dict[str, Any],
    ):
        super(TorchvisionModel, self).__init__(
            dataset=dataset,
        )

        self.model_name = model_name
        self.model_config = model_config

        assert (
            len(self.feature_keys) == 1
        ), "Only one feature key is supported if TorchvisionModel is initialized"
        self.feature_key = self.feature_keys[0]
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if TorchvisionModel is initialized"
        self.label_key = self.label_keys[0]
        assert (
            model_name in SUPPORTED_MODELS_FINAL_LAYER.keys()
        ), f"PyHealth does not currently include {model_name} model!"
        self.mode = self.dataset.output_schema[self.label_key]

        self.model = torchvision.models.get_model(model_name, **model_config)
        final_layer_name = SUPPORTED_MODELS_FINAL_LAYER[model_name]
        final_layer = self.model
        for name in final_layer_name.split("."):
            final_layer = getattr(final_layer, name)
        hidden_dim = final_layer.in_features
        self.hidden_dim = hidden_dim  # Store for embedding extraction
        output_size = self.get_output_size()
        layer_name = final_layer_name.split(".")[0]
        setattr(self.model, layer_name, nn.Linear(hidden_dim, output_size))
        
        # Initialize attention hooks storage for ViT interpretability
        self._attention_maps: List[torch.Tensor] = []
        self._attention_gradients: List[torch.Tensor] = []
        self._hooks: List[Any] = []
        
        # Setup attention hooks for ViT models
        if "vit" in model_name:
            self._setup_vit_attention_hooks()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Must contain feature_key and label_key from dataset.
                Can optionally contain 'embed' flag to return embeddings.

        Returns:
            Dictionary with:
                - loss: classification loss (if embed=False)
                - y_prob: predicted probabilities (if embed=False)
                - y_true: true labels (if embed=False)
                - embed: embeddings before final layer (if embed=True)
        """
        x = kwargs[self.feature_key]
        x = x.to(self.device)
        if x.shape[1] == 1:
            # Models from torchvision expect 3 channels
            x = x.repeat((1, 3, 1, 1))

        # Check if we should return embeddings
        embed = kwargs.get("embed", False)

        if embed:
            # Extract embeddings before the final layer
            embeddings = self._extract_embeddings(x)
            return {
                "embed": embeddings,
            }
        else:
            # Standard forward pass
            logits = self.model(x)
            y_true = kwargs[self.label_key].to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            y_prob = self.prepare_y_prob(logits)
            return {
                "loss": loss,
                "y_prob": y_prob,
                "y_true": y_true,
            }

    def _extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings before the final classification layer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Embeddings tensor of shape (batch_size, hidden_dim)
        """
        if "resnet" in self.model_name:
            # For ResNet: forward through all layers except fc
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            embeddings = torch.flatten(x, 1)

        elif "densenet" in self.model_name:
            # For DenseNet: forward through features, then adaptive avgpool
            features = self.model.features(x)
            out = torch.nn.functional.relu(features, inplace=True)
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
            embeddings = torch.flatten(out, 1)

        elif "vit" in self.model_name:
            # For Vision Transformer: use torchvision's public API methods
            # Process input (conv projection, reshape, add position embeddings)
            x = self.model._process_input(x)

            # Expand the CLS token to the full batch
            batch_size = x.shape[0]
            batch_class_token = self.model.class_token.expand(batch_size, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            # Pass through encoder
            x = self.model.encoder(x)

            # Extract CLS token (position 0)
            embeddings = x[:, 0]

        elif "swin" in self.model_name:
            # For Swin Transformer: forward through features, then avgpool
            x = self.model.features(x)
            x = self.model.norm(x)
            x = self.model.permute(x)
            embeddings = self.model.avgpool(x)
            embeddings = torch.flatten(embeddings, 1)

        else:
            raise NotImplementedError(
                f"Embedding extraction not implemented for {self.model_name}"
            )

        return embeddings

    def forward_from_embedding(
        self,
        embeddings: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass from pre-computed embeddings.
        
        This method allows running the classification head on embeddings that
        were computed externally, useful for interpretability methods like
        DeepLift and Integrated Gradients.
        
        Args:
            embeddings: Pre-computed embeddings tensor of shape (batch_size, hidden_dim).
            **kwargs: Must contain label_key for loss computation.
            
        Returns:
            Dictionary with:
                - loss: classification loss
                - y_prob: predicted probabilities
                - y_true: true labels
                - logit: raw logits
        """
        embeddings = embeddings.to(self.device)
        
        # Get the final classification layer
        final_layer_name = SUPPORTED_MODELS_FINAL_LAYER[self.model_name]
        layer_name = final_layer_name.split(".")[0]
        fc_layer = getattr(self.model, layer_name)
        
        # Apply classification head
        logits = fc_layer(embeddings)
        
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }

    # =========================================================================
    # ViT Attention Hooks for Interpretability (used by CheferRelevance)
    # =========================================================================
    
    def _setup_vit_attention_hooks(self) -> None:
        """Setup attention hooks for ViT models to capture attention maps.
        
        This enables Chefer-style interpretability by storing attention maps
        and their gradients during forward and backward passes.
        """
        if "vit" not in self.model_name:
            return
        
        # Access the encoder blocks (different paths for different torchvision versions)
        try:
            encoder = self.model.encoder
            if hasattr(encoder, 'layers'):
                blocks = encoder.layers
            else:
                blocks = list(encoder.children())
        except AttributeError:
            print(f"Warning: Could not setup attention hooks for {self.model_name}")
            return
        
        self._vit_blocks = blocks
    
    def clear_attention_storage(self) -> None:
        """Clear stored attention maps and gradients."""
        self._attention_maps = []
        self._attention_gradients = []
    
    def get_attention_maps(self) -> List[torch.Tensor]:
        """Get stored attention maps from last forward pass.
        
        Returns:
            List of attention tensors, one per encoder block.
        """
        return self._attention_maps
    
    def get_attention_gradients(self) -> List[torch.Tensor]:
        """Get stored attention gradients from last backward pass.
        
        Returns:
            List of attention gradient tensors, one per encoder block.
        """
        return self._attention_gradients

    def is_vit_model(self) -> bool:
        """Check if this is a Vision Transformer model.
        
        Returns:
            True if model is ViT, False otherwise.
        """
        return "vit" in self.model_name
    
    def _compute_manual_attention(
        self,
        mha: nn.MultiheadAttention,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention manually to enable gradient flow through attention weights.
        
        This method replaces the black-box nn.MultiheadAttention call with explicit
        QKV computation, ensuring that attention weights are part of the computational
        graph and gradients can flow through them for interpretability methods.
        
        Args:
            mha: The nn.MultiheadAttention module to extract weights from.
            x: Input tensor of shape [batch, seq_len, embed_dim].
            
        Returns:
            Tuple of (attn_output, attn_weights) where:
            - attn_output: [batch, seq_len, embed_dim] - the attention output
            - attn_weights: [batch, num_heads, seq_len, seq_len] - attention weights
              that ARE in the computation graph (gradients will flow through them)
        """
        batch_size, seq_len, embed_dim = x.shape
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5
        
        # Extract projection weights from MHA module
        # in_proj_weight: [3*embed_dim, embed_dim] contains Q, K, V projections
        W_qkv = mha.in_proj_weight  # [3*embed_dim, embed_dim]
        b_qkv = mha.in_proj_bias    # [3*embed_dim] or None
        
        # Project input to Q, K, V using the MHA's learned weights
        qkv = F.linear(x, W_qkv, b_qkv)  # [batch, seq_len, 3*embed_dim]
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        # [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores: [batch, heads, seq, seq]
        # THIS is the key: attn_weights is computed inline and stays in the graph
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = attn_weights.softmax(dim=-1)
        
        # Apply attention to values: [batch, heads, seq, head_dim]
        attn_output = attn_weights @ v
        
        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Apply output projection
        attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)
        
        return attn_output, attn_weights
    
    def forward_with_attention(
        self,
        x: torch.Tensor,
        register_hook: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with attention map capture for ViT models.
        
        This method provides explicit access to attention maps for interpretability,
        used by CheferRelevance interpreter.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            register_hook: If True, register hooks to capture attention gradients.
            
        Returns:
            Tuple of (logits, attention_maps) where attention_maps is a list of
            attention tensors from each encoder block.
            
        Raises:
            ValueError: If model is not a ViT model.
        """
        if not self.is_vit_model():
            raise ValueError("forward_with_attention only works with ViT models")
        
        self.clear_attention_storage()
        
        # Move input to device (consistent with forward method)
        x = x.to(self.device)
        
        # Handle channel dimension
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        
        # Ensure input requires grad for gradient-based attribution
        if register_hook and not x.requires_grad:
            x = x.requires_grad_(True)
        
        # Process input (conv projection + position embeddings)
        x = self.model._process_input(x)
        
        # Add CLS token
        batch_size = x.shape[0]
        batch_class_token = self.model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Forward through encoder blocks with attention capture
        attention_maps = []
        
        # Access encoder layers
        if hasattr(self.model.encoder, 'layers'):
            encoder_layers = self.model.encoder.layers
        else:
            encoder_layers = list(self.model.encoder.children())
        
        for idx, block in enumerate(encoder_layers):
            # Each block is an EncoderBlock with self_attention
            if hasattr(block, 'self_attention'):
                # Apply layer norm
                ln_x = block.ln_1(x)
                
                # Use manual attention computation for gradient flow
                # This computes Q, K, V inline so attention weights stay in the graph
                attn_output, attn_weights = self._compute_manual_attention(
                    block.self_attention, ln_x
                )
                
                # Store attention weights (now in computation graph!)
                attention_maps.append(attn_weights)
                if register_hook:
                    # Register hook to capture gradients during backprop
                    # Gradients will now flow through attn_weights!
                    attn_weights.register_hook(
                        lambda grad, i=idx: self._attention_gradients.insert(i, grad)
                    )
                
                # Continue with residual connections
                x = x + block.dropout(attn_output)
                x = x + block.mlp(block.ln_2(x))
            else:
                # Fallback: just pass through the block
                x = block(x)
        
        # Apply layer norm
        x = self.model.encoder.ln(x)
        
        # Get CLS token embedding and classify
        cls_embedding = x[:, 0]
        logits = self.model.heads(cls_embedding)
        
        self._attention_maps = attention_maps
        return logits, attention_maps
    
    def get_patch_size(self) -> int:
        """Get the patch size for ViT models.
        
        Returns:
            Patch size (e.g., 16 for vit_b_16).
            
        Raises:
            ValueError: If model is not a ViT model.
        """
        if not self.is_vit_model():
            raise ValueError("get_patch_size only works with ViT models")
        
        # Extract from model name
        parts = self.model_name.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        
        # Default fallback
        return 16
    
    def get_num_patches(self, input_size: int = 224) -> Tuple[int, int]:
        """Get the number of patches for ViT models.
        
        Args:
            input_size: Input image size (default 224).
        
        Returns:
            Tuple of (height_patches, width_patches). For standard 224x224 input
            with patch_size=16, this is (14, 14).
        """
        patch_size = self.get_patch_size()
        return (input_size // patch_size, input_size // patch_size)
