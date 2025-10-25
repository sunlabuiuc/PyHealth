from typing import Any, Dict

import torch
import torch.nn as nn
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
    """Models from PyTorch's torchvision package.

    This class is a wrapper for models from torchvision. It will automatically load
    the corresponding model and weights from torchvision. The final layer will be
    replaced with a linear layer with the correct output size.

    Supported Models:
    ----------------
    ResNet:
        Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
        Deep Residual Learning for Image Recognition. CVPR 2016.

    DenseNet:
        Paper: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
        Densely Connected Convolutional Networks. CVPR 2017.

    Vision Transformer (ViT):
        Paper: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
        An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
        ICLR 2021.

    Swin Transformer:
        Paper: Ze Liu, Yutong Lin, Yue Cao, et al.
        Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows.
        ICCV 2021.

        Paper: Ze Liu, Han Hu, Yutong Lin, et al.
        Swin Transformer V2: Scaling Up Capacity and Resolution. CVPR 2022.

    Args:
        dataset: The dataset to train the model. Used to query information
            such as the set of all tokens.
        model_name: Name of the model to use (e.g., "resnet18").
            See SUPPORTED_MODELS in the source code for the full list.
        model_config: Dictionary of kwargs to pass to the model constructor.
            Example: {"weights": "DEFAULT"}. See torchvision documentation for
            supported kwargs for each model.
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
