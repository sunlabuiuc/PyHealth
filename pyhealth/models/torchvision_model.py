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

        assert len(self.feature_keys) == 1, (
            "Only one feature key is supported if TorchvisionModel is initialized"
        )
        self.feature_key = self.feature_keys[0]
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if TorchvisionModel is initialized"
        )
        self.label_key = self.label_keys[0]
        assert model_name in SUPPORTED_MODELS_FINAL_LAYER.keys(), (
            f"PyHealth does not currently include {model_name} model!"
        )
        self.mode = self.dataset.output_schema[self.label_key]

        self.model = torchvision.models.get_model(model_name, **model_config)
        final_layer_name = SUPPORTED_MODELS_FINAL_LAYER[model_name]
        final_layer = self.model
        for name in final_layer_name.split("."):
            final_layer = getattr(final_layer, name)
        hidden_dim = final_layer.in_features
        output_size = self.get_output_size()
        layer_name = final_layer_name.split(".")[0]
        setattr(self.model, layer_name, nn.Linear(hidden_dim, output_size))

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        x = kwargs[self.feature_key]
        x = x.to(self.device)
        if x.shape[1] == 1:
            # Models from torchvision expect 3 channels
            x = x.repeat((1, 3, 1, 1))
        logits = self.model(x)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }
