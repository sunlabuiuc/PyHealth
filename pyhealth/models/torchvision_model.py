from typing import List, Dict

import torch
import torch.nn as nn
import torchvision

from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.models import BaseModel

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

    -----------------------------------ResNet------------------------------------------
    Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning
    for Image Recognition. CVPR 2016.
    -----------------------------------DenseNet----------------------------------------
    Paper: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks. CVPR 2017.
    ----------------------------Vision Transformer (ViT)-------------------------------
    Paper: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
    Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
    Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. An Image is Worth
    16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
    ----------------------------Swin Transformer (and V2)------------------------------
    Paper: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
    Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted
    Windows. ICCV 2021.

    Paper: Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning,
    Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo. Swin Transformer V2: Scaling
    Up Capacity and Resolution. CVPR 2022.
    -----------------------------------------------------------------------------------

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features, e.g., ["image"].
            Only one feature is supported.
        label_key: key in samples to use as label, e.g., "drugs".
        mode: one of "binary", "multiclass", or "multilabel".
        model_name: str, name of the model to use, e.g., "resnet18".
            See SUPPORTED_MODELS in the source code for the full list.
        model_config: dict, kwargs to pass to the model constructor,
            e.g., {"weights": "DEFAULT"}. See the torchvision documentation for the
            set of supported kwargs for each model.
    -----------------------------------------------------------------------------------
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        model_name: str,
        model_config: dict,
    ):
        super(TorchvisionModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )

        self.model_name = model_name
        self.model_config = model_config

        assert len(feature_keys) == 1, "Only one feature is supported!"
        assert model_name in SUPPORTED_MODELS_FINAL_LAYER.keys(), \
            f"PyHealth does not currently include {model_name} model!"

        self.model = torchvision.models.get_model(model_name, **model_config)
        final_layer_name = SUPPORTED_MODELS_FINAL_LAYER[model_name]
        final_layer = self.model
        for name in final_layer_name.split("."):
            final_layer = getattr(final_layer, name)
        hidden_dim = final_layer.in_features
        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        setattr(self.model, final_layer_name.split(".")[0], nn.Linear(hidden_dim, output_size))

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]]
        x = torch.stack(x, dim=0).to(self.device)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        logits = self.model(x)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    from pyhealth.datasets.utils import get_dataloader
    from torchvision import transforms
    from pyhealth.datasets import COVID19CXRDataset

    base_dataset = COVID19CXRDataset(
        root="/srv/local/data/zw12/raw_data/covid19-radiography-database/COVID-19_Radiography_Dataset",
    )

    sample_dataset = base_dataset.set_task()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
    ])


    def encode(sample):
        sample["path"] = transform(sample["path"])
        return sample


    sample_dataset.set_transform(encode)

    train_loader = get_dataloader(sample_dataset, batch_size=16, shuffle=True)

    model = TorchvisionModel(
        dataset=sample_dataset,
        feature_keys=["path"],
        label_key="label",
        mode="multiclass",
        model_name="resnet18",
        model_config={"weights": "DEFAULT"},
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()