from typing import List, Dict

import torch
import torch.nn as nn
from torchvision import models

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class ResNet(BaseModel):
    """ResNet model for image data.

    Paper: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition. CVPR 2016.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        pretrained: whether to use pretrained weights. Default is False.
        num_layers: number of resnet layers. Supported values are 18, 34, 50, 101, 152.
            Default is 18.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        pretrained=False,
        num_layers=18,
        **kwargs,
    ):
        super(ResNet, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        cnn_name = f"resnet{num_layers}"
        self.cnn = models.__dict__[cnn_name](pretrained=pretrained)
        hidden_dim = self.cnn.fc.in_features
        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        self.cnn.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]]
        x = torch.stack(x, dim=0).to(self.device)
        if x.shape[1] == 1:
            x = x.repeat((1, 3, 1, 1))
        logits = self.cnn(x)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    from pyhealth.datasets import COVID19CXRDataset, get_dataloader
    from torchvision import transforms

    base_dataset = COVID19CXRDataset(
        root="/srv/local/data/zw12/raw_data/covid19-radiography-database/COVID-19_Radiography_Dataset",
    )

    sample_dataset = base_dataset.set_task()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])
    ])
    def encode(sample):
        sample["path"] = transform(sample["path"])
        return sample

    sample_dataset.set_transform(encode)

    train_loader = get_dataloader(sample_dataset, batch_size=16, shuffle=True)

    model = ResNet(
        dataset=sample_dataset,
        feature_keys=[
            "path",
        ],
        label_key="label",
        mode="multiclass",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()