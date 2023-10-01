from typing import List, Dict

import torch
import torch.nn as nn
import torchvision
from diffusers import StableDiffusionPipeline

from pyhealth.datasets.sample_dataset_v2 import SampleDataset
from pyhealth.models import BaseModel


SUPPORTED_MODELS = [
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
    "IrohXu/stable-diffusion-mimic-cxr-v0.1"
]


class DiffusionModel(BaseModel):
    """Models from PyTorch's huggingface package.

    This class is a wrapper for diffusion models from huggingface.

    ------------------------------Stable Diffusion-------------------------------------
    Paper: Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. 
    High-resolution image synthesis with latent diffusion models. CVPR 2022
    -----------------------------------------------------------------------------------

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features, e.g., ["image"].
            Only one feature is supported.
        label_key: key in samples to use as label, e.g., "drugs".
        mode: one of "binary", "multiclass", or "multilabel".
        model_name: str, name of the model to use, e.g., "IrohXu/stable-diffusion-mimic-cxr-v0.1".
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
        super(DiffusionModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode
        )

        self.model_name = model_name
        self.model_config = model_config

        self.model = StableDiffusionPipeline.from_pretrained(model_name, **model_config)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]]
        out = self.model(x)
        return out


if __name__ == "__main__":
    from pyhealth.datasets.utils import get_dataloader
    from pyhealth.datasets import MIMICCXRDataset

    base_dataset = MIMICCXRDataset(
        root="/home/xucao2/xucao/PIEMedApp/checkpoints/mimic_cxr",
    )
    sample_dataset = base_dataset.set_task()
    
    def encode(sample):
        return sample

    sample_dataset.set_transform(encode)

    train_loader = get_dataloader(sample_dataset, batch_size=1, shuffle=True)
    
    device = "cuda"
    model = DiffusionModel(
        dataset=sample_dataset,
        feature_keys=["text"],
        label_key="text",
        mode="multiclass",
        model_name="IrohXu/stable-diffusion-mimic-cxr-v0.1",
        model_config={"torch_dtype": torch.float16, "safety_checker": None},
    )
    model.model = model.model.to(device)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    image = model(**data_batch).images[0]  
    image.save("result.png")

