from typing import List, Dict

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torchvision import models

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class HuggingfaceAutoModel(BaseModel):
    """AutoModel class for Huggingface models.
    """

    def __init__(
        self,
        model_name: str,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        pretrained=False,
        num_layers=18,
        **kwargs,
    ):
        super(HuggingfaceAutoModel, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_tokenizer = self.get_label_tokenizer()
        output_size = self.get_output_size(self.label_tokenizer)
        hidden_dim = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_keys[0]]
        x = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        x = x.to(self.device)
        embeddings = self.model(**x).pooler_output
        logits = self.fc(embeddings)
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    from pyhealth.datasets import MedicalTranscriptionsDataset, get_dataloader
    from torchvision import transforms

    base_dataset = MedicalTranscriptionsDataset(
        root="/srv/local/data/zw12/raw_data/MedicalTranscriptions"
    )

    sample_dataset = base_dataset.set_task()

    train_loader = get_dataloader(sample_dataset, batch_size=16, shuffle=True)

    model = HuggingfaceAutoModel(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        dataset=sample_dataset,
        feature_keys=[
            "transcription",
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