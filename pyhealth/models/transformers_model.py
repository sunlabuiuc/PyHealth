from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel


class TransformersModel(BaseModel):
    """Transformers class for Huggingface models."""

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str,
    ):
        super(TransformersModel, self).__init__(
            dataset=dataset,
        )
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        assert (
            len(self.feature_keys) == 1
        ), "Only one feature key is supported if Transformers is initialized"
        self.feature_key = self.feature_keys[0]
        assert (
            len(self.label_keys) == 1
        ), "Only one label key is supported if RNN is initialized"
        self.label_key = self.label_keys[0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        output_size = self.get_output_size()
        hidden_dim = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        # concat the info within one batch (batch, channel, length)
        x = kwargs[self.feature_key]
        # TODO: max_length should be a parameter
        x = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        x = x.to(self.device)
        # TODO: should not use pooler_output, but use the last hidden state
        embeddings = self.model(**x).pooler_output
        logits = self.fc(embeddings)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
        }


if __name__ == "__main__":
    from pyhealth.datasets import MedicalTranscriptionsDataset, get_dataloader

    base_dataset = MedicalTranscriptionsDataset(
        root="/srv/local/data/zw12/raw_data/MedicalTranscriptions"
    )

    sample_dataset = base_dataset.set_task()

    train_loader = get_dataloader(sample_dataset, batch_size=16, shuffle=True)

    model = TransformersModel(
        dataset=sample_dataset,
        feature_keys=["transcription"],
        label_key="label",
        mode="multiclass",
        model_name="emilyalsentzer/Bio_ClinicalBERT",
    )

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
