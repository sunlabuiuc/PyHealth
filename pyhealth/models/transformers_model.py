from typing import List, Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class TransformersModel(BaseModel):
    """Transformers class for Huggingface models.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        model_name: str,
    ):
        super(TransformersModel, self).__init__(
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