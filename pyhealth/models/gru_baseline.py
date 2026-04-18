from typing import Dict
 
import torch
import torch.nn as nn
 
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.rnn import EmbeddingModel
 
 
class GRUBaseline(BaseModel):
    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super(GRUBaseline, self).__init__(dataset=dataset)
 
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
 
        assert len(self.label_keys) == 1, "GRUBaseline supports exactly one label key."
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]
 
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
 
        self.grus = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.grus[feature_key] = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
 
        output_size = self.get_output_size()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(len(self.feature_keys) * hidden_dim, output_size)
 
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        inputs = {}
        masks = {}
 
        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)
 
            schema = self.dataset.input_processors[feature_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None
 
            if value is None:
                raise ValueError(f"Feature '{feature_key}' must contain 'value' in schema.")
 
            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask
 
        embedded = self.embedding_model(inputs, masks=masks)
 
        patient_emb = []
        for feature_key in self.feature_keys:
            x = embedded[feature_key]
 
            if x.dim() == 4:
                x = x.sum(dim=2)
            elif x.dim() == 2:
                x = x.unsqueeze(1)
 
            output, _ = self.grus[feature_key](x)
            x = output[:, -1, :]
            patient_emb.append(x)
 
        patient_emb = torch.cat(patient_emb, dim=1)
        patient_emb = self.dropout(patient_emb)
        logits = self.fc(patient_emb)
 
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
 
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
