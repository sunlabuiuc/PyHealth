from typing import List

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit


class MICRONLayer(nn.Module):
    """This MICRON layer.
    
    Args:
        input_size: the embedding size of the input
        output_size: the embedding size of the output
        dropout: dropout rate
    
    **Examples:**
        >>> from pyhealth.models import MICRONLayer
        >>> input = torch.randn(3, 128, 5) # [batch size, seq len, input_size]
        >>> model = MICRONLayer(5, 64, 0.5)
        
        >>> model.train()
        MICRONLayer(
        (dropout_layer): Dropout(p=0.5, inplace=False)
        (health_net): Linear(in_features=5, out_features=64, bias=True)
        (prescription_net): Linear(in_features=64, out_features=64, bias=True)
        )
        >>> [item.shape for item in model(input, mask=None)]
        [torch.Size([3, 128, 64]), torch.Size([3, 127, 64])]
        
        >>> model.eval()
        MICRONLayer(
        (dropout_layer): Dropout(p=0.5, inplace=False)
        (health_net): Linear(in_features=5, out_features=64, bias=True)
        (prescription_net): Linear(in_features=64, out_features=64, bias=True)
        )
        >>> model(input, mask=None).shape
        torch.Size([3, 128, 64]) # [batch size, hidden_size]
        
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_labels: int,
            dropout: float = 0.5,
    ):
        super(MICRONLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.health_net = nn.Linear(input_size, hidden_size)
        self.prescription_net = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_labels)

    @staticmethod
    def compute_reconstruction_loss(logits, residual_logits, mask):
        rec_loss = torch.mean(
            torch.square(
                torch.sigmoid(logits[:, 1:, :])
                - torch.sigmoid(logits[:, :-1, :] + residual_logits)
            )
            * mask[:, 1:].unsqueeze(2)
        )
        return rec_loss

    def forward(self, x: torch.tensor):
        """
        Args:
            x: [batch size, seq len, input_size]
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        mask = (torch.sum(x, dim=2) != 0)
        health_rep = self.health_net(x)  # (batch, visit, input_size)

        if self.training:
            # (batch, visit-1, input_size)
            health_rep_cur = health_rep[:, :-1, :]
            # (batch, visit-1, input_size)
            health_rep_last = health_rep[:, 1:, :]
            # (batch, visit-1, input_size)
            health_residual_rep = (health_rep_cur - health_rep_last)
            # drug representation
            drug_rep = self.prescription_net(health_rep)
            drug_residual_rep = self.prescription_net(health_residual_rep)
            #  logits
            logits = self.fc(drug_rep)
            residual_logits = self.fc(drug_residual_rep)
            rec_loss = self.compute_reconstruction_loss(logits, residual_logits, mask)
            logits = get_last_visit(logits, mask)
            return logits, rec_loss

        else:
            drug_rep = self.prescription_net(health_rep)
            logits = self.fc(drug_rep)
            logits = get_last_visit(logits, mask)
            return logits


class MICRON(BaseModel):
    """MICRON Class, use "task" as key to identify specific MICRON model and route there
    
    Args:
        dataset: the dataset object
        feature_keys: the list of table names to use
        label_key: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension
    
    **Examples:**
        >>> from pyhealth.datasets import OMOPDataset
        >>> dataset = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ... ) # load dataset
        >>> from pyhealth.tasks import drug_recommendation_omop_fn
        >>> dataset.set_task(drug_recommendation_omop_fn) # set task
        
        >>> from pyhealth.models import MICRON
        >>> model = MICRON(
        ...     dataset=dataset,
        ...     tables=["conditions", "procedures"],
        ...     target="label",
        ...     mode="multilabel",
        ... )
        
    """

    def __init__(
            self,
            dataset: BaseDataset,
            feature_keys: List[str],
            label_key: str,
            embedding_dim: int = 128,
            hidden_dim: int = 128,
            **kwargs
    ):
        super(MICRON, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.feat_tokenizers = self._get_feature_tokenizers()
        self.label_tokenizer = self._get_label_tokenizer()
        self.embeddings = self._get_embeddings(self.feat_tokenizers, embedding_dim)

        self.micron = MICRONLayer(
            input_size=len(feature_keys) * embedding_dim,
            hidden_size=hidden_dim,
            num_labels=self.label_tokenizer.get_vocabulary_size(),
            **kwargs
        )

    def forward(self, device, **kwargs):
        patient_emb = []
        for feature_key in self.feature_keys:
            assert type(kwargs[feature_key][0][0]) == list
            x = self.feat_tokenizers[feature_key].batch_encode_3d(kwargs[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=device)
            # (patient, visit, code, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, hidden_dim)
            patient_emb.append(x)

        # (patient, visit, embedding_dim)
        patient_emb = torch.cat(patient_emb, dim=2)

        if self.training:
            # the reconstruction loss
            logits, rec_loss = self.micron(patient_emb)

        else:
            logits = self.micron(patient_emb)

        # obtain target, loss, prob, pred
        loss, y_true, y_prob = self._calculate_output(logits, kwargs[self.label_key])

        if self.training:
            return {
                "loss": loss + 1e-1 * rec_loss,
                "y_prob": y_prob,
                "y_true": y_true,
            }
        else:
            return {
                "loss": loss,
                "y_prob": y_prob,
                "y_true": y_true,
            }


if __name__ == '__main__':
    from pyhealth.datasets import MIMIC3Dataset
    from torch.utils.data import DataLoader
    from pyhealth.utils import collate_fn_dict
    from pyhealth.tasks import drug_recommendation_mimic3_fn

    dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )

    # visit level + multilabel
    dataset.set_task(drug_recommendation_mimic3_fn)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
    )
    model = MICRON(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key="label",
    )
    model.to("cuda")
    batch = iter(dataloader).next()
    output = model(**batch, device="cuda")
    print(output["loss"])
