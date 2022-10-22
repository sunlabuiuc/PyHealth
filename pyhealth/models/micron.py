from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer
from pyhealth.models.utils import get_last_visit


class MICRONLayer(nn.Module):
    """This MICRON layer.
    Args:
        input_size: the embedding size of the input
        output_size: the embedding size of the output
        dropout: dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.5,
    ):
        super(MICRONLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.health_net = nn.Linear(input_size, hidden_size)
        self.prescription_net = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        Args:
            x: [batch size, seq len, input_size]
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        health_rep = self.health_net(x)  # (batch, visit, input_size)

        if self.training:
            health_rep_cur = health_rep[:, :-1, :]  # (batch, visit-1, input_size)
            health_rep_last = health_rep[:, 1:, :]  # (batch, visit-1, input_size)
            health_residual_rep = (
                health_rep_cur - health_rep_last
            )  # (batch, visit-1, input_size)

            # drug representation
            drug_rep = self.prescription_net(health_rep)
            drug_residual_rep = self.prescription_net(health_residual_rep)
            return drug_rep, drug_residual_rep

        else:
            drug_rep = self.prescription_net(health_rep)
            return drug_rep


class MICRON(BaseModel):
    """MICRON Class, use "task" as key to identify specific MICRON model and route there
    Args:
        dataset: the dataset object
        tables: the list of table names to use
        target: the target table name
        mode: the mode of the model, "multilabel", "multiclass" or "binary"
        embedding_dim: the embedding dimension
        hidden_dim: the hidden dimension
    """

    def __init__(
        self,
        dataset: BaseDataset,
        tables: List[str],
        target: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(MICRON, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # define the tokenizers
        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain), special_tokens=["<pad>", "<unk>"]
            )
        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))

        # define the embedding layers for each domain
        self.embeddings = nn.ModuleDict()
        for domain in tables:
            # TODO: use get_pad_token_id() instead of hard code
            self.embeddings[domain] = nn.Embedding(
                self.tokenizers[domain].get_vocabulary_size(),
                embedding_dim,
                padding_idx=0,
            )

        self.micron = MICRONLayer(
            input_size=len(tables) * embedding_dim, hidden_size=hidden_dim, **kwargs
        )
        self.fc = nn.Linear(hidden_dim, self.label_tokenizer.get_vocabulary_size())

    def forward(self, device, **kwargs):
        """
        if "kwargs[domain][0][0] is list" means "use history", then run visit level RNN
        elif "kwargs[domain][0][0] is not list" means not "use history", then run code level RNN
        """
        patient_emb = []
        for domain in self.tables:
            if type(kwargs[domain][0][0]) == list:
                kwargs[domain] = self.tokenizers[domain].batch_encode_3d(kwargs[domain])
                kwargs[domain] = torch.tensor(
                    kwargs[domain], dtype=torch.long, device=device
                )
                # (patient, visit, code, embedding_dim)
                kwargs[domain] = self.embeddings[domain](kwargs[domain])
                # (patient, visit, embedding_dim)
                kwargs[domain] = torch.sum(kwargs[domain], dim=2)
            else:
                raise ValueError("Sample data format is not correct")

            # get mask and run RNN
            mask = torch.sum(kwargs[domain], dim=2) != 0
            mask[:, 0] = 1
            # (patient, hidden_dim)

            patient_emb.append(kwargs[domain])

        # (patient, visit, embedding_dim)
        patient_emb = torch.cat(patient_emb, dim=2)

        if self.training:
            # the reconstruction loss
            drug_rep, drug_residual_rep = self.micron(patient_emb, mask)
            logits = self.fc(drug_rep)
            drug_residual_rep = self.fc(drug_residual_rep)

            rec_loss = (
                1
                / self.label_tokenizer.get_vocabulary_size()
                * torch.sum(
                    (
                        torch.sigmoid(logits[:, 1:, :])
                        - torch.sigmoid(logits[:, :-1, :] + drug_residual_rep)
                    )
                    ** 2
                    * mask[:, 1:].unsqueeze(2)
                )
            )
        else:
            drug_rep = self.micron(patient_emb, mask)
            logits = self.fc(drug_rep)

        logits = get_last_visit(logits, mask)

        # obtain target, loss, prob, pred
        loss, y_true, y_prod, y_pred = self.cal_loss_and_output(
            logits, device, **kwargs
        )

        if self.training:
            return {
                "loss": loss + 1e-1 * rec_loss,
                "y_prob": y_prod,
                "y_pred": y_pred,
                "y_true": y_true,
            }
        else:
            return {
                "loss": loss,
                "y_prob": y_prod,
                "y_pred": y_pred,
                "y_true": y_true,
            }
