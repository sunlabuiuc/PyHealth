from typing import List, Tuple, Union

import torch
import torch.nn as nn

from pyhealth.datasets import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer
from pyhealth.models.rnn import RNNLayer


class RETAINLayer(nn.Module):
    """The separate callable RETAIN layer.
    Args:
        input_size: the embedding size of the input
        output_size: the embedding size of the output
        num_layers: the number of layers in the RNN
        dropout: dropout rate
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super(RETAINLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.alpha_gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.beta_gru = nn.GRU(input_size, hidden_size, batch_first=True)

        self.alpha_li = nn.Linear(input_size, 1)
        self.beta_li = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """Using the sum of the embedding as the output of the transformer
        Args:
            x: [batch size, seq len, input_size]
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)

        g, _ = self.alpha_gru(x)  # (patient, seq_len, hidden_size)
        h, _ = self.beta_gru(x)  # (patient, seq_len, hidden_size)

        # TOFIX: mask out the visit (by adding a large negative number 1e10)
        # however, it does not work better than not mask out
        attn_g = torch.softmax(self.alpha_li(g), dim=1)  # (patient, seq_len, 1)
        # attn_g = torch.softmax((self.alpha_li(g) - mask[:, :, 0].unsqueeze(-1) * 1e10), dim=1)  # (patient, seq len, 1)
        attn_h = torch.tanh(self.beta_li(h))  # (patient, seq_len, hidden_size)

        c = attn_g * attn_h * x  # (patient, seq_len, hidden_size)
        c = torch.sum(c, dim=1)  # (patient, hidden_size)
        return c


class RETAIN(BaseModel):
    """RETAIN Class, use "task" as key to identify specific RETAIN model and route there
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
        super(RETAIN, self).__init__(
            dataset=dataset,
            tables=tables,
            target=target,
            mode=mode,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.tokenizers = {}
        for domain in tables:
            self.tokenizers[domain] = Tokenizer(
                dataset.get_all_tokens(key=domain), special_tokens=["<pad>", "<unk>"]
            )
        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=target))

        self.embeddings = nn.ModuleDict()
        for domain in tables:
            # TODO: use get_pad_token_id() instead of hard code
            self.embeddings[domain] = nn.Embedding(
                self.tokenizers[domain].get_vocabulary_size(),
                embedding_dim,
                padding_idx=0,
            )

        self.retain = nn.ModuleDict()
        for domain in tables:
            self.retain[domain] = RETAINLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        self.fc = nn.Linear(
            len(tables) * hidden_dim, self.label_tokenizer.get_vocabulary_size()
        )

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
            elif type(kwargs[domain][0][0]) in [int, str]:
                kwargs[domain] = self.tokenizers[domain].batch_encode_2d(kwargs[domain])
                kwargs[domain] = torch.tensor(
                    kwargs[domain], dtype=torch.long, device=device
                )
                # (patient, code, embedding_dim)
                kwargs[domain] = self.embeddings[domain](kwargs[domain])
            else:
                raise ValueError("Sample data format is not correct")

            # get mask and run RNN
            mask = torch.sum(kwargs[domain], dim=2) != 0
            mask[:, 0] = 1
            # (patient, hidden_dim)
            domain_emb = self.retain[domain](kwargs[domain], mask)
            patient_emb.append(domain_emb)

        # (patient, hidden_dim * N_tables)
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        # obtain target, loss, prob, pred
        loss, y_true, y_prod, y_pred = self.cal_loss_and_output(
            logits, device, **kwargs
        )

        return {
            "loss": loss,
            "y_prob": y_prod,
            "y_pred": y_pred,
            "y_true": y_true,
        }
