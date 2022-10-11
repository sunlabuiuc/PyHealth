from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.data import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer


class RNNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.tensor, mask: torch.tensor):
        """
        Args:
            x: [batch size, seq len, input_size]
            mask: [batch size, seq len]
        Returns:
            outputs [batch size, seq len, hidden_size]
        """
        # rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        length = torch.sum(mask.int(), dim=-1).cpu()
        x = rnn_utils.pack_padded_sequence(
            x, length, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if self.bidirectional:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            forward_last_outputs = outputs[torch.arange(batch_size), (length - 1), 0, :]
            backward_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat(
                [forward_last_outputs, backward_last_outputs], dim=-1
            )
            outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return last_outputs, outputs
        outputs = self.down_projection(outputs)
        last_outputs = outputs[torch.arange(batch_size), (length - 1), :]
        return last_outputs, outputs


class RNN(BaseModel):
    """RNN Class, use "task" as key to identify specific RNN model and route there"""

    def __init__(
        self,
        dataset: BaseDataset,
        tables: Union[List[str], Tuple[str]],
        target: str,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(RNN, self).__init__(
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

        self.rnn = nn.ModuleDict()
        for domain in tables:
            self.rnn[domain] = RNNLayer(
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
            domain_emb, _ = self.rnn[domain](kwargs[domain], mask)
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