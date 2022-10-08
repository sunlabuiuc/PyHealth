from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.data import BaseDataset
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_default_loss_module
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
        self.rnn = rnn_module(input_size,
                              hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=bidirectional,
                              batch_first=True)
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
        x = rnn_utils.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        if self.bidirectional:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            forward_last_outputs = outputs[torch.arange(batch_size), (length - 1), 0, :]
            backward_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([forward_last_outputs, backward_last_outputs], dim=-1)
            outputs = outputs.view(batch_size, outputs.shape[1], -1)
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return last_outputs, outputs
        outputs = self.down_projection(outputs)
        last_outputs = outputs[torch.arange(batch_size), (length - 1), :]
        return last_outputs, outputs


class RNN(BaseModel):
    """RNN Class, use "task" as key to identify specific RNN model and route there"""

    def __init__(self,
                 dataset: BaseDataset,
                 input_domains: Union[List[str], Tuple[str]],
                 output_domain: str,
                 mode: str,
                 embedding_dim: int = 128,
                 hidden_dim: int = 128,
                 **kwargs):
        super(RNN, self).__init__(dataset=dataset,
                                       input_domains=input_domains,
                                       output_domain=output_domain,
                                       mode=mode)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.tokenizers = {}
        for domain in input_domains:
            self.tokenizers[domain] = Tokenizer(dataset.get_all_tokens(key=domain),
                                                special_tokens=["<pad>", "<unk>"])
        # TODO: some task like mortality prediction do not need tokenizer for output domain
        self.label_tokenizer = Tokenizer(dataset.get_all_tokens(key=output_domain),
                                         special_tokens=["<pad>", "<unk>"])

        self.embeddings = nn.ModuleDict()
        for domain in input_domains:
            # TODO: use get_pad_token_id() instead of hard code
            self.embeddings[domain] = nn.Embedding(self.tokenizers[domain].get_vocabulary_size(),
                                                   embedding_dim,
                                                   padding_idx=0)

        self.rnn = RNNLayer(input_size=embedding_dim, hidden_size=hidden_dim, **kwargs)
        # TODO: remove the -2 hard code for <pad> and <unk>
        self.fc = nn.Linear(hidden_dim, self.label_tokenizer.get_vocabulary_size() - 2)
        self.loss_fn = get_default_loss_module(mode=mode)

    def forward(self, device, training, **kwargs):
        for domain in self.input_domains:
            # (patient, visit, code)
            kwargs[domain] = self.tokenizers[domain].batch_encode_3d(kwargs[domain])
            kwargs[domain] = torch.tensor(kwargs[domain], dtype=torch.long, device=device)
            # (patient, visit, code, embedding_dim)
            kwargs[domain] = self.embeddings[domain](kwargs[domain])
            # (patient, visit, embedding_dim)
            kwargs[domain] = torch.sum(kwargs[domain], dim=2)
        visit_emb = torch.stack([kwargs[domain] for domain in self.input_domains], dim=2).mean(2)
        visit_mask = torch.sum(visit_emb, dim=2) != 0
        # (patient, hidden_dim)
        patient_emb, _ = self.rnn(visit_emb, visit_mask)
        logits = self.fc(patient_emb)

        # target
        y = torch.zeros(len(kwargs[self.output_domain]), self.label_tokenizer.get_vocabulary_size(),
                        device=device)
        kwargs[self.output_domain] = [visits[-1] for visits in kwargs[self.output_domain]]
        kwargs[self.output_domain] = self.label_tokenizer.batch_encode_2d(kwargs[self.output_domain],
                                                                          padding=False,
                                                                          truncation=False)
        for idx, sample in enumerate(kwargs[self.output_domain]):
            y[idx, sample] = 1

        # remove <pad> and <unk> from loss
        y = y[:, 2:]

        # loss
        loss = self.loss_fn(logits, y.to(device))
        return {"loss": loss, "y_prob": logits, "y_true": y}
