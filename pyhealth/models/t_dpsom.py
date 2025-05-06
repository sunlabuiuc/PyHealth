import torch
import torch.nn as nn
from pyhealth.models import BaseModel

class SOMLayer(nn.Module):
    def __init__(self, input_dim, n_prototypes):
        super(SOMLayer, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, input_dim))

    def forward(self, x):
        # x: (batch, embed_dim)
        dists = torch.cdist(x, self.prototypes)  # (batch, n_prototypes)
        return dists


class TDPSOM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_prototypes, **kwargs):
        super(TDPSOM, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.som = SOMLayer(hidden_dim, n_prototypes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_last = self.rnn(x)
        h_last = h_last.squeeze(0)  # (batch, hidden_dim)
        dists = self.som(h_last)
        assignments = torch.argmin(dists, dim=1)
        return assignments

    def loss(self, x, assignments):
        _, h_last = self.rnn(x)
        h_last = h_last.squeeze(0)
        dists = self.som(h_last)
        loss = dists.gather(1, assignments.unsqueeze(1)).mean()
        return loss

