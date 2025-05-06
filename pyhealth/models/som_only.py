import torch
import torch.nn as nn
from pyhealth.models import BaseModel

class SOM(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(SOM, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_clusters, input_dim))

    def forward(self, x):
        distances = torch.cdist(x, self.weights)
        return distances

class SOMOnlyModel(BaseModel):
    def __init__(self, input_dim, n_clusters, **kwargs):
        super(SOMOnlyModel, self).__init__(**kwargs)
        self.som = SOM(input_dim, n_clusters)

    def forward(self, x):
        distances = self.som(x)
        assignments = torch.argmin(distances, dim=1)
        return assignments

    def loss(self, x, assignments):
        distances = self.som(x)
        loss = distances.gather(1, assignments.unsqueeze(1)).mean()
        return loss
