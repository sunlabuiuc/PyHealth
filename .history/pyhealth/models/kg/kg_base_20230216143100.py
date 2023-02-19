from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGEBaseModel(nn.Module):
    """ Abstract class for Knowledge Graph Embedding models

    Args:
        e_num: the number of entities in the dataset.
        r_num: the number of relations in the dataset.
        e_dim: the hidden embedding size for entity, 500 by default.
        r_dim: the hidden embedding size for relation, 500 by default.

    """


    def __init__(
        self, 
        e_num: int, r_num: int,
        e_dim: int, r_dim: int,
        
    ):
        super(KGEBaseModel, self).__init__()
        self.e_num = e_num
        self.r_num = r_num
        self.e_dim = e_dim
        self.r_dim = r_dim

        self.E_emb = nn.Parameter(torch.zeros(self.e_num, self.))