from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGEBaseModel(ABC, nn.Module):
    """ Abstract class for Knowledge Graph Embedding models.

    Args:
        e_num: the number of entities in the dataset.
        r_num: the number of relations in the dataset.
        e_dim: the hidden embedding size for entity, 500 by default.
        r_dim: the hidden embedding size for relation, 500 by default.
        ns: negative sampling technique to use: "uni" (uniform) or "adv" (self-adversarial).
        gamma: fixed margin (only need when ns="adv").
    """


    def __init__(
        self, 
        e_num: int, 
        r_num: int,
        e_dim: int = 500,
        r_dim: int = 500,
        ns: str = "uni",
        gamma: float = None
        
    ):
        super(KGEBaseModel, self).__init__()
        self.e_num = e_num
        self.r_num = r_num
        self.e_dim = e_dim
        self.r_dim = r_dim
        self.ns = ns
        self.eps = 2.0


        self.E_emb = nn.Parameter(torch.zeros(self.e_num, self.e_dim))
        self.R_emb = nn.Parameter(torch.zeros(self.r_num, self.r_dim))

        if ns == "adv":
            self.emb_range = nn.Parameter(
                torch.Tensor(
                    [(nn.Parameter(torch.Tensor([gamma]), requires_grad=False).item() + self.eps) / e_dim]
                     ), 
                requires_grad=False
            )

            nn.init.uniform_(
                tensor=self.E_emb, a=-self.emb_range.item(), b=self.emb_range.item()
            )

            nn.init.uniform_(
                tensor=self.R_emb, a=-self.emb_range.item(), b=self.emb_range.item()
            )

        else:
            nn.init.x





