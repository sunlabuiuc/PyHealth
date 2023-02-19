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
        ns: negative sampling technique to use: can be "uniform", "normal" or "adv" (self-adversarial).
        gamma: fixed margin (only need when ns="adv").
    """


    def __init__(
        self, 
        e_num: int, 
        r_num: int,
        e_dim: int = 500,
        r_dim: int = 500,
        ns: str = "uniform",
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

        elif ns == "normal":
            nn.init.xavier_normal_(tensor=self.E_emb)
            nn.init.xavier_normal_(tensor=self.R_emb)
        
        else:
            nn.init.xavier_uniform_(tensor=self.E_emb)
            nn.init.xavier_uniform_(tensor=self.R_emb)

    
    def data_process(self, sample_batch, mode):
        """ Data process function which converts the batch data batch into a batch of head, relation, tail

        Args:
            mode: 
                (1) 'pos': for possitive samples  
                (2) 'head': for negative samples with head prediction
                (3) 'tail' for negative samples with tail prediction
            sample_batch: 
                (1) If mode is 'pos', the sample_batch will be in shape of (batch_size, 3) where the 1-dim are 
                    triples of positive_sample in the format [head, relation, tail]
                (2) If mode is 'head', the sample_batch will be in shape of (batch size, 2) where the 1-dim are
                    tuples of (positive_sample, negative_sample), where positive_sample is a triple [head, relation, tail]
                    and negative triple
        
        
        """
        batch_size, negative_sample_size = sample_batch.size(0), 1 if mode == 'pos' else sample_batch.size(1)

        head_part, relation_part, tail_part = torch.split(sample_batch, [1, 1, 1], dim=1) if mode == 'pos' else (sample_batch[:, 0:1], sample_batch[:, 1:2], sample_batch[:, 2:3])

        head_index = (head_part.view(-1) if mode == 'head' else head_part[:, 0]) if mode != 'pos' else head_part[:, 0]
        tail_index = (tail_part.view(-1) if mode == 'tail' else tail_part[:, 0]) if mode != 'pos' else tail_part[:, 0]

        head = self.entity_embedding[head_index].unsqueeze(1)
        relation = self.relation_embedding[relation_part[:, 0]].unsqueeze(1)
        tail = self.entity_embedding[tail_index].view(batch_size, negative_sample_size, -1)

        return head, relation, tail

        

    def forward(self, sample_batch, mode='pos'):





