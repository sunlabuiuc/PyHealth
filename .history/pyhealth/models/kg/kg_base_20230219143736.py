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
                (2) If mode is 'head' or 'tail', the sample_batch will be in shape of (batch size, 2) where the 1-dim are
                    tuples of (positive_sample, negative_sample), where positive_sample is a triple [head, relation, tail]
                    and negative_sample is a 1-d array (length: e_num) with negative (head or tail) entities indecies filled
                    and positive entities masked.

        Returns:
            head:   torch.Size([batch_size, 1, e_dim]) for tail prediction 
                    or torch.Size([batch_size, negative_sample_size(e_num), e_dim]) for head prediction
            relation: torch.Size([batch_size, 1, r_dim])
            tail:   torch.Size([batch_size, 1, e_dim]) for head prediction 
                    or torch.Size([batch_size, negative_sample_size(e_num), e_dim]) for tail prediction

        
        """
        
        if mode == "head" or mode == "tail-batch":
            positive, negative = sample_batch
            batch_size, negative_sample_size = negative.size(0), negative.size(1)
        else:
            positive = sample_batch

        head_index = negative.view(-1) if mode == 'head-batch' else positive[:, 0]
        tail_index = negative.view(-1) if mode == 'tail-batch' else positive[:, 2]

        head_ = torch.index_select(self.entity_embedding, dim=0, index=head_index)
        head = head_.view(batch_size, negative_sample_size, -1) if mode == 'head-batch' else head_.unsqueeze(1)

        tail_ = torch.index_select(self.entity_embedding, dim=0, index=tail_index)
        tail = tail_.view(batch_size, negative_sample_size, -1) if mode == 'tail-batch' else tail_.unsqueeze(1)

        relation = self.relation_embedding[positive[:, 1]].unsqueeze(1)

        return head, relation, tail


    def forward(self, sample_batch, mode='pos'):
        raise NotImplementedError






