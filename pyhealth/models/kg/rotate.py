from.kg_base import KGEBaseModel
from pyhealth.datasets import SampleBaseDataset

import torch


class RotatE(KGEBaseModel):
    """ RotatE

        Paper: Sun, Z., Deng, Z.H., Nie, J.Y. and Tang, J., 2019. 
        Rotate: Knowledge graph embedding by relational rotation in complex space. ICLR 2019.

    """

    def __init__(
        self, 
        dataset: SampleBaseDataset, 
        e_dim: int = 500, 
        r_dim: int = 500, 
        ns='adv', 
        gamma=6.0     
        ):
        super().__init__(dataset, e_dim, r_dim, ns, gamma)
        self.pi = 3.14159265358979323846
    
    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg
       

    def forward(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)

        head_re, head_im = torch.chunk(head, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/self.pi)

        relation_re = torch.cos(phase_relation)
        relation_im = torch.sin(phase_relation)

        if mode == 'head':
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            re_score = re_score - head_re
            im_score = im_score - head_im
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            re_score = re_score - tail_re
            im_score = im_score - tail_im

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.margin.item() - score.sum(dim = 2)
        return score
