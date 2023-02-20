from.kg_base import KGEBaseModel
import torch


class DistMult(KGEBaseModel):
    """ DistMult

    Paper: Yang, B., Yih, W.T., He, X., Gao, J. and Deng, L. Embedding entities and 
    relations for learning and inference in knowledge bases. ICLR 2015.

    """
    def __init__(self):
        super().__init__()
    
    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg


    def l3_regularization(self):
        reg_l3 = self.E_emb.norm(p=3) **3 + self.R_emb.norm(p=3) **3
        return reg_l3


    def forward(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)

        if mode == 'head':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score