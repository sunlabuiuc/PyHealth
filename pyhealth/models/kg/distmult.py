from.kg_base import KGEBaseModel
from pyhealth.datasets import SampleBaseDataset
import torch


class DistMult(KGEBaseModel):
    """ DistMult

    Paper: Yang, B., Yih, W.T., He, X., Gao, J. and Deng, L. Embedding entities and 
    relations for learning and inference in knowledge bases. ICLR 2015.

    """
    def __init__(
        self, 
        dataset: SampleBaseDataset, 
        e_dim: int = 300, 
        r_dim: int = 300, 
        ns: str = "uniform", 
        gamma: float = None, 
        use_subsampling_weight: bool = False, 
        use_regularization: str = 'l3',
        ):
        super().__init__(dataset, e_dim, r_dim, ns, gamma, use_subsampling_weight, use_regularization)
    

    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(sample_batch, mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg


    def l3_regularization(self):
        reg_l3 = self.E_emb.norm(p=3) **3 + self.R_emb.norm(p=3) **3
        return reg_l3


    def calc(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(sample_batch, mode)

        if mode == 'head':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score