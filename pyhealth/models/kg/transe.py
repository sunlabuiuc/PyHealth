from.kg_base import KGEBaseModel
from pyhealth.datasets import SampleBaseDataset
import torch


class TransE(KGEBaseModel):
    """ TransE

    Paper: Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. and Yakhnenko,
    Translating embeddings for modeling multi-relational data. NIPS 2013.

    """

    def __init__(
        self, 
        dataset: SampleBaseDataset, 
        e_dim: int = 300, 
        r_dim: int = 300, 
        ns: str = "uniform", 
        gamma: float = None, 
        use_subsampling_weight: bool = False, 
        use_regularization: str = None,
        p_norm: int = 1.0
        ):
        super().__init__(dataset, e_dim, r_dim, ns, gamma, use_subsampling_weight, use_regularization)

        self.p_norm = p_norm


    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(sample_batch=sample_batch, mode=mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg


    def calc(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(sample_batch=sample_batch, mode=mode)

        if mode == 'head':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=self.p_norm, dim=2)

        if self.ns == 'adv':
            score = self.margin.item() - score

        return score