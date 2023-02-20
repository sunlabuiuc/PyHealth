from.kg_base import KGEBaseModel
import torch


class TransE(KGEBaseModel):
    """ TransE

    Paper: Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. and Yakhnenko,
    Translating embeddings for modeling multi-relational data. NIPS 2013.

    """

    def __init__(self, p_norm=1):
        super(TransE, self).__init__()
        self.p_norm = p_norm


    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg


    def forward(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)

        if mode == 'head':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=self.p_norm, dim=2)

        if self.ns == 'adv':
            score = self.margin.item() - score

        return score