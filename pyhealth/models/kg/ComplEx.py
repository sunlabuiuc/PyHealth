from.kg_base import KGEBaseModel
import torch


class ComplEx(KGEBaseModel):
    """ ComplEx

    Paper: Trouillon, T., Welbl, J., Riedel, S., Gaussier, É. and Bouchard, G., 2016, June. 
    Complex embeddings for simple link prediction. In International conference on machine learning (pp. 2071-2080). PMLR

    """

    def __init__(self):
        super().__init__()
    
    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)
        head_re, head_im = torch.chunk(head, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        reg = (torch.mean(head_re ** 2) + 
                 torch.mean(head_im ** 2) + 
                 torch.mean(tail_re ** 2) +
                 torch.mean(tail_im ** 2) +
                 torch.mean(relation_re ** 2) +
                 torch.mean(relation_im ** 2)) / 6

        return reg

    
    def l3_regularization(self):
        reg_l3 = self.E_emb.norm(p=3) **3 + self.R_emb.norm(p=3) **3
        return reg_l3


    def forward(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(self, sample_batch, mode)
        head_re, head_im = torch.chunk(head, 2, dim=2)
        relation_re, relation_im = torch.chunk(relation, 2, dim=2)
        tail_re, tail_im = torch.chunk(tail, 2, dim=2)

        if mode == 'head':
            re_score = relation_re * tail_re + relation_im * tail_im
            im_score = relation_re * tail_im - relation_im * tail_re
            score = head_re * re_score + head_im * im_score
        else:
            re_score = head_re * relation_re - head_im * relation_im
            im_score = head_re * relation_im + head_im * relation_re
            score = re_score * tail_re + im_score * tail_im

        score = score.sum(dim = 2)
        return score
