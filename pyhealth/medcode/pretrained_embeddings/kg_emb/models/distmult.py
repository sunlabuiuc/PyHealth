import torch

from pyhealth.datasets import SampleDataset

from .kg_base import KGEBaseModel


class DistMult(KGEBaseModel):
    """DistMult

    Paper: Yang, B., Yih, W.T., He, X., Gao, J. and Deng, L. Embedding entities and
    relations for learning and inference in knowledge bases. ICLR 2015.

    Examples:
        >>> from pyhealth.processors import KGTripleProcessor
        >>> class _Toy:
        ...     input_processors = {
        ...         "triple": KGTripleProcessor(num_entities=2, num_relations=1)
        ...     }
        >>> model = DistMult(_Toy(), e_dim=4, r_dim=4, ns="uniform")
        >>> tuple(model.E_emb.shape)
        (2, 4)
    """
    def __init__(
        self, 
        dataset: SampleDataset, 
        e_dim: int = 300, 
        r_dim: int = 300, 
        ns: str = "adv", 
        gamma: float = 24.0, 
        use_subsampling_weight: bool = False, 
        use_regularization: str = 'l3',
        mode: str = "multiclass",
        negative_sampling: int | None = None,
        ):
        super().__init__(
            dataset, e_dim, r_dim, ns, gamma, use_subsampling_weight,
            use_regularization, mode, negative_sampling=negative_sampling,
        )
    

    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(sample_batch, mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg


    def l3_regularization(self):
        reg_l3 = self.E_emb.norm(p=3) **3 + self.R_emb.norm(p=3) **3
        return reg_l3


    def calc(self, head, relation, tail, mode='pos'):

        if mode == 'head':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score
