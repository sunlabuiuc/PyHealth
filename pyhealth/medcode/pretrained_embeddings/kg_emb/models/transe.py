import torch

from pyhealth.datasets import SampleDataset

from .kg_base import KGEBaseModel


class TransE(KGEBaseModel):
    """TransE

    Paper: Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J. and Yakhnenko,
    Translating embeddings for modeling multi-relational data. NIPS 2013.

    Examples:
        >>> from pyhealth.processors import KGTripleProcessor
        >>> class _Toy:
        ...     input_processors = {
        ...         "triple": KGTripleProcessor(num_entities=2, num_relations=1)
        ...     }
        >>> model = TransE(_Toy(), e_dim=4, r_dim=4)
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
        use_regularization: str | None = None,
        mode: str = "multiclass",
        p_norm: int = 1.0,
        negative_sampling: int | None = None,
        ):
        super().__init__(
            dataset, e_dim, r_dim, ns, gamma, use_subsampling_weight,
            use_regularization, mode, negative_sampling=negative_sampling,
        )

        self.p_norm = p_norm


    def regularization(self, sample_batch, mode='pos'):
        head, relation, tail = self.data_process(sample_batch=sample_batch, mode=mode)
        reg = (torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)) / 3
        return reg


    def calc(self, head, relation, tail, mode='pos'):

        if mode == 'head':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = torch.norm(score, p=self.p_norm, dim=2)

        if self.ns == 'adv':
            score = self.margin.item() - score

        return score
