from abc import ABC
from pyhealth.datasets import SampleBaseDataset

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
        use_subsampling_weight: whether to use subsampling weight (like in word2vec) or not, False by default.
        use_regularization: whether to apply regularization or not, False by default.

    """

    @property
    def device(self):
        """Gets the device of the model."""
        return self._dummy_param.device


    def __init__(
        self, 
        dataset: SampleBaseDataset,
        e_dim: int = 500,
        r_dim: int = 500,
        ns: str = "uniform",
        gamma: float = None,
        use_subsampling_weight: bool = False,
        use_regularization: str = None
    ):
        super(KGEBaseModel, self).__init__()
        self.e_num = dataset.entity_num
        self.r_num = dataset.relation_num
        self.e_dim = e_dim
        self.r_dim = r_dim
        self.ns = ns
        self.eps = 2.0
        self.use_subsampling_weight = use_subsampling_weight
        self.use_regularization = use_regularization
        if gamma != None:
            self.margin = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)

        # used to query the device of the model
        self._dummy_param = nn.Parameter(torch.empty(0))


        self.E_emb = nn.Parameter(torch.zeros(self.e_num, self.e_dim))
        self.R_emb = nn.Parameter(torch.zeros(self.r_num, self.r_dim))

        if ns == "adv":
            self.e_emb_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.eps) / e_dim]), requires_grad=False
            )

            self.r_emb_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.eps) / r_dim]), requires_grad=False
            )

            nn.init.uniform_(
                tensor=self.E_emb, a=-self.e_emb_range.item(), b=self.e_emb_range.item()
            )

            nn.init.uniform_(
                tensor=self.R_emb, a=-self.r_emb_range.item(), b=self.r_emb_range.item()
            )

        elif ns == "normal":
            nn.init.xavier_normal_(tensor=self.E_emb)
            nn.init.xavier_normal_(tensor=self.R_emb)
        
        ## ns == "uniform"
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
        
        if mode == "head" or mode == "tail":
            positive, negative = sample_batch
            batch_size, negative_sample_size = negative.size(0), negative.size(1)
        else:
            positive = sample_batch

        head_index = negative.view(-1) if mode == 'head' else positive[:, 0]
        tail_index = negative.view(-1) if mode == 'tail' else positive[:, 2]

        head_ = torch.index_select(self.E_emb, dim=0, index=head_index)
        head = head_.view(batch_size, negative_sample_size, -1) if mode == 'head' else head_.unsqueeze(1)

        relation = self.R_emb[positive[:, 1]].unsqueeze(1)

        tail_ = torch.index_select(self.E_emb, dim=0, index=tail_index)
        tail = tail_.view(batch_size, negative_sample_size, -1) if mode == 'tail' else tail_.unsqueeze(1)

        return head, relation, tail


    def calc(self, sample_batch, mode='pos'):
        """ score calculation
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
        
        Return:
            score of positive/negative samples, in shape of braodcasted result of calulation with head, tail and relation.
            Example: 
                For a head prediction, suppose we have:
                    head:   torch.Size([16, 9737, 600])
                    rel:    torch.Size([16, 1, 300])
                    tail:   torch.Size([16, 1, 600])

                The unnormalized score will be in shape:
                    score:  torch.Size(16, 9737, 300)
                
                and the normalized score (return value) will be:
                    score:  torch.Size(16, 9737)
                
        """
        raise NotImplementedError


    def forward(self, **data):
        inputs, mode = (data['positive_sample'], data['negative_sample'], data['subsample_weight']), data['mode']
        inputs = [x.to(self.device) for x in inputs]
        pos_sample, neg_sample, subsampling_weight = inputs
        sample_batch = (pos_sample, neg_sample)

        neg_score = self.calc(sample_batch=sample_batch, mode=mode)

        if self.ns == 'adv':
            neg_score = (F.softmax(neg_score * 1.0, dim=1).detach() * F.logsigmoid(-neg_score)).sum(dim=1)

        else:
            neg_score = F.logsigmoid(-neg_score).mean(dim=1)

        pos_score = F.logsigmoid(self.calc(pos_sample)).squeeze(dim=1)

        
        pos_sample_loss = \
            - (subsampling_weight * pos_score).sum()/subsampling_weight.sum() if self.use_subsampling_weight else (- pos_score.mean())
        neg_sample_loss = \
            - (subsampling_weight * neg_score).sum()/subsampling_weight.sum() if self.use_subsampling_weight else (- neg_score.mean())

        loss = (pos_sample_loss + neg_sample_loss) / 2


        if self.use_regularization == 'l3':
            loss = loss + self.l3_regularization()
        elif self.use_regularization != None:
            loss = loss + self.regularization()

        return {"loss": loss}



