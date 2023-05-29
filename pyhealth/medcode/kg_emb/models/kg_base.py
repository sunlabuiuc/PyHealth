from abc import ABC
from pyhealth.datasets import SampleBaseDataset

import torch
import time
import numpy as np
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
        mode: evaluation metric type, one of "binary", "multiclass", or "multilabel", "multiclass" by default

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
        use_regularization: str = None,
        mode: str = "multiclass"
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
        self.mode = mode
        
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

    
    def train_neg_sample_gen(self, gt_head, gt_tail, negative_sampling):
        """
        (only run in train batch) 
        This function creates negative triples for training (sampling size: negative_sampling)
             with ground truth masked.
        """
        negative_sample_head = []
        negative_sample_tail = []
        for i in range(len(gt_head)):
            # head, relation, tail = triples[i]
            
            ## negative samples for head prediction
            negative_sample_list_head = []
            negative_sample_size_head = 0

            while negative_sample_size_head < negative_sampling:
                negative_sample = np.random.randint(self.e_num, size=negative_sampling*2)
                mask = np.in1d(
                    negative_sample,
                    gt_head[i],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list_head.append(negative_sample)
                negative_sample_size_head += negative_sample.size
            
            ## negative samples for tail prediction
            negative_sample_list_tail = []
            negative_sample_size_tail = 0

            while negative_sample_size_tail < negative_sampling:
                negative_sample = np.random.randint(self.e_num, size=negative_sampling*2)
                mask = np.in1d(
                    negative_sample,
                    gt_tail[i],
                    assume_unique=True,
                    invert=True
                )
                negative_sample = negative_sample[mask]
                negative_sample_list_tail.append(negative_sample)
                negative_sample_size_tail += negative_sample.size

            neg_head = torch.LongTensor(np.concatenate(negative_sample_list_head)[:negative_sampling])
            neg_tail = torch.LongTensor(np.concatenate(negative_sample_list_tail)[:negative_sampling])
            negative_sample_head.append(neg_head)
            negative_sample_tail.append(neg_tail)
        
        negative_sample_head = torch.stack([d for d in negative_sample_head], dim=0)
        negative_sample_tail = torch.stack([d for d in negative_sample_tail], dim=0)

        return negative_sample_head, negative_sample_tail


    def test_neg_sample_filter_bias_gen(self, triples, gt_head, gt_tail):
        """
        (only run in val/test batch) 
        This function creates negative triples for validation/testing with ground truth masked.
        """

        negative_sample_head = []
        negative_sample_tail = []
        filter_bias_head = []
        filter_bias_tail = []

        for i in range(len(triples)):
            head, _, tail = triples[i]
            gt_h_ = gt_head[i]
            gt_h = gt_h_[:]
            gt_h.remove(head)
            gt_t_ = gt_tail[i]
            gt_t = gt_t_[:]
            gt_t.remove(tail)

            neg_head = np.arange(0, self.e_num)
            neg_head[gt_h] = head
            neg_head = torch.LongTensor(neg_head)

            neg_tail = np.arange(0, self.e_num)
            neg_tail[gt_t] = tail
            neg_tail = torch.LongTensor(neg_tail)

            fb_head = np.zeros(self.e_num)
            fb_head[gt_h] = -1
            fb_head = torch.LongTensor(fb_head)

            fb_tail = np.zeros(self.e_num)
            fb_tail[gt_t] = -1
            fb_tail = torch.LongTensor(fb_tail)

            negative_sample_head.append(neg_head)
            negative_sample_tail.append(neg_tail)
            filter_bias_head.append(fb_head)
            filter_bias_tail.append(fb_tail)

        negative_sample_head = torch.stack([d for d in negative_sample_head], dim=0)
        negative_sample_tail = torch.stack([d for d in negative_sample_tail], dim=0)
        filter_bias_head = torch.stack([d for d in filter_bias_head], dim=0)
        filter_bias_tail = torch.stack([d for d in filter_bias_tail], dim=0)

        return negative_sample_head, negative_sample_tail, filter_bias_head, filter_bias_tail


    def calc(self, head, relation, tail, mode='pos'):
        """ score calculation
        Args:
            head:       head entity h
            relation:   relation    r
            tail:       tail entity t
            mode: 
                (1) 'pos': for possitive samples  
                (2) 'head': for negative samples with head prediction
                (3) 'tail' for negative samples with tail prediction
        
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

        positive_sample = torch.stack([torch.LongTensor(d) for d in data['triple']], dim=0).to(self.device)

        if data['train'][0]:
            negative_sample_head, negative_sample_tail = self.train_neg_sample_gen(
                gt_head=data['ground_truth_head'],
                gt_tail=data['ground_truth_tail'],
                negative_sampling=data['hyperparameters'][0]['negative_sampling']
            )

            negative_sample_head, negative_sample_tail = negative_sample_head.to(self.device), negative_sample_tail.to(self.device)
            
            head, relation, tail = self.data_process((positive_sample, negative_sample_head), mode="head")
            neg_score_head = self.calc(head=head, relation=relation, tail=tail, mode="head")
            head, relation, tail = self.data_process((positive_sample, negative_sample_tail), mode="tail")
            neg_score_tail = self.calc(head=head, relation=relation, tail=tail, mode="tail")

            neg_score = neg_score_head + neg_score_tail

            if self.ns == 'adv':
                neg_score = (F.softmax(neg_score * 1.0, dim=1).detach() * F.logsigmoid(-neg_score)).sum(dim=1)

            else:
                neg_score = F.logsigmoid(-neg_score).mean(dim=1)

            head, relation, tail = self.data_process((positive_sample), mode="pos")
            pos_score = F.logsigmoid(self.calc(head=head, relation=relation, tail=tail)).squeeze(dim=1)

            subsampling_weight = torch.cat([d for d in data['subsampling_weight']], dim=0).to(self.device)
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

        else: # valid/test
            inputs = self.test_neg_sample_filter_bias_gen(
                    triples=data['triple'],
                    gt_head=data['ground_truth_head'],
                    gt_tail=data['ground_truth_tail']
                )

            # inputs, mode = (data['positive_sample'], data['negative_sample'], data['filter_bias']), data['mode']
            inputs = [x.to(self.device) for x in inputs]
            negative_sample_head, negative_sample_tail, filter_bias_head, filter_bias_tail = inputs
            head, relation, tail = self.data_process((positive_sample, negative_sample_head), mode="head")
            score_head = self.calc(head=head, relation=relation, tail=tail, mode="head")
            head, relation, tail = self.data_process((positive_sample, negative_sample_tail), mode="tail")
            score_tail = self.calc(head=head, relation=relation, tail=tail, mode="tail")
            score_head += filter_bias_head
            score_tail += filter_bias_tail

            score = score_head + score_tail
            loss = (-F.logsigmoid(-score).mean(dim=1)).mean()
            
            
            y_true_head = positive_sample[:, 0]
            y_true_tail = positive_sample[:, 2]

            y_true = torch.cat((y_true_head, y_true_tail))
            y_prob = torch.cat((score_head, score_tail))

            return {
                "loss": loss,
                "y_true": y_true,
                "y_prob": y_prob
                }
    
    def inference(self, head=None, relation=None, tail=None, top_k=1):
        # Check if two or more arguments are None
        if sum(arg is None for arg in (head, relation, tail)) >= 2:
            print("At least 2 place holders need to be filled. ")
            return
        
        mode = "head" if head is None else ("tail" if tail is None else ("relation" if relation is None else "clf"))

        if mode == "head":
            tail_index = torch.tensor(tail)
            relation_index = torch.tensor(relation)
            relation = torch.index_select(self.R_emb, dim=0, index=relation_index).unsqueeze(1)
            tail = torch.index_select(self.E_emb, dim=0, index=tail_index).unsqueeze(1)
            head_all_idx = torch.tensor(np.arange(0, self.e_num))
            head_all = torch.index_select(self.E_emb, dim=0, index=head_all_idx).unsqueeze(1)
            score_head = self.calc(head=head_all, relation=relation, tail=tail, mode="head")
            result_eid = torch.topk(score_head.flatten(), top_k).indices
            return result_eid.tolist()

        if mode == "tail":
            head_index = torch.tensor(head)
            relation_index = torch.tensor(relation)
            head = torch.index_select(self.E_emb, dim=0, index=head_index).unsqueeze(1)
            relation = torch.index_select(self.R_emb, dim=0, index=relation_index).unsqueeze(1)
            tail_all_idx = torch.tensor(np.arange(0, self.e_num))
            tail_all = torch.index_select(self.E_emb, dim=0, index=tail_all_idx).unsqueeze(1)
            score_tail = self.calc(head=head, relation=relation, tail=tail_all, mode="tail")
            result_eid = torch.topk(score_tail.flatten(), top_k).indices
            return result_eid.tolist()
        
        if mode == "relation":
            print("Not implemented yet.")

        if mode == "clf":
            head_index = torch.tensor(head)
            relation_index = torch.tensor(relation)
            tail_index = torch.tensor(tail)
            head = torch.index_select(self.E_emb, dim=0, index=head_index).unsqueeze(1)
            relation = torch.index_select(self.R_emb, dim=0, index=relation_index).unsqueeze(1)
            tail = torch.index_select(self.E_emb, dim=0, index=tail_index).unsqueeze(1)
            score = self.calc(head=head, relation=relation, tail=tail, mode="pos")
            return score.tolist()

    
    def from_pretrained(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.update_embedding_size(state_dict)
        self.load_state_dict(state_dict)
        return

    def update_embedding_size(self, state_dict):
        e_emb_key = 'E_emb'
        r_emb_key = 'R_emb'
        
        if e_emb_key in state_dict and r_emb_key in state_dict:
            _, new_e_dim = state_dict[e_emb_key].shape
            _, new_r_dim = state_dict[r_emb_key].shape
            
            if new_e_dim != self.e_dim or new_r_dim != self.r_dim:
                self.e_dim = new_e_dim
                self.r_dim = new_r_dim
                
                self.E_emb = nn.Parameter(torch.zeros(self.e_num, self.e_dim))
                self.R_emb = nn.Parameter(torch.zeros(self.r_num, self.r_dim))
        return


            









