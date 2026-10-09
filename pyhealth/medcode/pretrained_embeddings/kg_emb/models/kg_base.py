import numbers
import warnings
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.processors import KGTripleProcessor

# Negatives per positive triple when the caller does not choose: the
# default of the reference implementation of Sun et al. (2019).
DEFAULT_NEGATIVE_SAMPLING = 128


class KGEBaseModel(ABC, nn.Module):
    """Abstract class for Knowledge Graph Embedding models.

    The numbers of entities and relations come from the dataset's fitted
    ``kg_triple`` processor (``KGLinkPrediction`` with
    ``set_task(..., split=PatientSplit(...))``). Training and evaluation
    follow ``model.train()`` / ``model.eval()``, as the
    :class:`~pyhealth.trainer.Trainer` sets them.

    Args:
        dataset: A sample dataset whose ``triple`` field uses the
            ``kg_triple`` processor, typically the training part returned
            by ``set_task``. A PyHealth 1.x ``SampleKGDataset`` is still
            accepted for one release.
        e_dim: the hidden embedding size for entity, 500 by default.
        r_dim: the hidden embedding size for relation, 500 by default.
        ns: negative sampling technique to use: can be "uniform", "normal" or "adv" (self-adversarial).
        gamma: fixed margin (only need when ns="adv").
        use_subsampling_weight: whether to use subsampling weight (like in word2vec) or not, False by default.
        use_regularization: whether to apply regularization or not, False by default.
        mode: evaluation metric type, one of "binary", "multiclass", or "multilabel", "multiclass" by default
        negative_sampling: Number of negative heads and of negative tails drawn
            per training triple. Defaults to the ``negative_sampling`` of a
            1.x ``SampleKGDataset`` if it has one, else to the value 1.x
            samples carry in ``hyperparameters``, else 128.

    Examples:
        >>> from pyhealth.processors import KGTripleProcessor
        >>> class _Toy:
        ...     input_processors = {
        ...         "triple": KGTripleProcessor(num_entities=2, num_relations=1)
        ...     }
        >>> model = KGEBaseModel(_Toy(), e_dim=4, r_dim=4, ns="uniform")
        >>> model.e_num, model.r_num, tuple(model.E_emb.shape)
        (2, 1, (2, 4))
        >>> model.negative_sampling
        128
    """

    @property
    def device(self):
        """Gets the device of the model."""
        return self._dummy_param.device


    def __init__(
        self,
        dataset: SampleDataset,
        e_dim: int = 500,
        r_dim: int = 500,
        ns: str = "uniform",
        gamma: float | None = None,
        use_subsampling_weight: bool = False,
        use_regularization: str | None = None,
        mode: str = "multiclass",
        negative_sampling: int | None = None,
    ):
        super().__init__()
        processor = getattr(dataset, "input_processors", {}).get("triple")
        if isinstance(processor, KGTripleProcessor):
            self.triple_processor: KGTripleProcessor | None = processor
            self.e_num = processor.num_entities
            self.r_num = processor.num_relations
        else:
            # A PyHealth 1.x SampleKGDataset carries the counts itself.
            warnings.warn(
                "Building a KGE model from a dataset without a kg_triple "
                "processor (e.g. a 1.x SampleKGDataset) is deprecated and will "
                "be removed in the next release; pass the training part of "
                "set_task(KGLinkPrediction(...), split=PatientSplit(...)).",
                DeprecationWarning,
                stacklevel=3,
            )
            self.triple_processor = None
            self.e_num = dataset.entity_num
            self.r_num = dataset.relation_num
        # Whether the value was chosen (argument or 1.x dataset) rather than
        # defaulted: a 1.x batch's own value only overrides a default.
        legacy = getattr(dataset, "task_spec_param", None) or {}
        if negative_sampling is None:
            negative_sampling = legacy.get("negative_sampling")
        self._negative_sampling_chosen = negative_sampling is not None
        if negative_sampling is None:
            negative_sampling = DEFAULT_NEGATIVE_SAMPLING
        if (
            not isinstance(negative_sampling, numbers.Integral)
            or isinstance(negative_sampling, bool)
            or negative_sampling < 1
        ):
            raise ValueError(
                f"negative_sampling must be a positive integer, got {negative_sampling!r}."
            )
        self.negative_sampling = int(negative_sampling)
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

    
    @staticmethod
    def _unpad_ground_truth(ground_truth):
        """Recover the exact, unpadded per-sample entity-id lists.

        ``KGProcessor`` pads ``ground_truth_head``/``ground_truth_tail`` to a
        fixed length with ``pad_token_id`` (0 by default) so that they collate
        into fixed-shape tensors. That padding value is not
        necessarily an invalid entity id, so it must be stripped via the
        accompanying mask before doing any set-membership filtering here;
        otherwise a real entity 0 would be spuriously treated as always
        "known true" (or, symmetrically, padding would be treated as a real
        entity to exclude/replace).

        Args:
            ground_truth: Either a ``{"value": Tensor(B, L), "mask": Tensor(B, L)}``
                pair (collated ``KGProcessor`` output), or already a list of
                raw per-sample entity-id lists (e.g. when a caller bypasses
                the processor and supplies unpadded lists directly).

        Returns:
            List of length ``B``, each entry the unpadded list of entity ids
            for that sample.
        """
        if isinstance(ground_truth, dict):
            value, mask = ground_truth["value"], ground_truth["mask"]
            return [
                value[i][mask[i].bool()].tolist() for i in range(value.size(0))
            ]
        return ground_truth

    def train_neg_sample_gen(self, gt_head, gt_tail, negative_sampling):
        """
        (only run in train batch)
        This function creates negative triples for training (sampling size: negative_sampling)
             with ground truth masked.

        Args:
            gt_head: Either a list of raw (unpadded) entity-id lists, or a
                ``{"value": Tensor(B, L), "mask": Tensor(B, L)}`` pair produced
                by ``KGProcessor``. The padded ``value`` is not usable on its
                own for membership filtering, since ``pad_token_id`` may
                collide with a real entity id (e.g. 0); the ``mask`` recovers
                the exact unpadded list first.
            gt_tail: Same shape as ``gt_head``, for tail entities.
        """
        gt_head = self._unpad_ground_truth(gt_head)
        gt_tail = self._unpad_ground_truth(gt_tail)

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

        Args:
            triples: Batch of ``(head, relation, tail)`` triples.
            gt_head: Either a list of raw (unpadded) entity-id lists, or a
                ``{"value": Tensor(B, L), "mask": Tensor(B, L)}`` pair produced
                by ``KGProcessor``. See ``_unpad_ground_truth`` for why the
                mask matters.
            gt_tail: Same shape as ``gt_head``, for tail entities.
        """
        gt_head = self._unpad_ground_truth(gt_head)
        gt_tail = self._unpad_ground_truth(gt_tail)

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


    def _batch_negative_sampling(self, data) -> int:
        """Negatives per triple for this batch.

        PyHealth 1.x samples may carry ``hyperparameters["negative_sampling"]``,
        which the models used to read. It still applies when the model's
        value was only the default, and is ignored, with a warning, when the
        model's value was chosen.
        """
        hyperparameters = data.get("hyperparameters")
        if not hyperparameters or not isinstance(hyperparameters[0], dict):
            return self.negative_sampling
        batch_value = hyperparameters[0].get("negative_sampling")
        if batch_value is None or batch_value == self.negative_sampling:
            return self.negative_sampling
        if not self._negative_sampling_chosen:
            return int(batch_value)
        warnings.warn(
            f"The samples' negative_sampling={batch_value} is ignored; the "
            f"model's negative_sampling={self.negative_sampling} is used.",
            UserWarning,
            stacklevel=3,
        )
        return self.negative_sampling

    def _training_filters(self, data, positive_sample):
        """The entities kept out of each triple's training negatives.

        With a ``kg_triple`` processor, these are its ``true_head`` /
        ``true_tail`` dicts, fitted on the training triples only, as in the
        reference implementation of Sun et al. (2019). The samples'
        ``ground_truth_*`` lists cover the whole graph and serve filtered
        evaluation; using them here would keep every validation and test
        positive out of the training negatives, so training would depend on
        the held-out triples. A 1.x ``SampleKGDataset`` has no processor and
        keeps its former behaviour.

        Returns:
            Two lists of entity-id lists, for head and tail negatives.
        """
        if self.triple_processor is None:
            return data["ground_truth_head"], data["ground_truth_tail"]
        true_head = self.triple_processor.true_head
        true_tail = self.triple_processor.true_tail
        gt_head, gt_tail = [], []
        for head, relation, tail in positive_sample.tolist():
            gt_head.append(true_head.get((relation, tail), []))
            gt_tail.append(true_tail.get((head, relation), []))
        return gt_head, gt_tail

    def _subsampling_weight(self, data, positive_sample):
        if "subsampling_weight" in data:
            # PyHealth 1.x samples carry their weight.
            return torch.cat([d for d in data["subsampling_weight"]], dim=0)
        if self.triple_processor is None:
            raise ValueError("use_subsampling_weight needs a kg_triple processor.")
        return self.triple_processor.subsampling_weight(positive_sample.cpu())

    def forward(self, **data):

        triples = data["triple"]
        if not isinstance(triples, torch.Tensor):
            triples = torch.stack([torch.as_tensor(d) for d in triples], dim=0)
        positive_sample = triples.long().to(self.device)

        if "train" in data and bool(data["train"][0]) != self.training:
            # PyHealth 1.x samples carry a per-sample train flag, which no
            # longer decides the branch.
            warnings.warn(
                "The samples' 'train' flag is ignored: the model trains or "
                "evaluates according to model.train() / model.eval().",
                UserWarning,
                stacklevel=2,
            )

        if self.training:
            gt_head, gt_tail = self._training_filters(data, positive_sample)
            negative_sample_head, negative_sample_tail = self.train_neg_sample_gen(
                gt_head=gt_head,
                gt_tail=gt_tail,
                negative_sampling=self._batch_negative_sampling(data),
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

            if self.use_subsampling_weight:
                subsampling_weight = self._subsampling_weight(data, positive_sample).to(self.device)
                pos_sample_loss = - (subsampling_weight * pos_score).sum() / subsampling_weight.sum()
                neg_sample_loss = - (subsampling_weight * neg_score).sum() / subsampling_weight.sum()
            else:
                pos_sample_loss = - pos_score.mean()
                neg_sample_loss = - neg_score.mean()

            loss = (pos_sample_loss + neg_sample_loss) / 2

            if self.use_regularization == 'l3':
                loss = loss + self.l3_regularization()
            elif self.use_regularization != None:
                loss = loss + self.regularization()

            return {"loss": loss}

        else: # valid/test
            inputs = self.test_neg_sample_filter_bias_gen(
                    triples=positive_sample.tolist(),
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
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.update_embedding_size(state_dict)
        self.load_state_dict(state_dict)

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


            









