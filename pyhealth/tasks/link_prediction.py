import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, List
from collections import defaultdict

def link_prediction_fn(
    triples: List[Tuple],
    entity_num: int,
    negative_sampling=128
):

    """Process a triple list for the link prediction task

    Link prediction is a task to either 
    Tail Prediction: predict tail entity t given a triple query (h, r, ?), or
    Head Prediction: predict head entity h given a triple query (?, r, t)

    Args:
        triples: a list of triples (indexed) from the knowledge graph
        entity_num: the number of entities contained in the knowledge graph
        negative_sampling: negative sample size, with default value 128
    
    Returns:
        samples: a dict of list, each list corresponds to a dataloader later

        samples['head_train']: [(positive_sample, negative_sample_head, subsampling_weight, 'head')]
        samples['tail_train']: [(positive_sample, negative_sample_tail, subsampling_weight, 'tail')]
        samples['head_test']: [(positive_sample, negative_sample_head, filter_bias_head, 'head'))]
        samples['tail_test']: [(positive_sample, negative_sample_tail, filter_bias_tail, 'tail')]

    """
    samples = defaultdict(list)
    count = count_frequency(triples)
    true_head, true_tail = get_true_head_and_tail(triples)
    triple_set = set(triples)
    gt_head, gt_tail = ground_truth_for_query(triple_set)

    for positive_sample in tqdm(triples):
        head, relation, tail = positive_sample

        # 1. generate sample['head_train'] and sample['tail_train']
        subsampling_weight = torch.sqrt(1 / torch.Tensor([count[(head, relation)] + count[(tail, -relation-1)]]))
        
        ## negative samples for head prediction
        negative_sample_list_head = []
        negative_sample_size_head = 0

        while negative_sample_size_head < negative_sampling:
            negative_sample = np.random.randint(entity_num, size=negative_sampling*2)
            mask = np.in1d(
                negative_sample,
                true_head[(relation, tail)],
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
            negative_sample = np.random.randint(entity_num, size=negative_sampling*2)
            mask = np.in1d(
                negative_sample,
                true_tail[(head, relation)],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            negative_sample_list_tail.append(negative_sample)
            negative_sample_size_tail += negative_sample.size

        positive_sample = torch.LongTensor(positive_sample)
        negative_sample_head = torch.LongTensor(np.concatenate(negative_sample_list_head)[:negative_sampling])
        negative_sample_tail = torch.LongTensor(np.concatenate(negative_sample_list_tail)[:negative_sampling])

        samples['head_train'].append((positive_sample, negative_sample_head, subsampling_weight, 'head'))
        samples['tail_train'].append((positive_sample, negative_sample_tail, subsampling_weight, 'tail'))

        # 2. generate sample['head_test'] and sample['tail_test']
        gt_h = gt_head[(relation, tail)]
        gt_h.remove(head)
        gt_t = gt_tail[(head, relation)]
        gt_t.remove(tail)

        positive_sample = torch.LongTensor(positive_sample)

        negative_sample_head = np.arange(0, entity_num)
        negative_sample_head[gt_h] = head
        negative_sample_head = torch.LongTensor(negative_sample_head)

        negative_sample_tail = np.arange(0, entity_num)
        negative_sample_tail[gt_t] = tail
        negative_sample_tail = torch.LongTensor(negative_sample_tail)

        filter_bias_head = np.zeros(entity_num)
        filter_bias_head[gt_h] = -1
        filter_bias_head = torch.LongTensor(filter_bias_head)

        filter_bias_tail = np.zeros(entity_num)
        filter_bias_tail[gt_t] = -1
        filter_bias_tail = torch.LongTensor(filter_bias_tail)

        samples['head_test'].append((positive_sample, negative_sample_head, filter_bias_head, 'head'))
        samples['tail_test'].append((positive_sample, negative_sample_tail, filter_bias_tail, 'tail'))
        
    return samples


def ground_truth_for_query(triple_set):
    gt_head = defaultdict(list)
    gt_tail = defaultdict(list)

    for triple in triple_set:
        head, relation, tail = triple
        gt_head[(relation, tail)].append(head)
        gt_tail[(head, relation)].append(tail)
    
    return gt_head, gt_tail


def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count


def get_true_head_and_tail(triples):
    '''
    Build a dictionary of true triples that will
    be used to filter these true triples for negative sampling
    '''
    
    true_head = {}
    true_tail = {}

    for head, relation, tail in triples:
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

    return true_head, true_tail
