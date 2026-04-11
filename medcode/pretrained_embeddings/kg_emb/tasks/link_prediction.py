import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple, List
from collections import defaultdict

def link_prediction_fn(
    triples: List[Tuple]
):

    """Process a triple list for the link prediction task

    Link prediction is a task to either 
    Tail Prediction: predict tail entity t given a triple query (h, r, ?), or
    Head Prediction: predict head entity h given a triple query (?, r, t)

    Args:
        triples: a list of triples (indexed) from the knowledge graph
    
    Returns:
        samples: a list of samples
    """
    samples = []
    count = count_frequency(triples)
    triple_set = set(triples)
    gt_head, gt_tail = ground_truth_for_query(triple_set)

    for positive_sample in tqdm(triples):
        head, relation, tail = positive_sample
        subsampling_weight = torch.sqrt(1 / torch.Tensor([count[(head, relation)] + count[(tail, -relation-1)]]))
        gt_h = gt_head[(relation, tail)]
        gt_t = gt_tail[(head, relation)]

        tmp = {
            "triple": positive_sample,
            "ground_truth_head": gt_h,
            "ground_truth_tail": gt_t,
            "subsampling_weight": subsampling_weight
        }

        samples.append(tmp)
        
    return samples


def ground_truth_for_query(triple_set):
    """
    Search ground truth of either query (h, r, ?) or (?, r, t) in the dataset
    """
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

