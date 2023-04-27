from typing import Dict, List, Optional

from collections import OrderedDict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def sequence_metrics_fn(
    y_true: List[Dict[int,str]],
    y_generated: List[Dict[int,str]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute metrics relevant for evaluating sequences.

    User can specify which metrics to compute by passing a list of metric names
    The accepted metric names are:
        - Bleu_{n_grams}: BiLingual Evaluation Understudy. 
                          Allowed n_grams = [1,2,3,4]
        - METEOR: Metric for Evaluation of Translation with Explicit ORdering
        - ROUGE: Recall-Oriented Understudy for Gisting Evaluation
        - CIDEr: Consensus-based Image Description Evaluation
    
    All metrics compute a score for comparing a candidate text to one or more 
    reference text.
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE"),
        (Cider(), "CIDER"),
    ]
    
    allowed_metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4",
                       "METEOR","ROUGE","CIDER"]
    if metrics:
        for metric in metrics:
            if metric not in allowed_metrics:
                raise ValueError(f"Unknown metric for evaluation: {metric}")
    else:
        metrics = allowed_metrics
         
    output = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(y_true, y_generated)
        if type(score) == list:
            for m, s in zip(method, score):
                if m in metrics:
                    output[m] = s
        else:
            if method in metrics:
                output[method] = score

    return output