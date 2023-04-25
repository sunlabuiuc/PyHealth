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
    """Compute metrics relevant for evaluating sequences
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    output = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(y_true, y_generated)
        if type(score) == list:
            for m, s in zip(method, score):
                output[m] = s
        else:
            output[method] = score

    return output