"""
LABEL: Least ambiguous set-valued classifiers with bounded error levels.

From:
    Sadinle, Mauricio, Jing Lei, and Larry Wasserman. 
    "Least ambiguous set-valued classifiers with bounded error levels." 
    Journal of the American Statistical Association 114, no. 525 (2019): 223-234.

"""

from typing import Dict, Union

import ipdb
import numpy as np
import torch

from pyhealth.models import BaseModel
from pyhealth.uq.base_classes import SetPredictor
from pyhealth.uq.utils import prepare_numpy_dataset

__all__ = ['LABEL']

def _query_quantile(scores, alpha):
    scores = np.sort(scores)
    N = len(scores)
    loc = int(np.floor(alpha * (N+1))) - 1
    return -np.inf if loc == -1 else scores[loc]

class LABEL(SetPredictor):
    def __init__(self, model:BaseModel, alpha: Union[float, np.ndarray], 
                 debug=False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        if model.mode != 'multiclass':
            raise NotImplementedError()
        self.mode = self.model.mode # multiclass
        for param in model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = model.device
        self.debug = debug

        if not isinstance(alpha, float):
            alpha = np.asarray(alpha)
        self.alpha = alpha

    def calibrate(self, cal_dataset):
        cal_dataset = prepare_numpy_dataset(self.model, cal_dataset, ['y_prob', 'y_true'], debug=self.debug)
        y_prob = cal_dataset['y_prob']
        y_true = cal_dataset['y_true']

        N, K = cal_dataset['y_prob'].shape
        if isinstance(self.alpha, float):
            t = _query_quantile(y_prob[np.arange(N), y_true], self.alpha)
        else:
            t = [_query_quantile(y_prob[y_true==k,k],self.alpha[k]) for k in range(K)]
        self.t = torch.tensor(t, device=self.device)
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        pred = self.model(**kwargs)
        pred['y_pred'] = pred['y_prob'] > self.t
        ipdb.set_trace()
        return pred