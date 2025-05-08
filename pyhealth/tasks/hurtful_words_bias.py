# =============================================================================
# Ritul Soni (rsoni27)  
# “Hurtful Words” Bias Quantification  
# Paper: Hurtful Words in Clinical Contextualized Embeddings  
# Link: https://arxiv.org/abs/2012.00355  
#
# Implements:
#  - log probability bias score per [Zhang et al., 2020]
#  - precision gap as an additional fairness metric
# =============================================================================

from typing import List, Tuple, Dict
import numpy as np
from pyhealth.tasks.base import BaseTask

class HurtfulWordsBiasTask(BaseTask):
    """Compute log-probability bias and precision-gap on ClinicalBERT outputs.

    Will be called in `dataset.set_task(hurtful_words_bias_fn)`.
    """

    def __init__(self, positive_group: str = "female", negative_group: str = "male"):
        """
        Args:
            positive_group (str): demographic label for privileged group.
            negative_group (str): demographic label for unprivileged group.
        """
        super().__init__()
        self.positive = positive_group
        self.negative = negative_group

    def get_ground_truth(self, patient_record: Dict) -> str:
        """Extract demographic label from the record.

        Args:
            patient_record: a dict containing at least 'gender'.

        Returns:
            str: either self.positive or self.negative.
        """
        gender = patient_record["gender"].lower()
        return self.positive if gender == self.positive else self.negative

    def get_prediction(self, model, text: str) -> float:
        """Mask target word in `text`, compute its log-probability under `model`.

        Args:
            model: a HuggingFace MaskedLM
            text (str): one clinical note with a single [MASK]

        Returns:
            float: log P(target_token | context)
        """
        # your helper logic here...
        return model.get_log_prob(text)

    def evaluate(self,
                 data: List[Dict],
                 model,
                 metrics: List[str] = ["log_bias", "precision_gap"]
                ) -> Dict[str, float]:
        """
        Compute requested metrics over the test split.

        Args:
            data (List[Dict]): list of records with 'text' and 'gender'
            model: a calibrated or uncalibrated ClinicalBERT wrapper
            metrics (List[str]): which metrics to compute

        Returns:
            Dict[str, float]: metric_name → value
        """
        # collect scores and labels
        scores, labels = [], []
        for rec in data:
            scores.append(self.get_prediction(model, rec["text"]))
            labels.append(self.get_ground_truth(rec))
        scores = np.array(scores)
        labels = np.array(labels)

        results = {}
        if "log_bias" in metrics:
            priv = scores[labels == self.positive].mean()
            unpriv = scores[labels == self.negative].mean()
            results["log_bias"] = priv - unpriv

        if "precision_gap" in metrics:
            # threshold at median score
            thresh = np.median(scores)
            preds = scores >= thresh
            def precision(y_true, y_pred, grp):
                mask = (labels == grp)
                tp = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
                fp = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1))
                return tp / (tp + fp + 1e-12)
            # map gender to binary y_true: privileged=1, unprivileged=0
            y_true = (labels == self.positive).astype(int)
            results["precision_gap"] = precision(y_true, preds, self.positive) - \
                                      precision(y_true, preds, self.negative)

        return results
