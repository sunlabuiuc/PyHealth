"""
Author(s): Nandan Tadi (ntadi2), Ayan Deka (ayand2)
Title: Reproducing GAN Image Diagnostic Utility
Task: Evaluate if synthetic X-rays support accurate diagnosis.
"""

from sklearn.metrics import classification_report
from typing import List, Dict

def evaluate_predictions(true_labels: List[str], predicted_labels: List[str]) -> Dict[str, float]:
    """Evaluates classification metrics.

    Args:
        true_labels (List[str]): True diagnosis labels.
        predicted_labels (List[str]): Labels assigned by participants.

    Returns:
        Dict[str, float]: Evaluation metrics from sklearn.
    """
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    return report
