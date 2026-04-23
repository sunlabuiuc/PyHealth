"""
PyHealth task for extracting features with STFT and Frequency Bands using the Temple University Hospital (TUH) EEG Seizure Corpus (TUSZ) dataset V2.0.5.

Dataset link:
    https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml

Dataset paper:
    Vinit Shah, Eva von Weltin, Silvia Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” arXiv preprint arXiv:1801.08085, 2018. Available: https://arxiv.org/abs/1801.08085

Dataset paper link:
    https://arxiv.org/abs/1801.08085

Author:
    Fernando Kenji Sakabe (fks@illinois.edu), 
    Jesica Hirsch (jesicah2@illinois.edu), 
    Jung-Jung Hsieh (jhsieh8@illinois.edu)
"""
import logging
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score
import torch
from numpy import ndarray

logger = logging.getLogger(__name__)

def eeg_margin_fn(
    y_true_multi: np.ndarray,
    y_pred_multi: np.ndarray,
    tnr_for_margintest: list[float],
    probability_list: list[float],
    final_target_list: list[int],
    margin_list: list[int],
) -> tuple[ndarray, float, float, float, float]:
    """Computes metrics for performance of seizure detection within an onset/offset margin.

    Args:
        y_true_multi: Ground truth. A list of true labels.
        y_pred_multi: Ranked results. A list of predicted labels.
        tnr_for_margintest: The threshold for true negative rate.
        probability_list: A list of predicted probabilities used to determine y_pred_multi.
        final_target_list: A list of true probabilities used to determine y_true_multi.
        margin_list: A list of onset/offset margin in seconds.

    Returns:
        result: A list of roc_auc_score, average_precision_score, and f1 score.
        tpr: True positive rate.
        fnr: False negative rate.
        tnr: True negative rate.
        fpr: False positive rate.

    Examples:
        >>> y_true_multi = [0, 0]
        >>> y_pred_multi = [0, 0]
        >>> tnr_for_margintest = [0.7, 0.85]
        >>> probability_list = [0.89, 0.11]
        >>> final_target_list = [0, 0]
        >>> margin_list = [3, 5]
        >>> k_values = [1, 2]
        >>> eeg_margin_fn(y_true_multi, y_pred_multi, tnr_for_margintest, probability_list, final_target_list, MARGIN_LIST)
        Best threshold is:  0.04016105
        Margin: 3, Threshold: 0.04016105, TPR: 0.3331, TNR: 0.9111
        rise_accuarcy:0.0, fall_accuracy:0.0
    """
    
    y_true_multi = np.concatenate(y_true_multi, 0)
    y_pred_multi = np.concatenate(y_pred_multi, 0)

    auc = roc_auc_score(y_true_multi[:,1], y_pred_multi[:,1])
    apr = average_precision_score(y_true_multi[:,1], y_pred_multi[:,1])
    y_true_multi_array = np.argmax(y_true_multi, axis=1)

    f1 = 0
    for i in range(1, 200):
        threshold = float(i) / 200
        temp_output = np.array(y_pred_multi[:,1])
        temp_output[temp_output>=threshold] = 1
        temp_output[temp_output<threshold] = 0
        temp_score = f1_score(y_true_multi_array, temp_output, average="binary")
        if temp_score > f1:
            f1 = temp_score
        
    result = np.round(np.array([auc, apr, f1]), decimals=4)
    fpr, tpr, thresholds = roc_curve(y_true_multi_array, y_pred_multi[:,1], pos_label=1)
    fnr = 1 - tpr 
    tnr = 1 - fpr
    best_threshold = np.argmax(tpr + tnr)
    logger.info(f"Best threshold is: {thresholds[best_threshold]}")

    tnr_list = list(tnr)

    picked_tnrs = []
    picked_tprs = []
    thresholds_margintest = []
    for tnr_one in tnr_for_margintest:
        picked_tnr = list([0 if x< tnr_one else x for x in tnr_list])
        picked_tnr_threshold = np.argmax(tpr + picked_tnr)        
        thresholds_margintest.append(thresholds[picked_tnr_threshold])
        picked_tnrs.append(np.round(tnr[picked_tnr_threshold], decimals=4))
        picked_tprs.append(np.round(tpr[picked_tnr_threshold], decimals=4))
    
    target_stack = torch.stack(final_target_list)
    for margin in margin_list:
        for threshold_idx, threshold in enumerate(thresholds_margintest):
            pred_stack = torch.stack(probability_list)
            pred_stack = (pred_stack > threshold).int()
            rise_true, rise_pred_correct, fall_true, fall_pred_correct = binary_detector_evaluator(pred_stack, target_stack, margin)
            logger.info("Margin: {}, Threshold: {}, TPR: {}, TNR: {}".format(str(margin), str(threshold), str(picked_tprs[threshold_idx]), str(picked_tnrs[threshold_idx])))
            logger.info("rise_accuarcy:{}, fall_accuracy:{}".format(str(np.round((rise_pred_correct/float(rise_true)), decimals=4)), str(np.round((fall_pred_correct/float(fall_true)), decimals=4))))

    return (
        result, 
        np.round(tpr[best_threshold], decimals=4), 
        np.round(fnr[best_threshold], decimals=4), 
        np.round(tnr[best_threshold], decimals=4), 
        np.round(fpr[best_threshold], decimals=4)
    )


def binary_detector_evaluator(
    pred_stack: torch.Tensor,
    target_stack: torch.Tensor,
    margin: int,
) -> tuple[int, int, int, int]:
    """Returns the count of true and predicted values for onset and offset margins."""
    rise_true, rise_pred_correct, fall_true, fall_pred_correct = 0, 0, 0, 0
    target_rotated = torch.cat([target_stack[0].unsqueeze(0), target_stack[:-1]], dim=0)
    pred_rotated = torch.cat([pred_stack[0].unsqueeze(0), pred_stack[:-1]], dim=0)

    # -1 is at where label goes 0 to 1 (at point of 1)
    # 1 is at where label goes 1 to 0 (at point of 0)
    target_change = torch.subtract(target_rotated, target_stack) 
    pred_change = torch.subtract(pred_rotated, pred_stack) 

    # fall = target_change == 1
    # rise = target_change == -1
    
    for idx, sample in enumerate(target_change.permute(1,0)):
        fall_index_list = (sample == 1).nonzero(as_tuple=True)[0]
        rise_index_list = (sample == -1).nonzero(as_tuple=True)[0]

        for fall_index in fall_index_list:
            start_margin_index = fall_index - margin
            end_margin_index = fall_index + margin
            if start_margin_index < 0:
                start_margin_index = 0
            if end_margin_index > len(sample):
                end_margin_index = len(sample)
            if 1 in pred_change[start_margin_index:end_margin_index+1]:
                fall_pred_correct += 1
            fall_true += 1
        for rise_index in rise_index_list:
            start_margin_index = rise_index - margin
            end_margin_index = rise_index + margin
            if start_margin_index < 0:
                start_margin_index = 0
            if end_margin_index > len(sample):
                end_margin_index = len(sample)
            if -1 in pred_change[start_margin_index:end_margin_index+1]:
                rise_pred_correct += 1
            rise_true += 1
    
    return rise_true, rise_pred_correct, fall_true, fall_pred_correct



if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
