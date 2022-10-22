import numpy as np
import torch
from tqdm import tqdm
from pyhealth.metrics import *


def evaluate(model, dataloader, device="cpu", disable_bar=False):
    """Evaluate model on dataloader.
    INPUT:
        - model: model to evaluate
        - dataloader: dataloader to evaluate on
        - device: device to run on
        - isMLModel: is model ML (True) or DL (False)
    OUTPUT:
        - y_gt_all: ground truth labels
        - y_prob_all: predicted probabilities
    """
    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    for data in tqdm(dataloader, desc="Evaluation", disable=disable_bar):
        if model.__class__.__name__ != "ClassicML":
            model.eval()
            with torch.no_grad():
                output = model(**data, device=device)
                y_true = output["y_true"].cpu().numpy()
                y_prob = output["y_prob"].cpu().numpy()
                y_pred = output["y_pred"].cpu().numpy()
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
                y_pred_all.append(y_pred)
        else:
            output = model(**data)
            y_true = output["y_true"]
            y_prob = output["y_prob"]
            y_pred = output["y_pred"]
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            y_pred_all.append(y_pred)
    y_gt_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    return y_gt_all, y_prob_all, y_pred_all
