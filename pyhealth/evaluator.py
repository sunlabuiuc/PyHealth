import numpy as np
import torch
from tqdm import tqdm
from pyhealth.metrics import *


def evaluate(model, dataloader, device="cpu", isMLModel=False):
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
    if not isMLModel:
        for data in tqdm(dataloader, desc="Evaluation"):
            model.eval()
            with torch.no_grad():
                output = model(**data, device=device, training=False)
                y_true = output["y_true"].cpu()
                y_prob = output["y_prob"].cpu()
                y_pred = output["y_pred"].cpu()
                y_true_all.append(y_true)
                y_prob_all.append(y_prob)
                y_pred_all.append(y_pred)
        y_gt_all = torch.cat(y_true_all).numpy()
        y_prob_all = torch.cat(y_prob_all).numpy()
        y_pred_all = torch.cat(y_pred_all).numpy()
        return y_gt_all, y_prob_all, y_pred_all

    else:
        return model.eval(dataloader)

