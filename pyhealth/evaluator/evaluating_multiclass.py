import numpy as np
import torch
from tqdm import tqdm


def metrics_multiclass(y_gt, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    acc = np.mean(y_pred == y_gt)
    return {"acc": acc}


def evaluate_multiclass(model, dataloader, device):
    loss_all = []
    y_true_all = []
    y_prob_all = []
    for data in tqdm(dataloader, desc='Evaluation'):
        model.eval()
        with torch.no_grad():
            output = model(**data, device=device, training=False)
            loss = output['loss'].cpu()
            y_true = output['y_true'].cpu()
            y_prob = output['y_prob'].cpu()
            loss_all.append(loss)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
    loss_avg = torch.tensor(loss_all).numpy().mean()
    y_gt_all = torch.cat(y_true_all).numpy()
    y_prob_all = torch.cat(y_prob_all).numpy()
    all_metric = metrics_multiclass(y_gt_all, y_prob_all)
    return {"loss": loss_avg, **all_metric}
