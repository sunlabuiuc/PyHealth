import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np
from tqdm import tqdm

# for drug recommendation
def multi_label_metric(pre, gt, threshold=0.4):
    """
    pre is a float matrix in [0, 1]
    gt is a binary matrix
    """

    def jaccard(pre, gt):
        score = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(pre[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            union = set(predicted) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def precision_auc(pre, gt):
        all_micro = []
        for b in range(gt.shape[0]):
            all_micro.append(average_precision_score(gt[b], pre[b], average="macro"))
        return np.mean(all_micro)

    def prc_recall(pre, gt):
        score_prc = []
        score_recall = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(pre[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            prc_score = 0 if len(predicted) == 0 else len(inter) / len(predicted)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score_prc.append(prc_score)
            score_recall.append(recall_score)
        return score_prc, score_recall

    def average_f1(prc, recall):
        score = []
        for idx in range(len(prc)):
            if prc[idx] + recall[idx] == 0:
                score.append(0)
            else:
                score.append(2 * prc[idx] * recall[idx] / (prc[idx] + recall[idx]))
        return np.mean(score)

    ja = jaccard(pre, gt)
    prauc = precision_auc(pre, gt)
    prc_ls, recall_ls = prc_recall(pre, gt)
    f1 = average_f1(prc_ls, recall_ls)

    return {"jaccard": ja, "prauc": prauc, "f1": f1}


def evaluate_multilabel(model, dataloader, device):
    predicted = []
    gt = []
    loss_all = []
    for data in tqdm(dataloader, desc="Evaluation"):
        model.eval()
        with torch.no_grad():
            output = model(**data, device=device)
            loss = output["loss"].cpu()
            y_true = output["y_true"].cpu()
            y_prob = output["y_prob"].cpu()
            predicted.append(torch.sigmoid(y_prob).cpu().numpy())
            gt.append(y_true.numpy())
            loss_all.append(loss)

    loss_avg = torch.tensor(loss_all).numpy().mean()
    predicted = np.concatenate(predicted, axis=0)
    gt = np.concatenate(gt, axis=0)
    all_metric = multi_label_metric(predicted, gt)
    return {"loss": loss_avg, **all_metric}
