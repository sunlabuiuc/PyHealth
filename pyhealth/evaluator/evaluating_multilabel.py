import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import numpy as np

# for drug recommendation
def multi_label_metric(pre, gt, threshold=0.4):
    """calculate the metrics for multi-label classification
    INPUT
        - pre: a float matrix in [0, 1]
        - gt: a binary matrix
    """

    def jaccard(pre, gt):
        score = []
        for i in range(gt.shape[0]):
            target = np.where(gt[i] == 1)[0]
            predicted = np.where(pre[i] >= threshold)[0]
            inter = set(predicted) & set(target)
            union = set(predicted) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def precision_auc(pre, gt):
        all_micro = []
        for i in range(gt.shape[0]):
            # Calculate metrics for each label, and find their unweighted mean
            all_micro.append(average_precision_score(gt[i], pre[i], average="macro"))
        return np.mean(all_micro)

    def prc_recall(pre, gt):
        score_prc = []
        score_recall = []
        for i in range(gt.shape[0]):
            target = np.where(gt[i] == 1)[0]
            predicted = np.where(pre[i] >= threshold)[0]
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


def ddi_rate_score(predicted, ddi_matrix, threshold=0.4):
    # ddi rate
    all_cnt = 0
    dd_cnt = 0
    for visit in predicted:
        cur_med = np.where(visit >= threshold)[0]
        for i, med_i in enumerate(cur_med):
            for j, med_j in enumerate(cur_med):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def evaluate_multilabel(model, dataloader, device="cpu"):
    predicted = []
    gt = []
    loss_all = []

    if str(model.__class__) != "<class 'pyhealth.models.MLModel.MLModel'>":
        model.to(device)
    for data in dataloader:
        if str(model.__class__) != "<class 'pyhealth.models.MLModel.MLModel'>":
            model.eval()
        with torch.no_grad():
            output = model(**data, device=device)
            try:
                loss = output["loss"].cpu()
                y_true = output["y_true"].cpu()
                y_prob = output["y_prob"].cpu()
                predicted.append(y_prob.cpu().numpy())
                gt.append(y_true.numpy())
            except:
                loss = output["loss"]
                y_true = output["y_true"]
                y_prob = output["y_prob"]
                predicted.append(y_prob)
                gt.append(y_true)
            loss_all.append(loss)

    loss_avg = torch.tensor(loss_all).numpy().mean()
    predicted = np.concatenate(predicted, axis=0)
    gt = np.concatenate(gt, axis=0)
    all_metric = multi_label_metric(predicted, gt)
    if not hasattr(dataloader.dataset.dataset, "ddi_adj"):
        ddi_adj = dataloader.dataset.dataset.get_ddi_matrix()
    else:
        ddi_adj = dataloader.dataset.dataset.ddi_adj
    ddi_rate = ddi_rate_score(predicted, ddi_adj)
    return {"loss": loss_avg, "ddi": ddi_rate, **all_metric}
