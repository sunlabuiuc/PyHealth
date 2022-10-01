import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


# TODO: adapt this to pytorch lightning framework


def multi_label_metric(y_gt, y_pred, y_prob):
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


class DrugRecEvaluator(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model

    def evaluate(self, dataloader):
        self.model.eval()
        if self.model.device.type == "cpu":
            self.model.to("cuda")
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        with torch.no_grad():
            for step, test_batch in enumerate(dataloader):
                conditions, procedures, drugs = test_batch.values()
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
                if self.model.__class__.__name__ == "MICRON":
                    for idx in range(len(conditions)):
                        if idx == 0:
                            representation_base = self.model.embedding(
                                conditions[idx : idx + 1], procedures[idx : idx + 1]
                            )
                            drugs_index = drugs[idx : idx + 1]
                            drugs_multihot_old = torch.zeros(1, self.model.voc_size[2])
                            drugs_multihot_old[0][drugs_index[0]] = 1
                            continue

                        drugs_index = drugs[idx : idx + 1]
                        drugs_multihot = torch.zeros(1, self.model.voc_size[2])
                        drugs_multihot[0][drugs_index[0]] = 1
                        y_gt.append(drugs_multihot[0].numpy())

                        _, _, residual, _ = self.model(
                            conditions[idx - 1 : idx + 1], procedures[idx - 1 : idx + 1]
                        )
                        # prediction prod
                        representation_base += residual
                        y_pred_tmp = torch.sigmoid(representation_base).cpu().numpy()[0]
                        y_pred_prob.append(y_pred_tmp)

                        # prediction med set
                        drugs_multihot_old[0][y_pred_tmp >= 0.8] = 1
                        drugs_multihot_old[0][y_pred_tmp < 0.2] = 0
                        y_pred.append(drugs_multihot_old[0].numpy())

                        # prediction label
                        y_pred_label_tmp = np.where(drugs_multihot_old == 1)[0]
                        y_pred_label.append(sorted(y_pred_label_tmp))
                        visit_cnt += 1
                        med_cnt += len(y_pred_label_tmp)
                else:
                    for i in range(len(conditions)):
                        if self.model.__class__.__name__ == "GAMENet":
                            target_output = self.model(
                                conditions[: i + 1], procedures[: i + 1], drugs[:i]
                            )
                        else:
                            target_output = self.model(
                                conditions[: i + 1], procedures[: i + 1]
                            )
                        drugs_index = drugs[i : i + 1]
                        drugs_multihot = torch.zeros(1, self.model.voc_size[2])
                        drugs_multihot[0][drugs_index[0]] = 1
                        y_gt.append(drugs_multihot[0].numpy())

                        # prediction prob
                        target_output = torch.sigmoid(target_output).cpu().numpy()[0]
                        y_pred_prob.append(target_output)

                        # prediction med set
                        y_pred_tmp = target_output.copy()
                        y_pred_tmp[y_pred_tmp >= 0.4] = 1
                        y_pred_tmp[y_pred_tmp < 0.4] = 0
                        y_pred.append(y_pred_tmp)

                        # prediction label
                        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
                        y_pred_label.append(y_pred_label_tmp)
                        med_cnt += len(y_pred_label_tmp)
                        visit_cnt += 1

                (
                    adm_ja,
                    adm_prauc,
                    adm_avg_p,
                    adm_avg_r,
                    adm_avg_f1,
                ) = multi_label_metric(
                    np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
                )

                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)
        print(
            "\nJaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n".format(
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )
        )
