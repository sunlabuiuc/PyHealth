import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.utils import multi_label_metric


class DrugRecommendationEvaluator(nn.Module):
    def __init__(
            self,
            model,
    ):
        super().__init__()
        self.model = model

    def evaluate(
            self,
            dataloader,
            device,
    ):
        self.model.eval()
        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        with torch.no_grad():
            for step, (X, y) in enumerate(dataloader):
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
                X = X.to(device)
                y = y.to(device)
                for i in range(len(X)):
                    target_output = self.model.forward_current_admission(X[:i + 1])
                    y_gt.append(y[i].cpu().numpy())

                    # prediction prob
                    target_output = F.sigmoid(target_output).cpu().numpy()[0]
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

                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                    multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)
        print(
            '\nJaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
            ))
