import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

VALID_MODE = [
    "binary",
    "multiclass",
    "multilabel"
]

class base(pl.LightningModule):
    def __init__(self, voc_size, params, model, emb_dim=64, mode: str = "binary"):
        super(base, self).__init__()

        self.params = params
        self.voc_size = voc_size

        self.emb_dim = emb_dim
        self.output_len = voc_size[2]

        self.condition_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], self.emb_dim, padding_idx=0),
            nn.Dropout(0.5)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(voc_size[1], self.emb_dim, padding_idx=0),
            nn.Dropout(0.5)
        )

        self.output = nn.Linear(emb_dim, self.output_len)

        if (model == "LR"):         # Logistic Regression
            self.output = nn.Linear(emb_dim, self.output_len)

        if mode not in VALID_MODE:
            raise ValueError(f"mode must be one of {VALID_MODE}")
        self.mode = mode
        if mode in ["binary", "multilabel"]:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif mode in ["multiclass"]:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{mode} is not supported")

    def forward(self, conditions, procedures):
        conditions = conditions.cuda()
        procedures = procedures.cuda()
        conditions_emb = self.condition_embedding(conditions).sum(dim=1)
        procedures_emb = self.procedure_embedding(procedures).sum(dim=1)
        visit_emb = conditions_emb + procedures_emb
        out = self.output(visit_emb)
        if self.model == 'LR':
            out = torch.sigmoid(self.output(visit_emb))

        return out

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = train_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[:i + 1], procedures[:i + 1])
            drugs_index = drugs[i: i + 1].cuda()
            drugs_multihot = torch.zeros(1, self.output_len).cuda()
            drugs_multihot[0][drugs_index[0]] = 1
            loss += self.loss_fn(output_logits, drugs_multihot)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = val_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[:i + 1], procedures[:i + 1])
            drugs_index = drugs[i: i + 1].cuda()
            drugs_multihot = torch.zeros(1, self.output_len).cuda()
            drugs_multihot[0][drugs_index[0]] = 1
            loss += self.loss_fn(output_logits, drugs_multihot)
        # self.log('val_loss', loss)

    def summary(self, output_path, test_dataloaders, ckpt_path):
        # load the best model
        self.model = torch.load(ckpt_path)
        self.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        smm_record = []

        with torch.no_grad():
            for step, (X, y) in enumerate(test_dataloaders):
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

                for i in range(len(X)):
                    target_output, _ = self.forward(X[:i + 1])
                    y_gt.append(y[i].cpu().numpy())

                    # prediction prob
                    target_output = F.sigmoid(target_output).cpu().numpy()[0]
                    y_pred_prob.append(target_output)
                    self.pat_info_test[step][i].append(target_output)

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

                smm_record.append(y_pred_label)
                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                    multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))

                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)

        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)
        print('--- Test Summary ---')
        print(
            'DDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
                ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1),
                med_cnt / visit_cnt
            ))

        # self.prepare_output(output_path)

    def prepare_output(self, output_path):
        """
        write self.pat_info_test to json format:
        {
            patient_id_1: {
                visit_id_1: {
                    "diagnoses": [xxx],
                    "procedures": [xxx],
                    "real_prescription": [xxx],
                    "predicted_prescription": [xxx],
                    "prediction_logits": {
                        "ATC3-1": xxx,
                        "ATC3-2": xxx,
                        ...
                    }
                },
                visit_id_2: {
                        ...
                    }
                },
                ...
            },
            patient_id_2: {
                ...
            },
            ...
        }
        """
        nested_dict = {}
        for cur_pat in self.pat_info_test:
            for cur_visit in cur_pat:
                pat_id = cur_visit[3]
                visit_id = cur_visit[4]
                diag = self.maps['diag'].decodes(cur_visit[0])
                if -1 in diag: diag.remove(-1)
                prod = self.maps['prod'].decodes(cur_visit[1])
                if -1 in prod: prod.remove(-1)
                gt_med = self.maps['med'].decodes(cur_visit[2])
                if -1 in gt_med: gt_med.remove(-1)
                pre_logits = cur_visit[5]
                pre_med = np.where(pre_logits >= 0.5)[0]
                if pat_id not in nested_dict:
                    nested_dict[pat_id] = {}
                nested_dict[pat_id][visit_id] = {
                    "diagnoses": diag,
                    "procedures": prod,
                    "real_prescription": gt_med,
                    "predicted_prescription": self.maps['med'].decodes(pre_med),
                    "prediction_logits": {
                        atc3: str(np.round(logit, 4)) for atc3, logit in
                        zip(self.maps['med'].code_to_idx.keys(), pre_logits)
                    }
                }

        with open(output_path, "w") as outfile:
            json.dump(nested_dict, outfile)

if __name__ == '__main__':
    from pyhealth.datasets.mimic3 import MIMIC3BaseDataset
    from pyhealth.data.dataset import DrugRecommendationDataset
    from torch.utils.data import DataLoader

    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")
    task_taskset = DrugRecommendationDataset(base_dataset)
    data_loader = DataLoader(task_taskset, batch_size=1, collate_fn=lambda x: x[0])
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    model = BaseModel(task_taskset)
    print(model.training_step(batch, 0))