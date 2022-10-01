from lib2to3.pgen2 import token
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetainLayer(nn.Module):
    def __init__(self, voc_size, emb_dim=64, device=torch.device("cpu:0")):
        super(RetainLayer, self).__init__()
        self.device = device

        self.embedding = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in voc_size])

        self.dropout = nn.Dropout(p=0.5)
        self.alpha_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.alpha_li = nn.Linear(emb_dim, 1)
        self.beta_li = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, padding_mask=None):

        visit_emb_ls = []
        for idx, input in enumerate(x):
            cur_emb = self.embedding[idx](input)  # (batch, visit, code_len, dim)
            visit_emb_ls.append(torch.sum(cur_emb, dim=2))  # (batch, visit, dim)
        visit_emb = torch.stack(visit_emb_ls, dim=2).mean(2)  # (batch, visit, dim)

        g, _ = self.alpha_gru(visit_emb)  # (batch, visit, dim)
        h, _ = self.beta_gru(visit_emb)  # (batch, visit, dim)

        # to mask out the visit
        attn_g = torch.softmax(self.alpha_li(g), dim=1)  # (batch, visit, 1)
        attn_h = torch.tanh(self.beta_li(h))  # (batch, visit, emb)

        c = attn_g * attn_h * visit_emb  # (batch, visit, emb)
        c = torch.sum(c, dim=1)  # (batch, emb)

        return c


class RetainDrugRec(nn.Module):
    def __init__(self, voc_size, tokenizers, emb_dim=64, device=torch.device("cpu:0")):
        super(RetainDrugRec, self).__init__()
        self.device = device
        self.retain = RetainLayer(voc_size, emb_dim, device)

        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]
        self.drug_fc = nn.Linear(emb_dim, self.drug_tokenizer.get_vocabulary_size())

    def forward(
        self, conditions, procedures, drugs, padding_mask=None, device=None, **kwargs
    ):
        diagT = self.condition_tokenizer.batch_tokenize(conditions).to(device)
        procT = self.procedure_tokenizer.batch_tokenize(procedures).to(device)
        x = [diagT, procT]
        c = self.retain(x, padding_mask)
        logits = self.drug_fc(c)
        y_prob = torch.sigmoid(logits)

        # target
        y = torch.zeros(diagT.shape[0], self.drug_tokenizer.get_vocabulary_size())
        for idx, sample in enumerate(drugs):
            y[idx, self.drug_tokenizer(sample[-1:])[0]] = 1

        # loss
        loss = F.binary_cross_entropy_with_logits(y_prob, y.to(device))

        return {"loss": loss, "y_prob": y_prob, "y_true": y}


class RETAIN(pl.LightningModule):
    def __init__(self, voc_size, params, emb_dim=64):
        super(RETAIN, self).__init__()

        # ddi_adj = dataset.ddi_adj
        self.params = params
        self.voc_size = voc_size

        self.emb_dim = emb_dim
        self.output_len = voc_size[2]

        self.condition_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], self.emb_dim, padding_idx=0), nn.Dropout(0.5)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(voc_size[1], self.emb_dim, padding_idx=0), nn.Dropout(0.5)
        )

        self.alpha_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.alpha_li = nn.Linear(emb_dim, 1)
        self.beta_li = nn.Linear(emb_dim, emb_dim)

        self.output = nn.Linear(emb_dim, self.output_len)

        # bipartite matrix
        # self.ddi_adj = ddi_adj
        # self.tensor_ddi_adj = nn.Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)

    def forward(self, conditions, procedures):
        conditions = conditions.cuda()
        procedures = procedures.cuda()
        conditions_emb = self.condition_embedding(conditions).sum(dim=1)
        procedures_emb = self.procedure_embedding(procedures).sum(dim=1)
        visit_emb = conditions_emb + procedures_emb  # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))  # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))  # h: (1, visit, emb)

        g = g.squeeze(dim=0)  # (visit, emb)
        h = h.squeeze(dim=0)  # (visit, emb)
        attn_g = torch.softmax(self.alpha_li(g), dim=-1)  # (visit, 1)
        attn_h = torch.tanh(self.beta_li(h))  # (visit, emb)

        c = attn_g * attn_h * visit_emb  # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, emb)

        drug_rep = self.output(c)
        # ddi_loss
        # neg_pred_prob = F.sigmoid(drug_rep)
        # neg_pred_prob = neg_pred_prob.T @ neg_pred_prob  # (voc_size, voc_size)
        # ddi_loss = 1 / self.voc_size[2] * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        # return  drug_rep, ddi_loss

        return drug_rep

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = train_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[: i + 1], procedures[: i + 1])
            drugs_index = drugs[i : i + 1].cuda()
            drugs_multihot = torch.zeros(1, self.output_len).cuda()
            drugs_multihot[0][drugs_index[0]] = 1
            loss += F.binary_cross_entropy_with_logits(output_logits, drugs_multihot)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = val_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[: i + 1], procedures[: i + 1])
            drugs_index = drugs[i : i + 1].cuda()
            drugs_multihot = torch.zeros(1, self.output_len).cuda()
            drugs_multihot[0][drugs_index[0]] = 1
            loss += F.binary_cross_entropy_with_logits(output_logits, drugs_multihot)
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
                    target_output, _ = self.forward(X[: i + 1])
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

        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)
        print("--- Test Summary ---")
        print(
            "DDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n".format(
                ddi_rate,
                np.mean(ja),
                np.mean(prauc),
                np.mean(avg_p),
                np.mean(avg_r),
                np.mean(avg_f1),
                med_cnt / visit_cnt,
            )
        )

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
                diag = self.maps["diag"].decodes(cur_visit[0])
                if -1 in diag:
                    diag.remove(-1)
                prod = self.maps["prod"].decodes(cur_visit[1])
                if -1 in prod:
                    prod.remove(-1)
                gt_med = self.maps["med"].decodes(cur_visit[2])
                if -1 in gt_med:
                    gt_med.remove(-1)
                pre_logits = cur_visit[5]
                pre_med = np.where(pre_logits >= 0.5)[0]
                if pat_id not in nested_dict:
                    nested_dict[pat_id] = {}
                nested_dict[pat_id][visit_id] = {
                    "diagnoses": diag,
                    "procedures": prod,
                    "real_prescription": gt_med,
                    "predicted_prescription": self.maps["med"].decodes(pre_med),
                    "prediction_logits": {
                        atc3: str(np.round(logit, 4))
                        for atc3, logit in zip(
                            self.maps["med"].code_to_idx.keys(), pre_logits
                        )
                    },
                }

        with open(output_path, "w") as outfile:
            json.dump(nested_dict, outfile)


if __name__ == "__main__":
    from pyhealth.datasets.mimic3 import MIMIC3BaseDataset
    from pyhealth.data.dataset import DrugRecommendationDataset
    from torch.utils.data import DataLoader

    base_dataset = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4"
    )
    task_taskset = DrugRecommendationDataset(base_dataset)
    data_loader = DataLoader(task_taskset, batch_size=1, collate_fn=lambda x: x[0])
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    model = RETAIN(task_taskset)
    print(model.training_step(batch, 0))