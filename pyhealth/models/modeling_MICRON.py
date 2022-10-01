import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.tokenizer import Tokenizer


class MICRON(pl.LightningModule):
    def __init__(self, dataset, emb_dim=64):
        super(MICRON, self).__init__()

        self.condition_tokenizer = Tokenizer(dataset.all_tokens["conditions"])
        self.procedure_tokenizer = Tokenizer(dataset.all_tokens["procedures"])
        self.drug_tokenizer = Tokenizer(dataset.all_tokens["drugs"])

        # ddi_adj = dataset.ddi_adj

        self.emb_dim = emb_dim
        self.output_len = self.drug_tokenizer.get_vocabulary_size()

        self.condition_embedding = nn.Sequential(
            nn.Embedding(
                self.condition_tokenizer.get_vocabulary_size(),
                self.emb_dim,
                padding_idx=0,
            ),
            nn.Dropout(0.5),
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(
                self.procedure_tokenizer.get_vocabulary_size(),
                self.emb_dim,
                padding_idx=0,
            ),
            nn.Dropout(0.5),
        )

        self.dropout = nn.Dropout(p=0.5)

        self.health_net = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim))
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, self.output_len)
        )

    def embedding(self, conditions, procedures):
        # get diag and prod embeddings
        conditions = self.condition_tokenizer(conditions).cuda()
        procedures = self.procedure_tokenizer(procedures).cuda()

        conditions_emb = self.condition_embedding(conditions).sum(dim=1)
        procedures_emb = self.procedure_embedding(procedures).sum(dim=1)

        # concat diag and prod embeddings
        health_representation = torch.cat(
            [conditions_emb, procedures_emb], dim=1
        )  # (#visit, emb_dim*2)
        health_rep = self.health_net(health_representation)  # (#visit, emb_dim)

        # get drug rep embedding
        drug_rep = self.prescription_net(health_rep)  # (#visit, voc_size[2])
        return drug_rep

    def forward(self, conditions, procedures):
        conditions_cur = self.condition_tokenizer(conditions[1:]).cuda()
        procedures_cur = self.procedure_tokenizer(procedures[1:]).cuda()
        conditions_last = self.condition_tokenizer(conditions[:-1]).cuda()
        procedures_last = self.procedure_tokenizer(procedures[:-1]).cuda()

        conditions_emb_cur = self.condition_embedding(conditions_cur).sum(dim=1)
        procedures_emb_cur = self.procedure_embedding(procedures_cur).sum(dim=1)
        conditions_emb_last = self.condition_embedding(conditions_last).sum(dim=1)
        procedures_emb_last = self.procedure_embedding(procedures_last).sum(dim=1)

        health_representation_cur = torch.cat(
            [conditions_emb_cur, procedures_emb_cur], dim=1
        )  # (#visit-1, emb_dim*2)
        health_representation_last = torch.cat(
            [conditions_emb_last, procedures_emb_last], dim=1
        )  # (#visit-1, emb_dim*2)

        health_rep_cur = self.health_net(health_representation_cur)  # (#visit-1, dim)
        health_rep_last = self.health_net(health_representation_last)  # (#visit-1, dim)
        health_residual_rep = health_rep_cur - health_rep_last  # (#visit-1, dim)

        # drug representation
        drug_rep_cur = self.prescription_net(health_rep_cur)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)

        # reconstructon loss
        rec_loss = (
            1
            / self.output_len
            * torch.sum(
                torch.pow(
                    (
                        torch.sigmoid(drug_rep_cur)
                        - torch.sigmoid(drug_rep_last + drug_residual_rep)
                    ),
                    2,
                )
            )
        )

        # ddi_loss
        neg_pred_prob = torch.sigmoid(drug_rep_cur)
        neg_pred_prob = neg_pred_prob.T @ neg_pred_prob  # (voc_size, voc_size)

        return drug_rep_cur, drug_rep_last, drug_residual_rep, rec_loss

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = train_batch.values()
        result, result_last, _, loss_rec = self.forward(conditions, procedures)
        drugs_multihot = torch.zeros(
            len(drugs), self.drug_tokenizer.get_vocabulary_size()
        ).cuda()
        drugs_index = self.drug_tokenizer(drugs).cuda()
        for i in range(len(drugs)):
            drugs_multihot[i][drugs_index[i]] = 1

        loss_bce = 0.75 * F.binary_cross_entropy_with_logits(
            result, drugs_multihot[1:]
        ) + (1 - 0.75) * F.binary_cross_entropy_with_logits(
            result_last, drugs_multihot[:-1]
        )
        loss += loss_bce + loss_rec
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = val_batch.values()
        result, result_last, _, loss_rec = self.forward(conditions, procedures)
        drugs_multihot = torch.zeros(
            len(drugs), self.drug_tokenizer.get_vocabulary_size()
        ).cuda()
        drugs_index = self.drug_tokenizer(drugs).cuda()
        for i in range(len(drugs)):
            drugs_multihot[i][drugs_index[i]] = 1

        loss_bce = 0.75 * F.binary_cross_entropy_with_logits(
            result, drugs_multihot[1:]
        ) + (1 - 0.75) * F.binary_cross_entropy_with_logits(
            result_last, drugs_multihot[:-1]
        )
        loss += loss_bce + loss_rec
        # self.log('val_loss', loss)

    def summary(
        self, output_path, test_dataloaders, ckpt_path, threshold1=0.8, threshold2=0.2
    ):
        # load the best model
        self.model = torch.load(ckpt_path)
        self.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        label_list, prob_list = [], []
        smm_record = []

        with torch.no_grad():
            for step, (diag, prod, y) in enumerate(test_dataloaders):
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

                for idx in range(len(diag)):
                    if idx == 0:
                        representation_base = self.embedding(
                            diag[idx : idx + 1], prod[idx : idx + 1]
                        )
                        y_old = y[idx].cpu().numpy()
                        continue

                    y_gt_tmp = y[idx].cpu().numpy()
                    y_gt.append(y_gt_tmp)
                    label_list.append(y_gt_tmp)

                    _, _, residual, _, _ = self.forward(
                        diag[idx - 1 : idx + 1], prod[idx - 1 : idx + 1]
                    )
                    # prediction prod
                    representation_base += residual
                    y_pred_tmp = (
                        F.sigmoid(representation_base).detach().cpu().numpy()[0]
                    )
                    y_pred_prob.append(y_pred_tmp)
                    prob_list.append(y_pred_tmp)
                    self.pat_info_test[step][idx].append(y_pred_tmp)

                    # prediction med set
                    y_old[y_pred_tmp >= threshold1] = 1
                    y_old[y_pred_tmp < threshold2] = 0
                    y_pred.append(y_old)

                    # prediction label
                    y_pred_label_tmp = np.where(y_old == 1)[0]
                    y_pred_label.append(sorted(y_pred_label_tmp))
                    visit_cnt += 1
                    med_cnt += len(y_pred_label_tmp)

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

        self.prepare_output(output_path)

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
    model = MICRON(task_taskset)
    print(model.training_step(batch, 0))
