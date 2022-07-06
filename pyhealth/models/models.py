import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from pyhealth.utils import multi_label_metric, ddi_rate_score

class RETAIN(pl.LightningModule):
    def __init__(self, voc_size, ddi_adj, emb_dim=64):
        super(RETAIN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_dim, padding_idx=self.input_len),
            nn.Dropout(0.5)
        )

        self.alpha_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.alpha_li = nn.Linear(emb_dim, 1)
        self.beta_li = nn.Linear(emb_dim, emb_dim)

        self.output = nn.Linear(emb_dim, self.output_len)

        # bipartite matrix
        self.ddi_adj = ddi_adj
        self.tensor_ddi_adj = Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)

    def forward(self, input):
        visit_emb = self.embedding(input) # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1) # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0)) # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0)) # h: (1, visit, emb)

        g = g.squeeze(dim=0) # (visit, emb)
        h = h.squeeze(dim=0) # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1) # (visit, 1)
        attn_h = F.tanh(self.beta_li(h)) # (visit, emb)

        c = attn_g * attn_h * visit_emb # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0) # (1, emb)

        drug_rep = self.output(c)
        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.T @ neg_pred_prob  # (voc_size, voc_size)

        ddi_loss = 1 / self.voc_size[2] *  neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, ddi_loss

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        loss = 0
        X, y = train_batch
        for i in range(len(X)):
            output_logits, ddi_loss = self.forward(X[:i+1])
            loss += F.binary_cross_entropy_with_logits(output_logits, y[i:i+1]) + 1e-2 * ddi_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        X, y = val_batch
        for i in range(len(X)):
            output_logits, ddi_loss = self.forward(X[:i+1])
            loss += F.binary_cross_entropy_with_logits(output_logits, y[i:i+1]) + 1e-2 * ddi_loss
        self.log('val_loss', loss)

    def summary(self, test_dataloaders, ckpt_path):
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
                    target_output, _ = self.forward(X[:i+1])
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

                smm_record.append(y_pred_label)
                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 =\
                        multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
                
                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)

        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)
        print('--- Test Summary ---')
        print('\nDDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))

class MICRON(pl.LightningModule):
    def __init__(self, voc_size, ddi_adj, emb_dim=64):
        super(MICRON, self).__init__()
        self.voc_size = voc_size
        
        # embedding tables for diag and prod
        self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        
        self.health_net = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim)
        )
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, voc_size[2])
        )
        # bipartite matrix
        self.ddi_adj = ddi_adj
        self.tensor_ddi_adj = Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)

    def embedding(self, diag, prod):
        # get diag and prod embeddings
        diag_emb = self.dropout(self.embeddings[0](diag).sum(1)) # (#visit, emb_dim)
        prod_emb = self.dropout(self.embeddings[1](prod).sum(1)) # (#visit, emb_dim)

        # concat diag and prod embeddings
        health_representation = torch.cat([diag_emb, prod_emb], dim=1) # (#visit, emb_dim*2)
        health_rep = self.health_net(health_representation) # (#visit, emb_dim)

        # get drug rep embedding
        drug_rep = self.prescription_net(health_rep) # (#visit, voc_size[2])
        return drug_rep

    def forward(self, diag, prod):

	    # patient health representation       
        diag_emb = self.dropout(self.embeddings[0](diag[1:]).sum(1)) # (#visit-1, emb_dim)
        prod_emb = self.dropout(self.embeddings[1](prod[1:]).sum(1)) # (#visit-1, emb_dim)

        diag_emb_last = self.dropout(self.embeddings[0](diag[:-1]).sum(1)) # (#visit-1, emb_dim)
        prod_emb_last = self.dropout(self.embeddings[1](prod[:-1]).sum(1)) # (#visit-1, emb_dim)

        health_representation = torch.cat([diag_emb, prod_emb], dim=1) # (#visit-1, emb_dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=1) # (#visit-1, emb_dim*2)

        health_rep = self.health_net(health_representation) # (#visit-1, dim)
        health_rep_last = self.health_net(health_representation_last) # (#visit-1, dim)
        health_residual_rep = health_rep - health_rep_last # (#visit-1, dim)

	    # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)  

        # reconstructon loss
        rec_loss = 1 / self.voc_size[2] * torch.sum(torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))
        
        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.T @ neg_pred_prob  # (voc_size, voc_size)

        ddi_loss = 1 / self.voc_size[2] *  neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, ddi_loss, rec_loss

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        loss = 0
        diag, prod, y = train_batch
        result, result_last, _, loss_ddi, loss_rec = self.forward(diag, prod)
        loss_bce = 0.75 * F.binary_cross_entropy_with_logits(result, y[1:]) + \
                (1 - 0.75) * F.binary_cross_entropy_with_logits(result_last, y[:-1])
        loss += loss_bce + loss_rec #+ 1e-4 * loss_ddi
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        diag, prod, y = val_batch
        result, result_last, _, loss_ddi, loss_rec = self.forward(diag, prod)
        loss_bce = 0.75 * F.binary_cross_entropy_with_logits(result, y[1:]) + \
                (1 - 0.75) * F.binary_cross_entropy_with_logits(result_last, y[:-1])
        loss += loss_bce + loss_rec #+ 1e-4 * loss_ddi
        self.log('val_loss', loss)

    def summary(self, test_dataloaders, ckpt_path, threshold1=0.8, threshold2=0.2):
        # load the best model
        self.model = torch.load(ckpt_path)
        self.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        label_list, prob_list = [], []
        smm_record = []

        with torch.no_grad():
            for diag, prod, y in test_dataloaders:
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

                for idx in range(len(diag)):
                    if idx == 0:
                        representation_base = self.embedding(diag[idx:idx+1], prod[idx:idx+1])
                        y_old = y[idx].cpu().numpy()
                        continue

                    y_gt_tmp = y[idx].cpu().numpy()
                    y_gt.append(y_gt_tmp)
                    label_list.append(y_gt_tmp)

                    _, _, residual, _, _ = self.forward(diag[idx-1:idx+1], prod[idx-1:idx+1])
                    # prediction prod
                    representation_base += residual
                    y_pred_tmp = F.sigmoid(representation_base).detach().cpu().numpy()[0]
                    y_pred_prob.append(y_pred_tmp)
                    prob_list.append(y_pred_tmp)
                    
                    # prediction med set
                    y_old[y_pred_tmp>=threshold1] = 1
                    y_old[y_pred_tmp<threshold2] = 0
                    y_pred.append(y_old)

                    # prediction label
                    y_pred_label_tmp = np.where(y_old == 1)[0]
                    y_pred_label.append(sorted(y_pred_label_tmp))
                    visit_cnt += 1
                    med_cnt += len(y_pred_label_tmp)
                
                smm_record.append(y_pred_label)
                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)

        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)
        print('--- Test Summary ---')
        print('\nDDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))

    


