import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pyhealth.utils_old import multi_label_metric, ddi_rate_score
import json

class RETAIN(pl.LightningModule):
    def __init__(self, dataset, emb_dim=64):
        super(RETAIN, self).__init__()
        # voc_size, ddi_adj from datasets
        voc_size = dataset.voc_size
        ddi_adj = dataset.ddi_adj

        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]
        self.pat_info_test = dataset.pat_info_test
        self.maps = dataset.maps

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
        self.tensor_ddi_adj = nn.Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)

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
                    target_output, _ = self.forward(X[:i+1])
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
                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 =\
                        multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
                
                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)

        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)
        print('--- Test Summary ---')
        print('DDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
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
                        atc3: str(np.round(logit, 4)) for atc3, logit in zip(self.maps['med'].code_to_idx.keys(), pre_logits)
                    }
                }

        with open(output_path, "w") as outfile:
            json.dump(nested_dict, outfile)

class MICRON(pl.LightningModule):
    def __init__(self, dataset, emb_dim=64):
        super(MICRON, self).__init__()
        # voc_size, ddi_adj from datasets
        voc_size = dataset.voc_size
        ddi_adj = dataset.ddi_adj
        self.pat_info_test = dataset.pat_info_test
        self.maps = dataset.maps

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
        self.tensor_ddi_adj = nn.Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)

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

    def summary(self, output_path, test_dataloaders, ckpt_path, threshold1=0.8, threshold2=0.2):
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
                    self.pat_info_test[step][idx].append(y_pred_tmp)
                    
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
        print('DDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))

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
                        atc3: str(np.round(logit, 4)) for atc3, logit in zip(self.maps['med'].code_to_idx.keys(), pre_logits)
                    }
                }

        with open(output_path, "w") as outfile:
            json.dump(nested_dict, outfile)

class _GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1)**0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class _GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj):
        super(_GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = nn.Parameter(torch.FloatTensor(adj), requires_grad=False)
        self.x = nn.Parameter(torch.eye(voc_size), requires_grad=False)

        self.gcn1 = _GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = _GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GAMENet(pl.LightningModule):
    def __init__(self, dataset, emb_dim=64, ddi_in_memory=True):
        super(GAMENet, self).__init__()
        # voc_size, ddi_adj, ehr_adj from datasets
        voc_size = dataset.voc_size
        ddi_adj = dataset.ddi_adj
        ehr_adj = dataset.ehr_adj
        self.pat_info_test = dataset.pat_info_test
        self.maps = dataset.maps

        self.voc_size = voc_size
        self.tensor_ddi_adj = nn.Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        self.ddi_adj = ddi_adj
        self.ehr_gcn = _GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=ehr_adj)
        self.ddi_gcn = _GCN(voc_size=voc_size[2], emb_dim=emb_dim, adj=ddi_adj)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, voc_size[2])
        )

    def forward(self, diag, prod, med_before):
        """
        diag: (seq, #diag)
        prod: (seq, #prod)
        """
        # generate medical embeddings and queries
        i1 = self.dropout(self.embeddings[0](diag)).sum(1) # (seq, emb_dim)
        i2 = self.dropout(self.embeddings[1](prod)).sum(1) # (seq, emb_dim)

        o1, _ = self.encoders[0](i1) # (seq, emb_dim)
        o2, _ = self.encoders[1](i2) 
        patient_representations = torch.cat([o1, o2], dim=1) # (seq, emb_dim*2)
        queries = self.query(patient_representations) # (seq, emb_dim*2)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:] # (1, emb_dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if diag.shape[0] > 1:
            history_keys = queries[:-1] # (seq-1, emb_dim)
            history_values = med_before # (seq-1, med_size)
            
        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, med_size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, emb_dim)

        if diag.shape[0] > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        neg_pred_prob = F.sigmoid(output)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        loss_ddi = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

        return output, loss_ddi

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        loss = 0
        diag, prod, y = train_batch
        for i in range(len(diag)):
            result, loss_ddi = self.forward(diag[:i+1], prod[:i+1], y[:i])
            loss += F.binary_cross_entropy_with_logits(result, y[i:i+1]) + 1e-2 * loss_ddi
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        diag, prod, y = val_batch
        for i in range(len(diag)):
            result, loss_ddi = self.forward(diag[:i+1], prod[:i+1], y[:i])
            loss += F.binary_cross_entropy_with_logits(result, y[i:i+1]) + 1e-2 * loss_ddi
        self.log('val_loss', loss)

    def summary(self, output_path, test_dataloaders, ckpt_path):
        # load the best model
        self.model = torch.load(ckpt_path)
        self.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        smm_record = []

        with torch.no_grad():
            for step, (diag, prod, y) in enumerate(test_dataloaders):
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

                for i in range(len(diag)):
                    target_output, _ = self.forward(diag[:i+1], prod[:i+1], y[:i])
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
                adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 =\
                        multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
                
                ja.append(adm_ja)
                prauc.append(adm_prauc)
                avg_p.append(adm_avg_p)
                avg_r.append(adm_avg_r)
                avg_f1.append(adm_avg_f1)

        ddi_rate = ddi_rate_score(smm_record, self.ddi_adj)
        print('--- Test Summary ---')
        print('DDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))

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
                        atc3: str(np.round(logit, 4)) for atc3, logit in zip(self.maps['med'].code_to_idx.keys(), pre_logits)
                    }
                }

        with open(output_path, "w") as outfile:
            json.dump(nested_dict, outfile)

class _MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(_MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class _MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden):
        super(_MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = nn.Parameter(torch.FloatTensor(np.zeros((M, N))), requires_grad=False)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

class SafeDrug(pl.LightningModule):
    def __init__(self, dataset, emb_dim=64):
        super(SafeDrug, self).__init__()
        # load 'voc_size', 'ddi_adj', 'ddi_mask_H', 'med_molecule_info' from datasets
        voc_size = dataset.voc_size
        ddi_adj = dataset.ddi_adj
        ddi_mask_H = dataset.ddi_mask_H
        med_molecule_info = dataset.med_molecule_info
        self.pat_info_test = dataset.pat_info_test
        self.maps = dataset.maps

        self.voc_size = voc_size
        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(voc_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = _MaskLinear(ddi_mask_H.shape[1], voc_size[2], False)
        
        # MPNN global embedding
        MPNNSet, N_fingerprints, average_projection = med_molecule_info
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN = _MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2)
        self.MPNN_emb = self.MPNN(self.MPNN_molecule_Set)

        self.MPNN_emb = nn.Parameter(torch.mm(average_projection, self.MPNN_emb))
        # self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(voc_size[2], voc_size[2])
        self.MPNN_layernorm = nn.LayerNorm(voc_size[2])
        
        # graphs, bipartite matrix
        self.ddi_adj = ddi_adj
        self.tensor_ddi_adj = nn.Parameter(torch.FloatTensor(ddi_adj), requires_grad=False)
        self.tensor_ddi_mask_H = nn.Parameter(torch.FloatTensor(ddi_mask_H), requires_grad=False)

    def forward(self, diag, prod):

	    # generate medical embeddings and queries
        i1 = self.dropout(self.embeddings[0](diag)).sum(1) # (seq, emb_dim)
        i2 = self.dropout(self.embeddings[1](prod)).sum(1) # (seq, emb_dim)

        o1, _ = self.encoders[0](i1) # (seq, emb_dim)
        o2, _ = self.encoders[1](i2) 
        patient_representations = torch.cat([o1, o2], dim=1) # (seq, emb_dim*2)
        query = self.query(patient_representations)[-1:, :] # (seq, dim)
        
	    # MPNN embedding
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))
        
	    # local embedding
        bipartite_emb = self.bipartite_output(F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())
        
        result = torch.mul(bipartite_emb, MPNN_att)
        
        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        loss_ddi = 1 / self.voc_size[2] * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, loss_ddi

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        loss = 0
        diag, prod, y = train_batch
        target_ddi, kp = 0.06, 0.05
        for i in range(len(diag)):
            result, loss_ddi = self.forward(diag[:i+1], prod[:i+1])
            loss_bce = F.binary_cross_entropy_with_logits(result, y[i:i+1])

            # # get current_ddi
            # result = F.sigmoid(result).detach().cpu().numpy()[0]
            # result[result >= 0.5] = 1
            # result[result < 0.5] = 0
            # y_label = np.where(result == 1)[0]
            # current_ddi_rate = ddi_rate_score([[y_label]], self.ddi_adj)
                
            # if current_ddi_rate <= target_ddi:
            loss += loss_bce + 1e-2 * loss_ddi
            # else:
            #     beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)
            #     loss += beta * loss_bce + (1 - beta) * loss_ddi

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        diag, prod, y = val_batch
        target_ddi, kp = 0.06, 0.05
        for i in range(len(diag)):
            result, loss_ddi = self.forward(diag[:i+1], prod[:i+1])
            loss_bce = F.binary_cross_entropy_with_logits(result, y[i:i+1])

            # get current_ddi
            result = F.sigmoid(result).detach().cpu().numpy()[0]
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
            y_label = np.where(result == 1)[0]
            current_ddi_rate = ddi_rate_score([[y_label]], self.ddi_adj)
                
            if current_ddi_rate <= target_ddi:
                loss += loss_bce
            else:
                beta = min(0, 1 + (target_ddi - current_ddi_rate) / kp)
                loss += beta * loss_bce + (1 - beta) * loss_ddi

        self.log('val_loss', loss)

    def summary(self, output_path, test_dataloaders, ckpt_path):
        # load the best model
        self.model = torch.load(ckpt_path)
        self.eval()

        ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
        med_cnt, visit_cnt = 0, 0
        smm_record = []

        with torch.no_grad():
            for step, (diag, prod, y) in enumerate(test_dataloaders):
                y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

                for i in range(len(diag)):
                    target_output, _ = self.forward(diag[:i+1], prod[:i+1])
                    y_gt.append(y[i].cpu().numpy())

                    # prediction prob
                    target_output = F.sigmoid(target_output).cpu().numpy()[0]
                    y_pred_prob.append(target_output)
                    self.pat_info_test[step][i].append(target_output)

                    # prediction med set
                    y_pred_tmp = target_output.copy()
                    y_pred_tmp[y_pred_tmp >= 0.5] = 1
                    y_pred_tmp[y_pred_tmp < 0.5] = 0
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
        print('DDI rate: {:.4}\nJaccard: {:.4}\nPRAUC: {:.4}\nAVG_PRC: {:.4}\nAVG_RECALL: {:.4}\nAVG_F1: {:.4}\nAVG_MED: {:.4}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))

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
                        atc3: str(np.round(logit, 4)) for atc3, logit in zip(self.maps['med'].code_to_idx.keys(), pre_logits)
                    }
                }

        with open(output_path, "w") as outfile:
            json.dump(nested_dict, outfile)
