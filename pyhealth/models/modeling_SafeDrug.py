from gettext import npgettext
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.models.tokenizer import Tokenizer

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

        self.condition_tokenizer = Tokenizer(dataset.all_tokens['conditions'])
        self.procedure_tokenizer = Tokenizer(dataset.all_tokens['procedures'])
        self.drug_tokenizer = Tokenizer(dataset.all_tokens['drugs'])

        # ddi_adj = dataset.ddi_adj

        self.emb_dim = emb_dim
        self.output_len = self.drug_tokenizer.get_vocabulary_size()

        self.condition_embedding = nn.Sequential(
            nn.Embedding(self.condition_tokenizer.get_vocabulary_size(), self.emb_dim, padding_idx=0),
            nn.Dropout(0.5)
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(self.procedure_tokenizer.get_vocabulary_size(), self.emb_dim, padding_idx=0),
            nn.Dropout(0.5)
        )

        self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # TODO: get the ddi_mask_H
        # bipartite local embedding
        ddi_mask_H = np.ones((self.output_len, emb_dim))
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = _MaskLinear(ddi_mask_H.shape[1], self.output_len, False)
        
        # TODO: get the molecule information
        # MPNN global embedding
        # MPNNSet, N_fingerprints, average_projection = med_molecule_info
        # self.MPNN_molecule_Set = list(zip(*MPNNSet))

        # self.MPNN = _MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2)
        # self.MPNN_emb = self.MPNN(self.MPNN_molecule_Set)

        self.MPNN_emb = nn.Parameter(torch.randn(self.output_len, emb_dim), requires_grad=True)
        self.MPNN_output = nn.Linear(self.output_len, self.output_len)
        self.MPNN_layernorm = nn.LayerNorm(self.output_len)

        self.tensor_ddi_mask_H = nn.Parameter(torch.FloatTensor(ddi_mask_H), requires_grad=False)
        
    def forward(self, conditions, procedures):
        conditions = self.condition_tokenizer(conditions).cuda()
        procedures = self.procedure_tokenizer(procedures).cuda()
        conditions_emb = self.condition_embedding(conditions).sum(dim=1)
        procedures_emb = self.procedure_embedding(procedures).sum(dim=1)

        o1, _ = self.encoders[0](conditions_emb) # (seq, emb_dim)
        o2, _ = self.encoders[1](procedures_emb) 
        patient_representations = torch.cat([o1, o2], dim=1) # (seq, emb_dim*2)
        query = self.query(patient_representations)[-1:, :] # (seq, dim)
        
	    # MPNN embedding
        MPNN_match = torch.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))
        
	    # local embedding
        bipartite_emb = self.bipartite_output(torch.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())
        
        result = torch.mul(bipartite_emb, MPNN_att)
        
        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        return result

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = train_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[:i + 1], procedures[:i + 1])
            drugs_index = self.drug_tokenizer(drugs[i: i + 1]).cuda()
            drugs_multihot = torch.zeros(1, self.drug_tokenizer.get_vocabulary_size()).cuda()
            drugs_multihot[0][drugs_index[0]] = 1
            loss += F.binary_cross_entropy_with_logits(output_logits, drugs_multihot)
        # self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = val_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[:i + 1], procedures[:i + 1])
            drugs_index = self.drug_tokenizer(drugs[i: i + 1]).cuda()
            drugs_multihot = torch.zeros(1, self.drug_tokenizer.get_vocabulary_size()).cuda()
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
    model = RETAIN(task_taskset)
    print(model.training_step(batch, 0))
