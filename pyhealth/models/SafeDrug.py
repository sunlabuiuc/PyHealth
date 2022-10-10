import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyhealth.metrics import ddi_rate_score
from .GAMENet import get_last_visit


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.weight.size(1) ** 0.5
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
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

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


class SafeDrugLayer(nn.Module):
    def __init__(
        self,
        voc_size,
        ddi_adj,
        ddi_mask_H,
        N_fingerprints,
        molecule_set,
        average_projection,
        emb_dim=64,
        target_ddi=0.08,
        kp=0.05,
        **kwargs
    ):
        super(SafeDrugLayer, self).__init__()

        self.ddi_cal_use = ddi_adj
        ddi_adj = torch.FloatTensor(ddi_adj)
        ddi_mask_H = torch.FloatTensor(ddi_mask_H)
        self.ddi_adj = nn.Parameter(
            ddi_adj, requires_grad=False
        )  # already removed 0 and 1 when created
        # remove 0 and 1 index (as they are invalid drugs)
        self.ddi_mask_H = nn.Parameter(ddi_mask_H[2:, :], requires_grad=False)

        out_dim = voc_size[2] - 2

        # parameters
        self.embedding = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in voc_size])
        # GRU encoder for conditions and procedures
        self.encoder = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )

        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        # local bipartite encoder
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
            # nn.Linear(emb_dim, out_dim)
        )
        # self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], out_dim, False)
        self.bipartite_output = nn.Linear(ddi_mask_H.shape[1], out_dim)

        # global MPNN encoder (add fingerprints and adjacency matrix to parameter list)
        MPNN_molecule_set = list(zip(*molecule_set))

        fingerprints = torch.cat(MPNN_molecule_set[0])
        self.fingerprints = nn.Parameter(fingerprints, requires_grad=False)
        adjacencies = self.pad(MPNN_molecule_set[1], 0)
        self.adjacencies = nn.Parameter(adjacencies, requires_grad=False)
        self.molecule_sizes = MPNN_molecule_set[2]

        self.average_projection = nn.Parameter(average_projection, requires_grad=False)

        self.MPNN = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2)
        self.MPNN_output = nn.Linear(out_dim, out_dim)
        self.MPNN_layernorm = nn.LayerNorm(out_dim)

        # hyperparameter
        self.target_ddi = target_ddi
        self.kp = kp

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
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def forward(self, tensors, masks=None):
        # tensors: [condition, procedure, drug]
        # masks: [condition_mask, procedure_mask, drug_mask]

        diag_tensor, proc_tensor, _ = tensors
        diag_mask, proc_mask, _ = masks
        diag_emb = self.embedding[0](diag_tensor)  # (batch, visit, code_len, dim)
        proc_emb = self.embedding[1](proc_tensor)  # (batch, visit, code_len, dim)

        # sum over code_len
        diag_emb = (diag_emb * diag_mask.unsqueeze(-1).float()).sum(
            dim=2
        )  # (batch, visit, dim)
        proc_emb = (proc_emb * proc_mask.unsqueeze(-1).float()).sum(
            dim=2
        )  # (batch, visit, dim)

        # use RNN encoder
        diag_emb, _ = self.encoder[0](diag_emb)  # (batch, visit, dim)
        proc_emb, _ = self.encoder[1](proc_emb)  # (batch, visit, dim)

        patient_representations = torch.cat(
            [diag_emb, proc_emb], dim=-1
        )  # (batch, visit, dim * 2)
        queries = self.query(patient_representations)  # (batch, visit, dim)
        query = get_last_visit(queries, diag_mask[:, :, 0])  # (batch, dim)

        # # MPNN Encoder
        # MPNN_emb = self.MPNN([self.fingerprints, self.adjacencies, self.molecule_sizes])
        # MPNN_emb = torch.mm(self.average_projection, MPNN_emb)[2:, :]
        # MPNN_match = F.sigmoid(torch.mm(query, MPNN_emb.T))
        # MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match)) # (batch, out_dim)

        # Bipartite Encoder (use the bipartite encoder only for now)
        bipartite_emb = self.bipartite_transform(query)  # (batch, dim)
        bipartite_att = self.bipartite_output(bipartite_emb)  # (batch, out_dim)

        # combine
        result = bipartite_att  # + MPNN_att

        # cal ddi loss
        pred_prob = F.sigmoid(result)
        mul_pred_prob = pred_prob.T @ pred_prob  # (voc_size, voc_size)
        batch_ddi_loss = (
            torch.sum(mul_pred_prob.mul(self.ddi_adj)) / self.ddi_adj.shape[0] ** 2
        )

        pred_prob = pred_prob.detach().cpu().numpy()
        pred_prob[pred_prob >= 0.5] = 1
        pred_prob[pred_prob < 0.5] = 0

        # cur ddi
        cur_ddi_rate = ddi_rate_score(pred_prob, self.ddi_cal_use)

        if cur_ddi_rate > self.target_ddi:
            beta = max(0, 1 + (self.target_ddi - cur_ddi_rate) / self.kp)
            return result, batch_ddi_loss, beta
        else:
            return result, 0, 1


class SafeDrug(nn.Module):
    def __init__(
        self,
        tokenizers,
        voc_size,
        ddi_adj,
        bipartite_info,
        MPNN_info,
        emb_dim=64,
        **kwargs
    ):
        super(SafeDrug, self).__init__()

        ddi_mask_H = bipartite_info
        molecule_set, N_fingerprints, average_projection = MPNN_info
        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]

        self.safedrug_layer = SafeDrugLayer(
            voc_size,
            ddi_adj,
            ddi_mask_H,
            N_fingerprints,
            molecule_set,
            average_projection,
            emb_dim,
            **kwargs
        )

    def forward(self, conditions, procedures, drugs, device=None, **kwargs):
        diag_tensor, diag_mask = [
            item.to(device)
            for item in self.condition_tokenizer.batch_tokenize(conditions)
        ]
        proc_tensor, proc_mask = [
            item.to(device)
            for item in self.procedure_tokenizer.batch_tokenize(procedures)
        ]
        drugs_tensor, drugs_mask = [
            item.to(device) for item in self.drug_tokenizer.batch_tokenize(drugs)
        ]
        tensors = [diag_tensor, proc_tensor, drugs_tensor]
        masks = [diag_mask, proc_mask, drugs_mask]

        logits, batch_ddi_loss, beta = self.safedrug_layer(tensors, masks)
        y_prob = torch.sigmoid(logits)

        # target
        y = torch.zeros(diag_tensor.shape[0], self.drug_tokenizer.get_vocabulary_size())
        for idx, sample in enumerate(drugs):
            y[idx, self.drug_tokenizer(sample[-1:])[0]] = 1
        # remove 0 and 1 index (invalid drugs)
        y = y[:, 2:]

        # loss
        loss = (
            beta * F.binary_cross_entropy_with_logits(logits, y.to(device))
            + (1 - beta) * batch_ddi_loss
        )

        return {"loss": loss, "y_prob": y_prob, "y_true": y}
