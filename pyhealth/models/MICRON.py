import torch
import torch.nn as nn
import torch.nn.functional as F

from .GAMENet import get_last_visit


class MICRONLayer(nn.Module):
    def __init__(self, voc_size, emb_dim=64, **kwargs):
        super(MICRONLayer, self).__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.out_dim = voc_size[2] - 2

        # parameters
        self.embedding = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in voc_size])

        self.health_net = nn.Sequential(nn.Linear(2 * emb_dim, emb_dim))
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, self.out_dim)
        )

    def forward(self, tensors, masks=None):
        """
        Args:
            tensors: list of input tensors, each tensor is of shape (batch, visit, code_len)
            masks: list of input masks, each mask is of shape (batch, visit)
        """
        diag_tensor, proc_tensor, med_tensor = tensors
        diag_mask, proc_mask, med_mask = masks
        diag_emb = self.embedding[0](diag_tensor)  # (batch, visit, code_len, dim)
        proc_emb = self.embedding[1](proc_tensor)  # (batch, visit, code_len, dim)

        # sum over code_len
        diag_emb = (diag_emb * diag_mask.unsqueeze(-1).float()).sum(
            dim=2
        )  # (batch, visit, dim)
        proc_emb = (proc_emb * proc_mask.unsqueeze(-1).float()).sum(
            dim=2
        )  # (batch, visit, dim)

        # concat diag and prod embeddings
        health_representation = torch.cat(
            [diag_emb, proc_emb], dim=-1
        )  # (batch, visit, emb_dim*2)
        # health_rep = self.health_net(health_representation)  # (batch, visit, emb_dim)

        health_rep = self.health_net(health_representation)  # (batch, visit, dim)
        health_rep_cur = health_rep[:, :-1, :]  # (batch, visit-1, dim)
        health_rep_last = health_rep[:, 1:, :]  # (batch, visit-1, dim)
        health_residual_rep = health_rep_cur - health_rep_last  # (batch, visit-1, dim)

        # drug representation
        drug_rep_cur = self.prescription_net(health_rep_cur)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)

        # reconstructon loss
        rec_loss = (
            1
            / self.out_dim
            * torch.sum(
                (
                    torch.sigmoid(drug_rep_cur)
                    - torch.sigmoid(drug_rep_last + drug_residual_rep)
                )
                ** 2
            )
        )

        # get drug rep embedding
        health_rep = get_last_visit(health_rep, diag_mask[:, :, 0])
        drug_rep = self.prescription_net(health_rep)  # (batch, visit, voc_size[2])

        return drug_rep, rec_loss


class MICRON(nn.Module):
    def __init__(self, voc_size, tokenizers, emb_dim=64, **kwargs):
        super(MICRON, self).__init__()

        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]

        self.micron_layer = MICRONLayer(voc_size, emb_dim, **kwargs)
        self.out_dim = voc_size[2] - 2

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

        logits, rec_loss = self.micron_layer(tensors, masks)
        y_prob = torch.sigmoid(logits)

        # target
        y = torch.zeros(diag_tensor.shape[0], self.drug_tokenizer.get_vocabulary_size())
        for idx, sample in enumerate(drugs):
            y[idx, self.drug_tokenizer(sample[-1:])[0]] = 1
        # remove 0 and 1 index (invalid drugs)
        y = y[:, 2:]

        # loss
        loss = (
            F.binary_cross_entropy_with_logits(logits, y.to(device)) + 1e-2 * rec_loss
        )
        return {"loss": loss, "y_prob": y_prob, "y_true": y}
