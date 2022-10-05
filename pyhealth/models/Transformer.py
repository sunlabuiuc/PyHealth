import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLayer(nn.Module):
    """flexible layers for Transformer model across different tasks"""

    def __init__(self, voc_size, emb_dim=64, device=torch.device("cpu:0")):
        """
        Args:
            voc_size: list of vocabulary size for each input (#diagnosis, #procedure, #drug)
            emb_dim: embedding dimension
        Attributes:
            embedding: embedding layer for each input
        """
        super(TransformerLayer, self).__init__()
        self.device = device
        self.embedding = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in voc_size])

        self.dropout = nn.Dropout(p=0.5)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            dim_feedforward=256,
            nhead=8,
            dropout=0.25,
            activation="relu",
        )
        encoder_norm = nn.LayerNorm(emb_dim)
        num_encoder_layers = 3
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

    def forward(self, tensors, masks=None):
        """
        Args:
            tensors: list of input tensors, each tensor is of shape (batch, visit, code_len)
            masks: list of input masks, each mask is of shape (batch, visit)
        """
        visit_emb_ls = []
        for i, (tensor, mask) in enumerate(zip(tensors, masks)):
            cur_emb = self.embedding[i](tensor)  # (batch, visit, code_len, dim)
            # mask out padding
            cur_emb = cur_emb * mask.unsqueeze(-1).float()
            visit_emb_ls.append(torch.sum(cur_emb, dim=2))  # (batch, visit, dim)
        visit_emb = torch.stack(visit_emb_ls, dim=2).mean(2)  # (batch, visit, dim)
        visit_emb = self.dropout(visit_emb)

        mask = masks[0][:, :, 0]
        # since the transformer encoder cannot use "batch_first"
        visit_emb = self.encoder(
            visit_emb.permute(1, 0, 2), src_key_padding_mask=~mask
        )  # (batch, visit, dim)
        visit_emb = (visit_emb.permute(1, 0, 2) * mask.unsqueeze(-1).float()).sum(
            1
        )  # (batch, dim)
        return visit_emb


class TransformerDrugRec(nn.Module):
    """Transformer model for drug recommendation task"""

    def __init__(self, voc_size, tokenizers, emb_dim=64, device=torch.device("cpu:0")):
        super(TransformerDrugRec, self).__init__()
        self.device = device
        self.retain = TransformerLayer(voc_size, emb_dim, device)

        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]
        self.drug_fc = nn.Linear(emb_dim, voc_size[2] - 2)

    def forward(self, conditions, procedures, drugs, device=None, **kwargs):
        diagT, diagMask = [
            item.to(device)
            for item in self.condition_tokenizer.batch_tokenize(conditions)
        ]
        procT, procMask = [
            item.to(device)
            for item in self.procedure_tokenizer.batch_tokenize(procedures)
        ]
        tensors = [diagT, procT]
        masks = [diagMask, procMask]
        embedding = self.retain(tensors, masks)
        logits = self.drug_fc(embedding)
        y_prob = torch.sigmoid(logits)

        # target
        y = torch.zeros(diagT.shape[0], self.drug_tokenizer.get_vocabulary_size())
        for idx, sample in enumerate(drugs):
            y[idx, self.drug_tokenizer(sample[-1:])[0]] = 1
        # remove 0 and 1 index (invalid drugs)
        y = y[:, 2:]

        # loss
        loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
        return {"loss": loss, "y_prob": y_prob, "y_true": y}


class Transformer(nn.Module):
    """Transformer Class, use "task" as key to identify specific Transformer model and route there"""

    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        task = kwargs["task"]
        kwargs.pop("task")
        if task == "drug_recommendation":
            self.model = TransformerDrugRec(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
