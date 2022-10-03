import torch
import torch.nn as nn
import torch.nn.functional as F


def get_last_visit(hidden_states, mask):
    last_visit = torch.sum(mask, 1) - 1
    last_visit = last_visit.unsqueeze(-1)
    last_visit = last_visit.expand(-1, hidden_states.shape[1] * hidden_states.shape[2])
    last_visit = torch.reshape(last_visit, hidden_states.shape)
    last_hidden_states = torch.gather(hidden_states, 1, last_visit)
    last_hidden_state = last_hidden_states[:, 0, :]
    return last_hidden_state


class RNNLayer(nn.Module):
    """flexible layers for RNN model across different tasks"""

    def __init__(self, voc_size, emb_dim=64, device=torch.device("cpu:0")):
        """
        Args:
            voc_size: list of vocabulary size for each input (#diagnosis, #procedure, #drug)
            emb_dim: embedding dimension
        Attributes:
            embedding: embedding layer for each input
        """
        super(RNNLayer, self).__init__()
        self.device = device
        self.embedding = nn.ModuleList([nn.Embedding(s + 1, emb_dim) for s in voc_size])

        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=3,
            dropout=0.5,
            batch_first=True,
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
        mask = masks[0][:, :, 0]
        visit_emb = torch.stack(visit_emb_ls, dim=2).mean(2)  # (batch, visit, dim)
        visit_emb = self.dropout(visit_emb)

        visit_emb, _ = self.gru(visit_emb)  # (batch, visit, dim)
        visit_emb = get_last_visit(visit_emb, mask)  # (batch, dim)
        return visit_emb


class RNNDrugRec(nn.Module):
    """RNN model for drug recommendation task"""

    def __init__(self, voc_size, tokenizers, emb_dim=64, device=torch.device("cpu:0")):
        super(RNNDrugRec, self).__init__()
        self.device = device
        self.retain = RNNLayer(voc_size, emb_dim, device)

        self.condition_tokenizer = tokenizers[0]
        self.procedure_tokenizer = tokenizers[1]
        self.drug_tokenizer = tokenizers[2]
        self.drug_fc = nn.Linear(emb_dim, self.drug_tokenizer.get_vocabulary_size() - 2)

    def forward(
        self, conditions, procedures, drugs, padding_mask=None, device=None, **kwargs
    ):
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


class RNN(nn.Module):
    """RNN Class, use "task" as key to identify specific RNN model and route there"""

    def __init__(self, **kwargs):
        super(RNN, self).__init__()
        task = kwargs["task"]
        kwargs.pop("task")
        if task == "drug_recommendation":
            self.model = RNNDrugRec(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
