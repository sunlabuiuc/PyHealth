# pyhealth/models/labtop_transformer.py
import torch
import torch.nn as nn

class LabTOPTransformer(nn.Module):
    """
    Transformer model for LabTOP sequences.
    """
    def __init__(self, vocab, d_model=128, nhead=4, nlayers=2, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(d_model, vocab)
        self.max_len = max_len

    def _generate_mask(self, L, device):
        mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, x):
        B, L = x.shape
        if L > self.max_len:
            raise ValueError("Sequence length > max_len")
        pos_idx = torch.arange(L, device=x.device).unsqueeze(0)
        x_emb = self.embedding(x) + self.pos(pos_idx)
        src_mask = self._generate_mask(L, x.device)
        out = self.encoder(x_emb, mask=src_mask)
        logits = self.fc(out)
        return logits
