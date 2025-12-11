import torch
from torch import nn

class MiniViewXGen(nn.Module):
    def __init__(self, vocab_size=256, hidden=128, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.transformer(x)
        return self.out(x)
