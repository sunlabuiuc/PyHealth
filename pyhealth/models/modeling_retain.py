import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.modeling_base import BaseModel


# TODO: consider move padding inside forward (like tokenizer)

class RETAIN(BaseModel):
    def __init__(
            self,
            voc_size,
            emb_size=64,
            mode: str = "binary",
            num_class: int = 1,
    ):
        super(RETAIN, self).__init__(mode=mode)
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.num_class = num_class

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.5)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.num_class)

    def forward_current_admission(self, input):
        visit_emb = self.embedding(input)  # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1)  # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))  # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))  # h: (1, visit, emb)

        g = g.squeeze(dim=0)  # (visit, emb)
        h = h.squeeze(dim=0)  # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1)  # (visit, 1)
        attn_h = F.tanh(self.beta_li(h))  # (visit, emb)

        c = attn_g * attn_h * visit_emb  # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, emb)

        return self.output(c)

    def forward(self, X, y):
        loss = 0
        for i in range(len(X)):
            output_logits = self.forward_current_admission(X[:i + 1])
            # TODO: switch to self.loss_fn later
            loss += F.binary_cross_entropy_with_logits(output_logits, y[i:i + 1])
        return loss
