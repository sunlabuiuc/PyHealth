import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.tokenizer import Tokenizer


class RNN(pl.LightningModule):
    def __init__(self, dataset, emb_dim=64):
        super(RNN, self).__init__()

        self.condition_tokenizer = Tokenizer(dataset.all_tokens["conditions"])
        self.procedure_tokenizer = Tokenizer(dataset.all_tokens["procedures"])
        self.drug_tokenizer = Tokenizer(dataset.all_tokens["drugs"])

        self.emb_dim = emb_dim
        self.output_len = self.drug_tokenizer.get_vocabulary_size()

        self.condition_embedding = nn.Sequential(
            nn.Embedding(
                self.condition_tokenizer.get_vocabulary_size(),
                self.emb_dim,
                padding_idx=0,
            ),
            nn.Dropout(0.5),
        )
        self.procedure_embedding = nn.Sequential(
            nn.Embedding(
                self.procedure_tokenizer.get_vocabulary_size(),
                self.emb_dim,
                padding_idx=0,
            ),
            nn.Dropout(0.5),
        )

        self.rnn = nn.GRU(emb_dim, emb_dim, batch_first=True)

        self.output = nn.Linear(emb_dim, self.output_len)

    def forward(self, conditions, procedures):
        conditions = self.condition_tokenizer(conditions).to(self.device)
        procedures = self.procedure_tokenizer(procedures).to(self.device)
        conditions_emb = self.condition_embedding(conditions).sum(dim=1)
        procedures_emb = self.procedure_embedding(procedures).sum(dim=1)
        visit_emb = conditions_emb + procedures_emb  # (visit, emb)

        h, _ = self.rnn(visit_emb.unsqueeze(dim=0))  # h: (1, visit, emb)
        h = h.squeeze(dim=0)  # (visit, emb)
        c = torch.sum(h, dim=0).unsqueeze(dim=0)  # (1, emb)

        drug_rep = self.output(c)
        return drug_rep

    def configure_optimizers(self, lr=5e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # TODO: batch processing
        loss = 0
        conditions, procedures, drugs = train_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[: i + 1], procedures[: i + 1])
            drugs_index = self.drug_tokenizer(drugs[i : i + 1])
            drugs_multihot = torch.zeros(
                1, self.drug_tokenizer.get_vocabulary_size()
            ).to(self.device)
            drugs_multihot[0][drugs_index[0]] = 1
            loss += F.binary_cross_entropy_with_logits(output_logits, drugs_multihot)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = 0
        conditions, procedures, drugs = val_batch.values()
        for i in range(len(conditions)):
            output_logits = self.forward(conditions[: i + 1], procedures[: i + 1])
            drugs_index = self.drug_tokenizer(drugs[i : i + 1])
            drugs_multihot = torch.zeros(
                1, self.drug_tokenizer.get_vocabulary_size()
            ).to(self.device)
            drugs_multihot[0][drugs_index[0]] = 1
            loss += F.binary_cross_entropy_with_logits(output_logits, drugs_multihot)
        return loss


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3BaseDataset
    from pyhealth.tasks.drug_recommendation import DrugRecommendationDataset
    from torch.utils.data import DataLoader

    base_dataset = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4"
    )
    task_taskset = DrugRecommendationDataset(base_dataset)
    data_loader = DataLoader(task_taskset, batch_size=1, collate_fn=lambda x: x[0])
    data_loader_iter = iter(data_loader)
    batch = next(data_loader_iter)
    model = RNN(task_taskset)
    print(model.training_step(batch, 0))
