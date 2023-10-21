from typing import Dict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models import TransformerLayer
from pyhealth.tokenizer import Tokenizer


def batch_to_one_hot(label_batch, num_class):
    """ convert to one hot label """
    label_batch_onehot = []
    for label in label_batch:
        label_batch_onehot.append(F.one_hot(label, num_class).sum(dim=0))
    label_batch_onehot = torch.stack(label_batch_onehot, dim=0)
    label_batch_onehot[label_batch_onehot > 1] = 1
    return label_batch_onehot


class AdmissionPrediction(nn.Module):
    def __init__(self, tokenizer, embedding_dim, heads=2, dropout=0.5, num_layers=1):
        super(AdmissionPrediction, self).__init__()
        self.tokenizer = tokenizer
        self.vocabs_size = tokenizer.get_vocabulary_size()
        self.embedding = nn.Embedding(
            self.vocabs_size,
            embedding_dim,
            padding_idx=tokenizer.get_padding_index()
        )
        self.encoder = TransformerLayer(
            feature_size=embedding_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def encode_one_hot(self, input: List[str], device):
        input_batch = self.tokenizer.batch_encode_2d(input, padding=True)
        input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
        input_onehot = batch_to_one_hot(input_batch, self.vocabs_size)
        input_onehot = input_onehot.float().to(device)
        input_onehot[:, self.tokenizer.vocabulary("<pad>")] = 0
        input_onehot[:, self.tokenizer.vocabulary("<cls>")] = 0
        return input_onehot

    def encode_dense(self, input: List[str], device):
        input_batch = self.tokenizer.batch_encode_2d(input, padding=True)
        input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
        mask = input_batch != 0
        input_embeddings = self.embedding(input_batch)
        input_embeddings, _ = self.encoder(input_embeddings)
        return input_embeddings, mask

    def get_loss(self, logits, target_onehot):
        true_batch_size = min(logits.shape[0], target_onehot.shape[0])
        loss = self.criterion(logits[:true_batch_size], target_onehot[:true_batch_size])
        return loss

    def forward(self, input, vocab_emb, device):
        input_dense, mask = self.encode_dense(input, device)
        input_one_hot = self.encode_one_hot(input, device)
        logits = torch.matmul(input_dense, vocab_emb.T)
        logits[~mask] = -1e9
        logits = logits.max(dim=1)[0]
        return logits, input_one_hot


class MedLink(BaseModel):
    """MedLink model.

    Paper: Zhenbang Wu et al. MedLink: De-Identified Patient Health
    Record Linkage. KDD 2023.

    IMPORTANT: This implementation differs from the original paper in order to
    make it work with the PyHealth framework. Specifically, we do not use the
    pre-trained GloVe embeddings. And we only monitor the loss on the validation
    set instead of the ranking metrics. As a result, the performance of this model
    is different from the original paper. To reproduce the results in the paper,
    please use the official GitHub repo: https://github.com/zzachw/MedLink.

    Args:
        dataset: SampleEHRDataset.
        feature_keys: List of feature keys. MedLink only supports one feature key.
        embedding_dim: Dimension of embedding.
        alpha: Weight of the forward prediction loss.
        beta: Weight of the backward prediction loss.
        gamma: Weight of the retrieval loss.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        embedding_dim: int = 128,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        **kwargs,
    ):
        assert len(feature_keys) == 1, "MedLink only supports one feature key"
        super(MedLink, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=None,
            mode=None,
        )
        self.feature_key = feature_keys[0]
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        q_tokens = self.dataset.get_all_tokens(key=self.feature_key)
        d_tokens = self.dataset.get_all_tokens(key="d_" + self.feature_key)
        tokenizer = Tokenizer(
            tokens=q_tokens + d_tokens,
            special_tokens=["<pad>", "<unk>", "<cls>"],
        )
        self.fwd_adm_pred = AdmissionPrediction(tokenizer, embedding_dim, **kwargs)
        self.bwd_adm_pred = AdmissionPrediction(tokenizer, embedding_dim, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.vocabs_size = tokenizer.get_vocabulary_size()
        return

    def encode_queries(self, queries: List[str]):
        all_vocab = torch.tensor(list(range(self.vocabs_size)), device=self.device)
        bwd_vocab_emb = self.bwd_adm_pred.embedding(all_vocab)
        pred_corpus, queries_one_hot = self.bwd_adm_pred(
            queries, bwd_vocab_emb, device=self.device
        )
        pred_corpus = torch.log(1 + torch.relu(pred_corpus))
        queries_emb = pred_corpus + queries_one_hot
        return queries_emb

    def encode_corpus(self, corpus: List[str]):
        all_vocab = torch.tensor(list(range(self.vocabs_size)), device=self.device)
        fwd_vocab_emb = self.fwd_adm_pred.embedding(all_vocab)
        pred_queries, corpus_one_hot = self.fwd_adm_pred(
            corpus, fwd_vocab_emb, device=self.device
        )
        pred_queries = torch.log(1 + torch.relu(pred_queries))
        corpus_emb = corpus_one_hot + pred_queries
        return corpus_emb

    def compute_scores(self, queries_emb, corpus_emb):
        n = torch.tensor(corpus_emb.shape[0]).to(queries_emb.device)
        df = (corpus_emb > 0).sum(dim=0)
        idf = torch.log(1 + n) - torch.log(1 + df)

        tf = torch.einsum('ac,bc->abc', queries_emb, corpus_emb)

        tf_idf = tf * idf
        final_scores = tf_idf.sum(dim=-1)
        return final_scores

    def get_loss(self, scores):
        label = torch.tensor(list(range(scores.shape[0])), device=scores.device)
        loss = self.criterion(scores, label)
        return loss

    def forward(self, query_id, id_p, s_q, s_p, s_n=None) -> Dict[str, torch.Tensor]:
        corpus = s_p if s_n is None else s_p + s_n
        queries = s_q
        all_vocab = torch.tensor(list(range(self.vocabs_size)), device=self.device)
        fwd_vocab_emb = self.fwd_adm_pred.embedding(all_vocab)
        bwd_vocab_emb = self.bwd_adm_pred.embedding(all_vocab)
        pred_queries, corpus_one_hot = self.fwd_adm_pred(
            corpus, fwd_vocab_emb, self.device
        )
        pred_corpus, queries_one_hot = self.bwd_adm_pred(
            queries, bwd_vocab_emb, self.device
        )

        fwd_cls_loss = self.fwd_adm_pred.get_loss(pred_queries, queries_one_hot)
        bwd_cls_loss = self.bwd_adm_pred.get_loss(pred_corpus, corpus_one_hot)

        pred_queries = torch.log(1 + torch.relu(pred_queries))
        pred_corpus = torch.log(1 + torch.relu(pred_corpus))

        corpus_emb = corpus_one_hot + pred_queries
        queries_emb = pred_corpus + queries_one_hot

        scores = self.compute_scores(queries_emb, corpus_emb)
        ret_loss = self.get_loss(scores)

        loss = self.alpha * fwd_cls_loss + \
               self.beta * bwd_cls_loss + \
               self.gamma * ret_loss
        return {"loss": loss}

    def search(self, queries_ids, queries_embeddings, corpus_ids, corpus_embeddings):
        scores = self.compute_scores(queries_embeddings, corpus_embeddings)
        results = {}
        for q_idx, q_id in enumerate(queries_ids):
            results[q_id] = {}
            for c_idx, c_id in enumerate(corpus_ids):
                results[q_id][c_id] = scores[q_idx, c_idx].item()
        return results

    def evaluate(self, corpus_dataloader, queries_dataloader):
        self.eval()
        all_corpus_ids, all_corpus_embeddings = [], []
        all_queries_ids, all_queries_embeddings = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(corpus_dataloader)):
                corpus_ids, corpus = batch["corpus_id"], batch["s"]
                corpus_embeddings = self.encode_corpus(corpus)
                all_corpus_ids.extend(corpus_ids)
                all_corpus_embeddings.append(corpus_embeddings)
            for i, batch in enumerate(tqdm.tqdm(queries_dataloader)):
                queries_ids, queries = batch["query_id"], batch["s"]
                queries_embeddings = self.encode_queries(queries)
                all_queries_ids.extend(queries_ids)
                all_queries_embeddings.append(queries_embeddings)
            all_corpus_embeddings = torch.cat(all_corpus_embeddings)
            all_queries_embeddings = torch.cat(all_queries_embeddings)
            results = self.search(
                all_queries_ids,
                all_queries_embeddings,
                all_corpus_ids,
                all_corpus_embeddings
            )
        return results


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.models import MedLink
    from pyhealth.models.medlink import convert_to_ir_format
    from pyhealth.models.medlink import get_train_dataloader
    from pyhealth.models.medlink import tvt_split
    from pyhealth.tasks import patient_linkage_mimic3_fn
    from pyhealth.datasets import MIMIC3Dataset

    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD"],
        code_mapping={"ICD9CM": ("CCSCM", {})},
        dev=False,
        refresh_cache=False,
    )

    sample_dataset = base_dataset.set_task(patient_linkage_mimic3_fn)
    corpus, queries, qrels = convert_to_ir_format(sample_dataset.samples)
    tr_queries, va_queries, te_queries, tr_qrels, va_qrels, te_qrels = tvt_split(
        queries, qrels
    )
    train_dataloader = get_train_dataloader(
        corpus, tr_queries, tr_qrels, batch_size=32, shuffle=True
    )
    batch = next(iter(train_dataloader))
    model = MedLink(
        dataset=sample_dataset,
        feature_keys=["conditions"],
        embedding_dim=128,
    )
    with torch.autograd.detect_anomaly():
        o = model(**batch)
        print("loss:", o["loss"])
        o["loss"].backward()
