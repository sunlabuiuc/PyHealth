from __future__ import annotations

from typing import Dict, List, Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.nn.utils.rnn import pad_sequence

from ...datasets import SampleDataset
from ..base_model import BaseModel
from ..transformer import TransformerLayer
from ...processors import SequenceProcessor

from ..embedding import init_embedding_with_pretrained


def _build_shared_vocab(
    q_processor: SequenceProcessor,
    d_processor: SequenceProcessor,
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
) -> Dict[str, int]:
    """Build a shared token->index mapping from two fitted SequenceProcessors.

    The returned vocabulary is deterministic (sorted token order) and always
    includes `pad_token` and `unk_token`.
    """

    vocab: Dict[str, int] = {pad_token: 0, unk_token: 1}

    tokens = set(str(t) for t in q_processor.code_vocab.keys()) | set(
        str(t) for t in d_processor.code_vocab.keys()
    )
    tokens.discard(pad_token)
    tokens.discard(unk_token)

    for t in sorted(tokens):
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def _build_index_remap(
    processor: SequenceProcessor,
    shared_vocab: Dict[str, int],
    unk_idx: int,
) -> torch.Tensor:
    """Build a dense remap tensor old_idx -> shared_idx."""

    size = len(processor.code_vocab)
    remap = torch.full((size,), unk_idx, dtype=torch.long)
    for tok, old_idx in processor.code_vocab.items():
        tok_s = str(tok)
        remap[old_idx] = shared_vocab.get(tok_s, unk_idx)
    return remap


def _to_index_tensor(
    seq: Any,
    processor: SequenceProcessor,
) -> torch.Tensor:
    """Converts a single sequence to an index tensor using the fitted processor."""
    if isinstance(seq, torch.Tensor):
        return seq.long()
    if isinstance(seq, (list, tuple)):
        return processor.process(seq)
    # single token
    return processor.process([seq])


def _pad_and_remap(
    sequences: Sequence[Any],
    processor: SequenceProcessor,
    remap: torch.Tensor,
    pad_value: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pads a batch of sequences and remaps indices into the shared vocab.

    Returns:
        ids_shared: LongTensor [B, L]
        mask: BoolTensor [B, L] where True indicates valid token positions.
    """

    ids = [_to_index_tensor(s, processor) for s in sequences]
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=pad_value)
    if device is not None:
        ids_padded = ids_padded.to(device)
        remap = remap.to(device)
    ids_shared = remap[ids_padded]
    mask = ids_shared != 0
    return ids_shared, mask


class AdmissionPrediction(nn.Module):
    """Admission prediction module used by MedLink.

    This is a lightly-adapted version of the original MedLink implementation,
    refactored to work with PyHealth 2.0 processors (i.e., indexed tensors).
    """

    def __init__(
        self,
        code_vocab: Dict[str, int],
        embedding_dim: int,
        heads: int = 2,
        dropout: float = 0.5,
        num_layers: int = 1,
        pretrained_emb_path: Optional[str] = None,
        freeze_pretrained: bool = False,
    ):
        super().__init__()
        self.code_vocab = code_vocab
        self.vocab_size = len(code_vocab)
        self.pad_idx = code_vocab.get("<pad>", 0)
        self.cls_idx = code_vocab.get("<cls>")

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.pad_idx,
        )
        if pretrained_emb_path:
            init_embedding_with_pretrained(
                self.embedding,
                code_vocab,
                pretrained_emb_path,
                embedding_dim=embedding_dim,
                freeze=freeze_pretrained,
            )

        self.encoder = TransformerLayer(
            feature_size=embedding_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _multi_hot(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Builds a multi-hot label vector per sample."""

        # input_ids: [B, L]
        bsz = input_ids.size(0)
        out = torch.zeros(bsz, self.vocab_size, device=input_ids.device, dtype=torch.float)
        src = torch.ones_like(input_ids, dtype=torch.float)
        out.scatter_add_(1, input_ids, src)
        out = (out > 0).float()
        # Remove special tokens from labels.
        if self.pad_idx is not None:
            out[:, self.pad_idx] = 0
        if self.cls_idx is not None:
            out[:, self.cls_idx] = 0
        return out

    def get_loss(self, logits: torch.Tensor, target_multi_hot: torch.Tensor) -> torch.Tensor:
        true_batch_size = min(logits.shape[0], target_multi_hot.shape[0])
        return self.criterion(logits[:true_batch_size], target_multi_hot[:true_batch_size])

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute vocabulary logits and target multi-hot labels.

        Args:
            input_ids: LongTensor [B, L] in shared vocabulary indices.

        Returns:
            logits: FloatTensor [B, V]
            target: FloatTensor [B, V] multi-hot labels.
        """

        mask = input_ids != self.pad_idx
        x = self.embedding(input_ids)
        x, _ = self.encoder(x, mask=mask)

        # Use embedding table as vocabulary embedding.
        vocab_emb = self.embedding.weight  # [V, D]
        logits = torch.matmul(x, vocab_emb.T)  # [B, L, V]
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        logits = logits.max(dim=1).values  # [B, V]

        target = self._multi_hot(input_ids)
        return logits, target


class MedLink(BaseModel):
    """MedLink model (KDD 2023).

    Paper: Zhenbang Wu et al. MedLink: De-Identified Patient Health Record
    Linkage. KDD 2023.

    IMPORTANT: This implementation differs from the original paper to fit the
    PyHealth 2.0 framework. By default, it uses randomly-initialized embeddings.
    Optionally, you may initialize the embedding tables using a GloVe-style
    text vector file.

    Args:
        dataset: SampleDataset.
        feature_keys: List of feature keys. MedLink only supports one feature.
            If not provided, the model tries to infer it from the dataset.
        embedding_dim: embedding dimension.
        alpha: weight for forward prediction loss.
        beta: weight for backward prediction loss.
        gamma: weight for retrieval loss.
        pretrained_emb_path: optional path to a GloVe-style embedding file.
        freeze_pretrained: if True, freezes embedding weights after init.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import MedLink
        >>> samples = [{"patient_id": "1", "admissions": ["ICD9_430", "ICD9_401"]}, ...]
        >>> input_schema = {"admissions": "code"}
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(samples=samples, input_schema=input_schema, output_schema=output_schema)
        >>> model = MedLink(dataset=dataset, feature_keys=["admissions"])
        >>> batch = {"query_id": [...], "id_p": [...], "s_q": [["ICD9_430", "ICD9_401"]], "s_p": [[...]], "s_n": None}
        >>> out = model(**batch)
        >>> print(out["loss"])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: Optional[List[str]] = None,
        embedding_dim: int = 128,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        pretrained_emb_path: Optional[str] = None,
        freeze_pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(dataset=dataset)

        # Infer feature_keys if not provided
        if feature_keys is None:
            # Try to find a valid pair (x, d_x) in input_processors where both are SequenceProcessors
            candidates = []
            for k in dataset.input_processors:
                if not k.startswith("d_"):
                    # potential q field
                    d_k = "d_" + k
                    if d_k in dataset.input_processors:
                        # check types
                        qp = dataset.input_processors[k]
                        dp = dataset.input_processors[d_k]
                        if isinstance(qp, SequenceProcessor) and isinstance(dp, SequenceProcessor):
                            candidates.append(k)
            
            if len(candidates) == 0:
                raise ValueError("Could not infer a valid feature key pair (x, d_x) from dataset.")
            elif len(candidates) == 1:
                feature_keys = [candidates[0]]
            else:
                # Ambiguous, prioritize "conditions" or "admissions" if present
                if "conditions" in candidates:
                    feature_keys = ["conditions"]
                elif "admissions" in candidates:
                    feature_keys = ["admissions"]
                else:
                    feature_keys = [candidates[0]] # Just pick the first one

        assert len(feature_keys) == 1, "MedLink only supports one feature key"
        self.feature_keys = feature_keys
        self.feature_key = feature_keys[0]
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        q_field = self.feature_key
        d_field = "d_" + self.feature_key
        if q_field not in self.dataset.input_processors or d_field not in self.dataset.input_processors:
            raise KeyError(
                f"MedLink expects both '{q_field}' and '{d_field}' in dataset.input_schema"
            )

        q_processor = self.dataset.input_processors[q_field]
        d_processor = self.dataset.input_processors[d_field]
        if not isinstance(q_processor, SequenceProcessor) or not isinstance(d_processor, SequenceProcessor):
            raise TypeError(
                "MedLink currently supports SequenceProcessor for both query and corpus fields"
            )

        self.q_processor = q_processor
        self.d_processor = d_processor

        # Shared vocabulary across query/corpus streams.
        self.code_vocab = _build_shared_vocab(q_processor, d_processor)
        self.vocab_size = len(self.code_vocab)
        self.unk_idx = self.code_vocab.get("<unk>", 1)

        # Remap tensors from per-field vocab -> shared vocab.
        self.q_remap = _build_index_remap(q_processor, self.code_vocab, self.unk_idx)
        self.d_remap = _build_index_remap(d_processor, self.code_vocab, self.unk_idx)

        self.fwd_adm_pred = AdmissionPrediction(
            code_vocab=self.code_vocab,
            embedding_dim=embedding_dim,
            pretrained_emb_path=pretrained_emb_path,
            freeze_pretrained=freeze_pretrained,
            **kwargs,
        )
        self.forward_encoder = self.fwd_adm_pred.encoder

        self.bwd_adm_pred = AdmissionPrediction(
            code_vocab=self.code_vocab,
            embedding_dim=embedding_dim,
            pretrained_emb_path=pretrained_emb_path,
            freeze_pretrained=freeze_pretrained,
            **kwargs,
        )
        self.backward_encoder = self.bwd_adm_pred.encoder

        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _prepare_queries(self, queries: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return _pad_and_remap(
            queries,
            processor=self.q_processor,
            remap=self.q_remap,
            pad_value=0,
            device=self.device,
        )

    def _prepare_corpus(self, corpus: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return _pad_and_remap(
            corpus,
            processor=self.d_processor,
            remap=self.d_remap,
            pad_value=0,
            device=self.device,
        )

    def encode_queries(self, queries: Sequence[Any]) -> torch.Tensor:
        q_ids, _ = self._prepare_queries(queries)
        pred_corpus, queries_one_hot = self.bwd_adm_pred(q_ids)
        pred_corpus = torch.log1p(torch.relu(pred_corpus))
        emb = pred_corpus + queries_one_hot
        # Keep special tokens out of retrieval scoring.
        emb[:, self.code_vocab.get("<pad>", 0)] = 0
        if "<cls>" in self.code_vocab:
            emb[:, self.code_vocab["<cls>"]] = 0
        return emb

    def encode_corpus(self, corpus: Sequence[Any]) -> torch.Tensor:
        c_ids, _ = self._prepare_corpus(corpus)
        pred_queries, corpus_one_hot = self.fwd_adm_pred(c_ids)
        pred_queries = torch.log1p(torch.relu(pred_queries))
        emb = corpus_one_hot + pred_queries
        emb[:, self.code_vocab.get("<pad>", 0)] = 0
        if "<cls>" in self.code_vocab:
            emb[:, self.code_vocab["<cls>"]] = 0
        return emb

    # ------------------------------------------------------------------
    # Scoring / losses
    # ------------------------------------------------------------------

    def compute_scores(self, queries_emb: torch.Tensor, corpus_emb: torch.Tensor) -> torch.Tensor:
        """TF-IDF-like score used by MedLink.

        queries_emb: [Q, V]
        corpus_emb: [C, V]
        returns: [Q, C]
        """

        n = torch.tensor(float(corpus_emb.shape[0]), device=queries_emb.device)
        df = (corpus_emb > 0).sum(dim=0).float()
        idf = torch.log1p(n) - torch.log1p(df)
        # Equivalent to sum_c q[c] * d[c] * idf[c]
        return torch.matmul(queries_emb * idf, corpus_emb.T)

    def get_loss(self, scores: torch.Tensor) -> torch.Tensor:
        label = torch.arange(scores.shape[0], device=scores.device)
        return self.criterion(scores, label)

    def forward(self, query_id, id_p, s_q, s_p, s_n=None) -> Dict[str, torch.Tensor]:
        # corpus is positives optionally concatenated with negatives.
        corpus = s_p if s_n is None else (s_p + s_n)
        queries = s_q

        q_ids, _ = self._prepare_queries(queries)
        c_ids, _ = self._prepare_corpus(corpus)

        pred_queries, corpus_one_hot = self.fwd_adm_pred(c_ids)
        pred_corpus, queries_one_hot = self.bwd_adm_pred(q_ids)

        fwd_cls_loss = self.fwd_adm_pred.get_loss(pred_queries, queries_one_hot)
        bwd_cls_loss = self.bwd_adm_pred.get_loss(pred_corpus, corpus_one_hot)

        pred_queries = torch.log1p(torch.relu(pred_queries))
        pred_corpus = torch.log1p(torch.relu(pred_corpus))

        corpus_emb = corpus_one_hot + pred_queries
        queries_emb = pred_corpus + queries_one_hot

        scores = self.compute_scores(queries_emb, corpus_emb)
        ret_loss = self.get_loss(scores)

        loss = self.alpha * fwd_cls_loss + self.beta * bwd_cls_loss + self.gamma * ret_loss
        return {"loss": loss}

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

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
            for batch in tqdm.tqdm(corpus_dataloader):
                corpus_ids, corpus = batch["corpus_id"], batch["s"]
                corpus_embeddings = self.encode_corpus(corpus)
                all_corpus_ids.extend(corpus_ids)
                all_corpus_embeddings.append(corpus_embeddings)
            for batch in tqdm.tqdm(queries_dataloader):
                queries_ids, queries = batch["query_id"], batch["s"]
                queries_embeddings = self.encode_queries(queries)
                all_queries_ids.extend(queries_ids)
                all_queries_embeddings.append(queries_embeddings)
            all_corpus_embeddings = torch.cat(all_corpus_embeddings, dim=0)
            all_queries_embeddings = torch.cat(all_queries_embeddings, dim=0)
            return self.search(
                all_queries_ids,
                all_queries_embeddings,
                all_corpus_ids,
                all_corpus_embeddings,
            )


if __name__ == "__main__":
    # Minimal smoke-test matching the public example script.
    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.models.medlink import (
        convert_to_ir_format,
        get_train_dataloader,
        tvt_split,
    )
    from pyhealth.tasks import patient_linkage_mimic3_fn

    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD"],
        code_mapping={"ICD9CM": ("CCSCM", {})},
        dev=False,
        refresh_cache=False,
    )

    sample_dataset = base_dataset.set_task(patient_linkage_mimic3_fn)
    corpus, queries, qrels, *_ = convert_to_ir_format(sample_dataset.samples)
    tr_queries, _, _, tr_qrels, _, _ = tvt_split(queries, qrels)
    train_dataloader = get_train_dataloader(corpus, tr_queries, tr_qrels, batch_size=4)
    batch = next(iter(train_dataloader))
    model = MedLink(dataset=sample_dataset, feature_keys=["conditions"], embedding_dim=32)
    out = model(**batch)
    print("loss:", out["loss"].item())
