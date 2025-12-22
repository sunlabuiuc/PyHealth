from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.models.transformer import TransformerLayer
from pyhealth.tokenizer import Tokenizer


def batch_to_multi_hot(label_batch: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert a 2D batch of label indices into a multi-hot representation.

    Parameters
    ----------
    label_batch:
        Long tensor of shape (batch_size, seq_len) with token indices.
    num_classes:
        Size of vocabulary.

    Returns
    -------
    multi_hot:
        Float tensor of shape (batch_size, num_classes), entries in {0,1}.
    """
    # label_batch: (B, T)
    batch_size, seq_len = label_batch.shape
    flat = label_batch.view(-1)  # (B*T,)
    # Build index for scatter
    row_idx = torch.arange(batch_size, device=label_batch.device).repeat_interleave(seq_len)
    multi_hot = torch.zeros(batch_size, num_classes, device=label_batch.device, dtype=torch.float32)
    multi_hot.index_put_((row_idx, flat), torch.ones_like(flat, dtype=torch.float32), accumulate=True)
    multi_hot.clamp_max_(1.0)
    return multi_hot


class AdmissionEncoder(nn.Module):
    """
    Encodes a sequence of discrete tokens (code sequence) for MedLink.

    It uses:
      - a learnable embedding over the vocabulary
      - a TransformerLayer backbone
      - a BCE-with-logits loss over multi-hot targets
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        embedding_dim: int,
        heads: int = 2,
        dropout: float = 0.5,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocabulary_size()

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

        self.encoder = TransformerLayer(
            feature_size=embedding_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def _encode_tokens(self, seqs: List[List[str]], device: torch.device):
        """
        Turn a batch of token sequences into contextual embeddings and a padding mask.

        seqs: list of list of token strings, e.g. [["250.0","401.9"], ["414.0"], ...]
        """
        token_ids = self.tokenizer.batch_encode_2d(seqs, padding=True)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)  # (B, T)
        pad_idx = self.tokenizer.get_padding_index()
        mask = token_ids != pad_idx  # (B, T)

        emb = self.embedding(token_ids)  # (B, T, D)
        encoded, _ = self.encoder(emb)   # (B, T, D)
        return encoded, mask, token_ids

    def _multi_hot_targets(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Build a multi-hot target vector for each sequence in the batch.
        """
        multi_hot = batch_to_multi_hot(token_ids, self.vocab_size)  # (B, V)
        # Clear special tokens
        pad_id = self.tokenizer.vocabulary("<pad>")
        cls_id = self.tokenizer.vocabulary("<cls>")
        if pad_id is not None:
            multi_hot[:, pad_id] = 0.0
        if cls_id is not None:
            multi_hot[:, cls_id] = 0.0
        return multi_hot

    def logits_and_targets(
        self,
        seqs: List[List[str]],
        vocab_embeddings: torch.Tensor,
        device: torch.device,
    ):
        """
        Compute:
          - per-token logits against vocab embeddings
          - multi-hot label vectors for the sequence.

        Returns
        -------
        logits: (B, V) tensor
        targets: (B, V) tensor multi-hot
        """
        encoded, mask, token_ids = self._encode_tokens(seqs, device=device)  # (B,T,D), (B,T), (B,T)
        targets = self._multi_hot_targets(token_ids)                         # (B,V)

        # encoded: (B,T,D), vocab_embeddings: (V,D)
        # per-token logits: (B,T,V)
        logits = torch.matmul(encoded, vocab_embeddings.T)  # (B,T,V)
        # mask padded positions with large negative value
        mask_expanded = mask.unsqueeze(-1)  # (B,T,1)
        logits = logits.masked_fill(~mask_expanded, -1e9)
        # max-pool over time
        logits = logits.max(dim=1).values  # (B,V)

        return logits, targets

    def classification_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        BCE loss on multi-hot labels; handles potential size mismatches defensively.
        """
        # In case of tiny mismatches, truncate to the smaller dimension.
        batch = min(logits.size(0), targets.size(0))
        logits = logits[:batch]
        targets = targets[:batch]
        return self.criterion(logits, targets)


class MedLink(BaseModel):
    """
    MedLink: de-identified patient record linkage model (KDD 2023).

    This implementation is adapted to the PyHealth framework:
      * no pre-trained GloVe; embeddings are learned from scratch
      * training monitored via loss instead of ranking metrics

    It implements three losses:
      - forward admission prediction (corpus -> queries)
      - backward admission prediction (queries -> corpus)
      - retrieval loss via TF-IDF-style matching
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: List[str],
        embedding_dim: int = 128,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        heads: int = 2,
        dropout: float = 0.5,
        num_layers: int = 1,
        **kwargs,
    ) -> None:
        # MedLink is defined over a single textual / code sequence feature
        assert len(feature_keys) == 1, "MedLink supports exactly one feature key"
        # BaseModel only accepts dataset parameter, not feature_keys, label_key, or mode
        super().__init__(dataset=dataset)
        # Set feature_keys manually since BaseModel extracts it from dataset.input_schema
        # but MedLink needs to use the provided feature_keys
        self.feature_keys = feature_keys
        self.feature_key = feature_keys[0]
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Build vocabulary for both queries and corpus sides
        q_tokens = dataset.get_all_tokens(key=self.feature_key)
        d_tokens = dataset.get_all_tokens(key="d_" + self.feature_key)

        tokenizer = Tokenizer(
            tokens=q_tokens + d_tokens,
            special_tokens=["<pad>", "<unk>", "<cls>"],
        )
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocabulary_size()

        # Two direction-specific encoders (forward / backward)
        self.forward_encoder = AdmissionEncoder(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.backward_encoder = AdmissionEncoder(
            tokenizer=tokenizer,
            embedding_dim=embedding_dim,
            heads=heads,
            dropout=dropout,
            num_layers=num_layers,
        )

        # Retrieval / ranking loss
        self.rank_loss = nn.CrossEntropyLoss()

    # ------------------------
    # Encoding utilities
    # ------------------------
    def _all_vocab_ids(self) -> torch.Tensor:
        return torch.arange(self.vocab_size, device=self.device, dtype=torch.long)

    def encode_queries(self, queries: List[List[str]]) -> torch.Tensor:
        """
        Encode query records into embeddings for retrieval.

        queries: list of token sequences, e.g. [["250.0","401.9"], ...]
        Returns: (num_queries, vocab_size) embedding matrix.
        """
        all_vocab = self._all_vocab_ids()             # (V,)
        bwd_vocab_emb = self.backward_encoder.embedding(all_vocab)  # (V,D)

        logits, multi_hot = self.backward_encoder.logits_and_targets(
            seqs=queries,
            vocab_embeddings=bwd_vocab_emb,
            device=self.device,
        )
        logits = torch.log1p(F.relu(logits))          # smooth nonlinearity
        return logits + multi_hot                     # (Q,V)

    def encode_corpus(self, corpus: List[List[str]]) -> torch.Tensor:
        """
        Encode corpus records into embeddings for retrieval.

        corpus: list of token sequences.
        Returns: (num_docs, vocab_size) embedding matrix.
        """
        all_vocab = self._all_vocab_ids()
        fwd_vocab_emb = self.forward_encoder.embedding(all_vocab)  # (V,D)

        logits, multi_hot = self.forward_encoder.logits_and_targets(
            seqs=corpus,
            vocab_embeddings=fwd_vocab_emb,
            device=self.device,
        )
        logits = torch.log1p(F.relu(logits))
        return logits + multi_hot                                # (D,V)

    # ------------------------
    # Retrieval scoring
    # ------------------------
    @staticmethod
    def compute_scores(queries_emb: torch.Tensor, corpus_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute TF-IDF-like matching scores between queries and corpus.

        queries_emb: (Q,V)
        corpus_emb:  (D,V)

        Returns:
            scores: (Q,D)
        """
        # Inverse document frequency per term
        n_docs = torch.tensor(corpus_emb.shape[0], device=corpus_emb.device, dtype=torch.float32)
        df = (corpus_emb > 0).sum(dim=0)  # (V,)
        idf = torch.log1p(n_docs) - torch.log1p(df)

        # Term-frequency contribution per (query, doc, term)
        tf = torch.einsum("qv,dv->qdv", queries_emb, corpus_emb)  # (Q,D,V)
        tf_idf = tf * idf                                         # broadcast idf over last dim

        scores = tf_idf.sum(dim=-1)  # (Q,D)
        return scores

    def get_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Retrieval loss: each query is matched to its corresponding positive
        document at the same index.
        """
        num_queries = scores.size(0)
        target = torch.arange(num_queries, device=scores.device, dtype=torch.long)
        return self.rank_loss(scores, target)

    # ------------------------
    # Training forward
    # ------------------------
    def forward(
        self,
        query_id,
        id_p,
        s_q,
        s_p,
        s_n=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass used for training.

        Parameters in the batch (dict passed as **batch):
        - query_id: list of query identifiers (unused by the loss)
        - id_p:     list of positive record ids (unused here, used for evaluation)
        - s_q:      list of query sequences (list[list[str]])
        - s_p:      list of positive corpus sequences (list[list[str]])
        - s_n:      optional list of negative corpus sequences (list[list[str]])

        Returns
        -------
        dict with key "loss": scalar tensor.
        """
        # Build full corpus: positives plus negatives if provided
        if s_n is None:
            corpus = s_p
        else:
            corpus = s_p + s_n
        queries = s_q

        # Precompute vocab embeddings for both encoders
        all_vocab = self._all_vocab_ids()
        fwd_vocab_emb = self.forward_encoder.embedding(all_vocab)     # (V,D)
        bwd_vocab_emb = self.backward_encoder.embedding(all_vocab)    # (V,D)

        # Forward and backward admission prediction losses
        # Corpus -> query distributions
        pred_queries, corpus_targets = self.forward_encoder.logits_and_targets(
            seqs=corpus,
            vocab_embeddings=fwd_vocab_emb,
            device=self.device,
        )
        # Query -> corpus distributions
        pred_corpus, query_targets = self.backward_encoder.logits_and_targets(
            seqs=queries,
            vocab_embeddings=bwd_vocab_emb,
            device=self.device,
        )

        fwd_cls_loss = self.forward_encoder.classification_loss(pred_queries, query_targets)
        bwd_cls_loss = self.backward_encoder.classification_loss(pred_corpus, corpus_targets)

        # Turn predictions into dense embeddings
        pred_queries_act = torch.log1p(F.relu(pred_queries))
        pred_corpus_act = torch.log1p(F.relu(pred_corpus))

        corpus_emb = corpus_targets + pred_queries_act
        queries_emb = query_targets + pred_corpus_act

        scores = self.compute_scores(queries_emb, corpus_emb)
        retrieval_loss = self.get_loss(scores)

        total_loss = (
            self.alpha * fwd_cls_loss
            + self.beta * bwd_cls_loss
            + self.gamma * retrieval_loss
        )
        return {"loss": total_loss}

    # ------------------------
    # Retrieval helpers
    # ------------------------
    def search(
        self,
        queries_ids: List[str],
        queries_embeddings: torch.Tensor,
        corpus_ids: List[str],
        corpus_embeddings: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute scores for all (query, corpus) pairs and return as nested dict:
            {query_id: {corpus_id: score, ...}, ...}
        """
        scores = self.compute_scores(queries_embeddings, corpus_embeddings)  # (Q,D)
        results: Dict[str, Dict[str, float]] = {}
        for q_idx, q_id in enumerate(queries_ids):
            row_scores = scores[q_idx]
            results[q_id] = {c_id: row_scores[c_idx].item() for c_idx, c_id in enumerate(corpus_ids)}
        return results

    def evaluate(self, corpus_dataloader, queries_dataloader) -> Dict[str, Dict[str, float]]:
        """
        Run MedLink in retrieval mode on dataloaders for corpus and queries.

        corpus_dataloader yields batches with keys: "corpus_id", "s".
        queries_dataloader yields batches with keys: "query_id", "s".
        """
        self.eval()
        all_corpus_ids: List[str] = []
        all_queries_ids: List[str] = []
        all_corpus_embeddings: List[torch.Tensor] = []
        all_queries_embeddings: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm.tqdm(corpus_dataloader):
                corpus_ids = batch["corpus_id"]
                corpus_seqs = batch["s"]
                corpus_emb = self.encode_corpus(corpus_seqs)
                all_corpus_ids.extend(corpus_ids)
                all_corpus_embeddings.append(corpus_emb)

            for batch in tqdm.tqdm(queries_dataloader):
                query_ids = batch["query_id"]
                query_seqs = batch["s"]
                query_emb = self.encode_queries(query_seqs)
                all_queries_ids.extend(query_ids)
                all_queries_embeddings.append(query_emb)

        corpus_mat = torch.cat(all_corpus_embeddings, dim=0)
        queries_mat = torch.cat(all_queries_embeddings, dim=0)

        return self.search(
            queries_ids=all_queries_ids,
            queries_embeddings=queries_mat,
            corpus_ids=all_corpus_ids,
            corpus_embeddings=corpus_mat,
        )
