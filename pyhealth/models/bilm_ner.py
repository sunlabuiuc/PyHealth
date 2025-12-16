
"""
BiLM + BiLSTM NER Example (Biomedical NER)

This example shows how to:
  1) Load a CoNLL-style biomedical NER dataset.
  2) Train a bidirectional language model (BiLM) on unlabeled tokens.
  3) Train a BiLSTM-based token classifier for NER.
  4) Compare a baseline (no pretraining) vs BiLM-initialized NER.
"""

import argparse
from collections import Counter
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------


def read_conll_file(path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Reads a simple CoNLL-style file: TOKEN<TAB>TAG, blank line separates sentences."""
    sentences, tags = [], []
    curr_tokens, curr_tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if curr_tokens:
                    sentences.append(curr_tokens)
                    tags.append(curr_tags)
                    curr_tokens, curr_tags = [], []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue  # skip malformed lines
            tok, tag = parts
            curr_tokens.append(tok)
            curr_tags.append(tag)
    if curr_tokens:
        sentences.append(curr_tokens)
        tags.append(curr_tags)
    return sentences, tags


def build_vocab(sentences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """Builds a simple token vocabulary from sentences."""
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def build_tag_vocab(tags: List[List[str]]) -> Dict[str, int]:
    """Builds a tag vocabulary."""
    tags_set = set()
    for seq in tags:
        tags_set.update(seq)
    tag2idx = {tag: i for i, tag in enumerate(sorted(tags_set))}
    return tag2idx


class NerDataset(Dataset):
    """Dataset for token-level NER with simple integer encoding."""

    def __init__(
        self,
        sentences: List[List[str]],
        tags: List[List[str]],
        token2idx: Dict[str, int],
        tag2idx: Dict[str, int],
    ):
        assert len(sentences) == len(tags)
        self.sentences = sentences
        self.tags = tags
        self.token2idx = token2idx
        self.tag2idx = tag2idx

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        tokens = self.sentences[idx]
        tags = self.tags[idx]
        token_ids = [self.token2idx.get(tok, self.token2idx["<unk>"]) for tok in tokens]
        tag_ids = [self.tag2idx[tag] for tag in tags]
        return tokens, token_ids, tag_ids


def ner_collate_fn(batch):
    """Pads batch to max length in batch."""
    # batch: list of (tokens, token_ids, tag_ids)
    token_ids_list = [torch.tensor(b[1], dtype=torch.long) for b in batch]
    tag_ids_list = [torch.tensor(b[2], dtype=torch.long) for b in batch]
    lengths = torch.tensor([len(x) for x in token_ids_list], dtype=torch.long)

    max_len = lengths.max().item()
    B = len(batch)

    token_ids_pad = torch.full((B, max_len), fill_value=0, dtype=torch.long)
    tag_ids_pad = torch.full((B, max_len), fill_value=-1, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)

    for i, (t_ids, y_ids) in enumerate(zip(token_ids_list, tag_ids_list)):
        L = len(t_ids)
        token_ids_pad[i, :L] = t_ids
        tag_ids_pad[i, :L] = y_ids
        mask[i, :L] = True

    return {
        "token_ids": token_ids_pad,
        "tag_ids": tag_ids_pad,
        "mask": mask,
        "lengths": lengths,
    }


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------


class BiLM(nn.Module):
    """Simple bidirectional language model over token IDs."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm_fwd = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_bwd = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.out_fwd = nn.Linear(hidden_dim, vocab_size)
        self.out_bwd = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids, mask):
        """
        token_ids: [B, T]
        mask: [B, T] (bool)
        Returns average cross-entropy loss and perplexity.
        """
        B, T = token_ids.shape

        emb = self.emb(token_ids)  # [B, T, D]

        # Forward LM: predict token t from prefix < t
        fwd_input = emb[:, :-1, :]
        fwd_target = token_ids[:, 1:]
        fwd_mask = mask[:, 1:]

        fwd_out, _ = self.lstm_fwd(fwd_input)
        fwd_logits = self.out_fwd(fwd_out)  # [B, T-1, V]

        # Backward LM: predict token t from suffix > t
        rev_ids = torch.flip(token_ids, dims=[1])  # reverse along time
        rev_emb = self.emb(rev_ids)
        bwd_input = rev_emb[:, :-1, :]
        bwd_target = rev_ids[:, 1:]
        bwd_mask = mask[:, 1:]

        bwd_out, _ = self.lstm_bwd(bwd_input)
        bwd_logits = self.out_bwd(bwd_out)  # [B, T-1, V]

        def lm_loss(logits, target, mask_):
            B2, T2, V = logits.shape
            logits_flat = logits.reshape(B2 * T2, V)
            target_flat = target.reshape(B2 * T2)
            mask_flat = mask_.reshape(B2 * T2)
            ignore_index = -100
            target_flat = target_flat.clone()
            target_flat[~mask_flat] = ignore_index
            loss = F.cross_entropy(
                logits_flat, target_flat, ignore_index=ignore_index, reduction="sum"
            )
            n_tokens = mask_flat.sum().clamp_min(1)
            return loss / n_tokens

        loss_fwd = lm_loss(fwd_logits, fwd_target, fwd_mask)
        loss_bwd = lm_loss(bwd_logits, bwd_target, bwd_mask)

        loss = 0.5 * (loss_fwd + loss_bwd)
        return loss


class BiLSTMTagger(nn.Module):
    """BiLSTM token classifier (NER) optionally initialized from BiLM."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_tags: int,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * hidden_dim, num_tags)

    def forward(self, token_ids, mask):
        """
        token_ids: [B, T]
        mask: [B, T] bool
        Returns logits [B, T, num_tags].
        """
        emb = self.emb(token_ids)  # [B, T, D]
        lengths = mask.sum(dim=1)
        # Pack for efficiency
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        emb_sorted = emb.index_select(0, sort_idx)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=token_ids.size(1)
        )
        _, inv_sort_idx = sort_idx.sort()
        out = out.index_select(0, inv_sort_idx)  # [B, T, 2H]
        logits = self.fc(out)
        return logits

    def neg_log_likelihood(self, batch):
        """Token-level cross entropy with masking."""
        token_ids = batch["token_ids"]
        tag_ids = batch["tag_ids"]
        mask = batch["mask"]

        logits = self.forward(token_ids, mask)  # [B, T, C]
        B, T, C = logits.shape

        logits_flat = logits.view(B * T, C)
        tags_flat = tag_ids.view(B * T)
        mask_flat = mask.view(B * T)

        ignore_index = -100
        targets = tags_flat.clone()
        targets[~mask_flat] = ignore_index

        loss = F.cross_entropy(
            logits_flat, targets, ignore_index=ignore_index, reduction="sum"
        )
        n_tokens = mask_flat.sum().clamp_min(1)
        loss = loss / n_tokens
        return loss

    def decode(self, batch):
        """Greedy per-token argmax decoding, respecting mask lengths."""
        token_ids = batch["token_ids"]
        mask = batch["mask"]
        logits = self.forward(token_ids, mask)
        pred_ids = logits.argmax(dim=-1)  # [B, T]
        preds = []
        for i in range(mask.size(0)):
            L = int(mask[i].sum().item())
            preds.append(pred_ids[i, :L].tolist())
        return preds


# -------------------------------------------------------------------
# Training / evaluation helpers
# -------------------------------------------------------------------


def train_bilm(
    model: BiLM,
    dataloader: DataLoader,
    num_epochs: int = 5,
    lr: float = 1e-3,
) -> None:
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(dataloader, desc=f"BiLM epoch {epoch}"):
            token_ids = batch["token_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            optimizer.zero_grad()
            loss = model(token_ids, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"[BiLM] Epoch {epoch} avg_loss={avg_loss:.4f}")


def train_ner(
    model: BiLSTMTagger,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"NER epoch {epoch}"):
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        p, r, f1 = eval_ner(model, dev_loader)
        print(f"[NER] Epoch {epoch} train_loss={avg_loss:.4f}, "
              f"dev_f1={f1:.4f} (P={p:.4f}, R={r:.4f})")


def eval_ner(model: BiLSTMTagger, dataloader: DataLoader):
    model.eval()
    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            # CPU batch for convenience in decode
            gold_ids = batch["tag_ids"]
            mask = batch["mask"]
            batch_gpu = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            preds = model.decode(batch_gpu)  # list of [L_i]
            for i in range(mask.size(0)):
                L = int(mask[i].sum().item())
                gold_seq = gold_ids[i, :L].tolist()
                pred_seq = preds[i]
                all_gold.extend(gold_seq)
                all_pred.extend(pred_seq)
    p, r, f1, _ = precision_recall_fscore_support(
        all_gold, all_pred, average="micro", zero_division=0
    )
    return p, r, f1


def build_unlabeled_dataloader(
    sentences: List[List[str]],
    token2idx: Dict[str, int],
    batch_size: int = 32,
) -> DataLoader:
    """Turn sentences into a BiLM dataset (only token_ids + mask)."""

    class LMDataset(Dataset):
        def __init__(self, sents, tok2idx):
            self.sents = sents
            self.tok2idx = tok2idx

        def __len__(self):
            return len(self.sents)

        def __getitem__(self, idx):
            tokens = self.sents[idx]
            ids = [self.tok2idx.get(tok, self.tok2idx["<unk>"]) for tok in tokens]
            return torch.tensor(ids, dtype=torch.long)

    def lm_collate(batch_ids):
        lengths = torch.tensor([len(x) for x in batch_ids], dtype=torch.long)
        max_len = lengths.max().item()
        B = len(batch_ids)
        token_ids = torch.full((B, max_len), fill_value=0, dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.bool)
        for i, ids in enumerate(batch_ids):
            L = len(ids)
            token_ids[i, :L] = ids
            mask[i, :L] = True
        return {"token_ids": token_ids, "mask": mask, "lengths": lengths}

    ds = LMDataset(sentences, token2idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lm_collate)
    return dl


# -------------------------------------------------------------------
# Synthetic tiny dataset helper
# -------------------------------------------------------------------


def build_synthetic_dataset():
    """Returns tiny synthetic train/dev/test splits.

    This is for quick sanity checks and tests without external files.
    """
    train_sents = [
        ["BRAF", "mutation", "in", "melanoma"],
        ["EGFR", "mutations", "are", "common"],
        ["TP53", "mutation", "in", "cancer"],
    ]
    train_tags = [
        ["B-GENE", "O", "O", "B-DISEASE"],
        ["B-GENE", "O", "O", "O"],
        ["B-GENE", "O", "O", "B-DISEASE"],
    ]
    dev_sents = [
        ["BRAF", "mutation"],
        ["EGFR", "mutation"],
    ]
    dev_tags = [
        ["B-GENE", "O"],
        ["B-GENE", "O"],
    ]
    test_sents = [
        ["TP53", "mutation"],
        ["BRAF", "in", "cancer"],
    ]
    test_tags = [
        ["B-GENE", "O"],
        ["B-GENE", "O", "B-DISEASE"],
    ]
    return (train_sents, train_tags), (dev_sents, dev_tags), (test_sents, test_tags)
