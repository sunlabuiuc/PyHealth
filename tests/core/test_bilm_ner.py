import unittest
import torch
from torch.utils.data import DataLoader

from pyhealth.models.bilm_ner import (
    BiLM,
    build_unlabeled_dataloader,
    BiLSTMTagger,
    build_synthetic_dataset,
    build_vocab,
    build_tag_vocab,
    ner_collate_fn,
    NerDataset
)

class TestBiLMBiLSTMNER(unittest.TestCase):
    """Test cases for the BiLM + BiLSTM NER example."""

    def setUp(self):
        """Set up synthetic data, vocab, datasets, and models."""
        # Synthetic tiny dataset (no external files needed)
        (train_sents, train_tags), (dev_sents, dev_tags), (test_sents, test_tags) = \
            build_synthetic_dataset()

        self.train_sents = train_sents
        self.train_tags = train_tags
        self.dev_sents = dev_sents
        self.dev_tags = dev_tags
        self.test_sents = test_sents
        self.test_tags = test_tags

        # Build vocabularies
        self.token2idx = build_vocab(
            self.train_sents + self.dev_sents + self.test_sents,
            min_freq=1,
        )
        self.tag2idx = build_tag_vocab(
            self.train_tags + self.dev_tags + self.test_tags
        )

        # Build datasets and dataloaders
        self.train_ds = NerDataset(self.train_sents, self.train_tags,
                                   self.token2idx, self.tag2idx)
        self.dev_ds = NerDataset(self.dev_sents, self.dev_tags,
                                 self.token2idx, self.tag2idx)
        self.test_ds = NerDataset(self.test_sents, self.test_tags,
                                  self.token2idx, self.tag2idx)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=2,
            shuffle=False,
            collate_fn=ner_collate_fn,
        )
        self.dev_loader = DataLoader(
            self.dev_ds,
            batch_size=2,
            shuffle=False,
            collate_fn=ner_collate_fn,
        )

        # Unlabeled loader for BiLM (train + dev sentences)
        unlabeled_sents = self.train_sents + self.dev_sents
        self.lm_loader = build_unlabeled_dataloader(
            unlabeled_sents,
            self.token2idx,
            batch_size=2,
        )

        self.vocab_size = len(self.token2idx)
        self.num_tags = len(self.tag2idx)
        self.emb_dim = 16
        self.hidden_dim = 32
        self.pad_idx = self.token2idx["<pad>"]

        # Instantiate models
        self.bilm = BiLM(
            vocab_size=self.vocab_size,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            pad_idx=self.pad_idx,
        )
        self.ner_model = BiLSTMTagger(
            vocab_size=self.vocab_size,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            num_tags=self.num_tags,
            pad_idx=self.pad_idx,
        )

    # ---------------------------------------------------------
    # BiLM tests
    # ---------------------------------------------------------

    def test_bilm_initialization(self):
        """Test that the BiLM model initializes correctly."""
        self.assertIsInstance(self.bilm, BiLM)
        self.assertEqual(self.bilm.emb.num_embeddings, self.vocab_size)
        self.assertEqual(self.bilm.emb.embedding_dim, self.emb_dim)

    def test_bilm_forward_pass(self):
        """Test that the BiLM forward pass works and returns a finite loss."""
        batch = next(iter(self.lm_loader))
        token_ids = batch["token_ids"]
        mask = batch["mask"]

        self.bilm.eval()
        with torch.no_grad():
            loss = self.bilm(token_ids, mask)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar tensor")

    # ---------------------------------------------------------
    # NER model tests
    # ---------------------------------------------------------

    def test_ner_initialization(self):
        """Test that the BiLSTMTagger initializes correctly."""
        self.assertIsInstance(self.ner_model, BiLSTMTagger)
        self.assertEqual(self.ner_model.emb.num_embeddings, self.vocab_size)
        self.assertEqual(self.ner_model.emb.embedding_dim, self.emb_dim)
        # BiLSTM is bidirectional: hidden_dim * 2 in the output
        self.assertEqual(self.ner_model.lstm.hidden_size, self.hidden_dim)
        self.assertEqual(self.ner_model.fc.out_features, self.num_tags)

    def test_ner_forward_pass(self):
        """Test that the NER forward pass produces logits of the right shape."""
        batch = next(iter(self.train_loader))
        token_ids = batch["token_ids"]
        mask = batch["mask"]

        self.ner_model.eval()
        with torch.no_grad():
            logits = self.ner_model(token_ids, mask)

        B, T = token_ids.shape
        self.assertEqual(logits.shape[0], B)
        self.assertEqual(logits.shape[1], T)
        self.assertEqual(logits.shape[2], self.num_tags)

    def test_ner_backward_pass(self):
        """Test that the NER model backward pass computes gradients."""
        batch = next(iter(self.train_loader))
        # Move everything to CPU for a quick backward pass
        batch = {k: v for k, v in batch.items()}
        loss = self.ner_model.neg_log_likelihood(batch)
        loss.backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.ner_model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_ner_decode_lengths(self):
        """Test that decode() returns sequences matching the unpadded lengths."""
        batch = next(iter(self.train_loader))
        mask = batch["mask"]
        # Use CPU batch for convenience
        preds = self.ner_model.decode(batch)

        self.assertEqual(len(preds), mask.shape[0])
        for i, seq in enumerate(preds):
            L = int(mask[i].sum().item())
            self.assertEqual(
                len(seq),
                L,
                f"Decoded sequence length {len(seq)} does not match mask length {L}",
            )


if __name__ == "__main__":
    unittest.main()
