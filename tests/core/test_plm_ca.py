"""Unit tests for PLMCA and ClassWiseCrossAttention.

Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
Explainability in Healthcare Records", EMNLP 2024.

All tests use two synthetic notes and a tiny BERT written to a temporary
directory, so nothing is downloaded and the suite runs in about a second.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import PLMCA, ClassWiseCrossAttention

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "chest",
    "pain",
    "sepsis",
    "heart",
    "failure",
    "fever",
    "acute",
    "renal",
]

# Note p0 tokenizes to 22 tokens (3 chunks of 8); note p1 to 4 tokens, so in
# a batch it ends in two chunks made only of padding.
SAMPLES = [
    {
        "patient_id": "p0",
        "text": "chest pain heart failure " * 5,
        "icd_codes": ["428.0", "786.50"],
    },
    {"patient_id": "p1", "text": "sepsis fever", "icd_codes": ["038.9"]},
]

CHUNK_SIZE = 8
MAX_LENGTH = 32
HIDDEN_DIM = 16


def save_tiny_tokenizer(vocab: list[str], folder: str) -> None:
    """Saves a BERT-style word-piece tokenizer for ``vocab`` to ``folder``.

    Built with the ``tokenizers`` library because ``BertTokenizerFast``
    silently ignores ``vocab_file`` in transformers 5.x, which would map
    every word to ``[UNK]``.
    """
    ids = {token: index for index, token in enumerate(vocab)}
    backend = Tokenizer(models.WordPiece(vocab=ids, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    ).save_pretrained(folder)


class TestClassWiseCrossAttention(unittest.TestCase):
    """Tests for the decoder on its own."""

    def test_shapes_and_padding(self):
        torch.manual_seed(0)
        decoder = ClassWiseCrossAttention(hidden_dim=8, num_classes=3, dropout=0.0)
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])

        logits, attention = decoder(hidden, mask)

        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(attention.shape, (2, 3, 5))
        self.assertTrue(torch.all(attention[1, :, 2:] == 0))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(2, 3)))


class TestPLMCA(unittest.TestCase):
    """Tests for the full model with a tiny local encoder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        save_tiny_tokenizer(VOCAB, cls.tmp_dir)
        config = BertConfig(
            vocab_size=len(VOCAB),
            hidden_size=HIDDEN_DIM,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
        torch.manual_seed(0)
        BertModel(config).save_pretrained(cls.tmp_dir)
        cls.dataset = create_sample_dataset(
            samples=SAMPLES,
            input_schema={"text": "text"},
            output_schema={"icd_codes": "multilabel"},
            dataset_name="test_plm_ca",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def setUp(self):
        torch.manual_seed(42)
        self.model = PLMCA(
            self.dataset,
            model_name=self.tmp_dir,
            chunk_size=CHUNK_SIZE,
            max_length=MAX_LENGTH,
        )
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=2)))

    def test_initialization(self):
        self.assertIsInstance(self.model, PLMCA)
        self.assertEqual(self.model.feature_key, "text")
        self.assertEqual(self.model.label_key, "icd_codes")
        self.assertEqual(self.model.mode, "multilabel")
        self.assertEqual(self.model.decoder.class_queries.shape, (3, HIDDEN_DIM))

    def test_tokenizer_keeps_known_words(self):
        # Guards against a tokenizer that maps every word to [UNK]; the other
        # tests would still pass on shapes alone.
        self.assertEqual(
            self.model.tokenizer.tokenize("Chest pain sepsis"),
            ["chest", "pain", "sepsis"],
        )

    def test_tokenize_pads_to_chunk_multiple(self):
        input_ids, mask = self.model.tokenize(self.batch["text"])
        self.assertEqual(input_ids.shape, mask.shape)
        self.assertEqual(input_ids.shape[1] % CHUNK_SIZE, 0)
        single_ids, _ = self.model.tokenize("sepsis fever")
        self.assertEqual(single_ids.shape[0], 1)

    def test_tokenize_truncates_long_notes(self):
        input_ids, _ = self.model.tokenize(["fever " * 100])
        self.assertEqual(input_ids.shape[1], MAX_LENGTH)

    def test_forward_output_shapes(self):
        out = self.model(**self.batch)
        input_ids, _ = self.model.tokenize(self.batch["text"])

        self.assertEqual(set(out), {"loss", "y_prob", "y_true", "logit", "attention"})
        self.assertEqual(out["logit"].shape, (2, 3))
        self.assertEqual(out["y_prob"].shape, (2, 3))
        self.assertEqual(out["y_true"].shape, (2, 3))
        self.assertEqual(out["attention"].shape, (2, 3, input_ids.shape[1]))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.all((out["y_prob"] >= 0) & (out["y_prob"] <= 1)))

    def test_padding_gets_no_attention(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(**self.batch)
        _, mask = self.model.tokenize(self.batch["text"])

        self.assertFalse(torch.isnan(out["logit"]).any())
        padded = out["attention"].masked_select(mask[:, None, :] == 0)
        self.assertTrue(torch.all(padded == 0))

    def test_backward(self):
        out = self.model(**self.batch)
        out["loss"].backward()

        self.assertIsNotNone(self.model.decoder.class_queries.grad)
        self.assertIsNotNone(self.model.get_input_embeddings().weight.grad)

    def test_forward_without_labels(self):
        out = self.model(text=self.batch["text"])

        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)
        self.assertEqual(out["y_prob"].shape, (2, 3))

    def test_forward_from_embedding_matches_forward(self):
        self.model.eval()
        out = self.model(**self.batch)
        input_ids, mask = self.model.tokenize(self.batch["text"])
        embeds = self.model.get_input_embeddings()(input_ids).detach()
        embeds.requires_grad_()

        out_embed = self.model.forward_from_embedding(embeds, mask)
        out_embed["logit"][:, 0].sum().backward()

        self.assertTrue(torch.allclose(out_embed["logit"], out["logit"], atol=1e-5))
        self.assertIsNotNone(embeds.grad)
        self.assertEqual(embeds.grad.shape, embeds.shape)

    def test_other_configurations(self):
        for chunk_size, max_length, dropout in [(4, 16, 0.0), (16, 48, 0.5)]:
            with self.subTest(chunk_size=chunk_size, max_length=max_length):
                model = PLMCA(
                    self.dataset,
                    model_name=self.tmp_dir,
                    chunk_size=chunk_size,
                    max_length=max_length,
                    dropout=dropout,
                )
                out = model(**self.batch)
                out["loss"].backward()

                self.assertEqual(out["logit"].shape, (2, 3))
                self.assertLessEqual(
                    out["attention"].shape[-1], max_length + chunk_size
                )
                self.assertEqual(out["attention"].shape[-1] % chunk_size, 0)
                self.assertIsNotNone(model.decoder.class_queries.grad)

    def test_save_and_load_state_dict(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(**self.batch)["logit"]
        path = Path(self.tmp_dir) / "plm_ca_test.pt"
        torch.save(self.model.state_dict(), path)

        torch.manual_seed(123)  # different initial decoder weights
        restored = PLMCA(
            self.dataset,
            model_name=self.tmp_dir,
            chunk_size=CHUNK_SIZE,
            max_length=MAX_LENGTH,
        )
        restored.load_state_dict(torch.load(path))
        restored.eval()
        with torch.no_grad():
            actual = restored(**self.batch)["logit"]

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_rejects_non_multilabel_task(self):
        binary_dataset = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "text": "fever", "label": 0},
                {"patient_id": "p1", "text": "sepsis", "label": 1},
            ],
            input_schema={"text": "text"},
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            PLMCA(binary_dataset, model_name=self.tmp_dir, chunk_size=CHUNK_SIZE)

    def test_rejects_chunk_longer_than_encoder(self):
        with self.assertRaises(ValueError):
            PLMCA(self.dataset, model_name=self.tmp_dir, chunk_size=128)


if __name__ == "__main__":
    unittest.main()
