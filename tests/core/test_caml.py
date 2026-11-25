"""Unit tests for CAML.

Paper: Mullenbach et al., "Explainable Prediction of Medical Codes from
Clinical Text", NAACL 2018.

All tests use two synthetic notes and a tiny BERT written to a temporary
directory (the same as test_plm_ca), so nothing is downloaded and the suite
runs in a few seconds.

Contributors: Juri Moon (jmoon326)
Paper link: https://arxiv.org/abs/1802.05695
"""

import shutil
import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import AttInGrad
from pyhealth.models import CAML
from tests.core.test_plm_ca import HIDDEN_DIM, VOCAB, save_tiny_encoder, tiny_dataset

MAX_LENGTH = 32
NUM_FILTERS = 6


class TestCAML(unittest.TestCase):
    """Tests for CAML with a tiny local tokenizer and embedding matrix."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.encoder = save_tiny_encoder(cls.tmp_dir)
        cls.dataset = tiny_dataset("test_caml")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def build(self, **kwargs):
        torch.manual_seed(42)
        options = dict(
            model_name=self.tmp_dir, max_length=MAX_LENGTH, num_filters=NUM_FILTERS
        )
        options.update(kwargs)
        return CAML(self.dataset, **options)

    def setUp(self):
        self.model = self.build()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=2)))
        self.input_ids, self.mask = self.model.tokenize(self.batch["text"])

    def test_pretrained_embeddings_are_copied(self):
        expected = self.encoder.get_input_embeddings().weight
        actual = self.model.get_input_embeddings().weight
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.requires_grad)
        self.assertEqual(self.model.attention_vectors.shape, (3, NUM_FILTERS))

    def test_random_embeddings(self):
        model = self.build(pretrained_embeddings=False, embed_dim=8)
        self.assertEqual(model.get_input_embeddings().weight.shape, (len(VOCAB), 8))
        self.assertEqual(model(**self.batch)["logit"].shape, (2, 3))

    def test_tokenize_keeps_special_tokens_and_truncates(self):
        ids, _ = self.model.tokenize("sepsis fever")
        tokens = self.model.tokenizer.convert_ids_to_tokens(ids[0])
        self.assertEqual(tokens, ["[CLS]", "sepsis", "fever", "[SEP]"])
        long_ids, _ = self.model.tokenize(["fever " * 100])
        self.assertEqual(long_ids.shape[1], MAX_LENGTH)

    def test_forward_output_shapes(self):
        out = self.model(**self.batch)
        n = self.input_ids.shape[1]
        self.assertEqual(set(out), {"loss", "y_prob", "y_true", "logit", "attention"})
        self.assertEqual(out["logit"].shape, (2, 3))
        self.assertEqual(out["y_prob"].shape, (2, 3))
        self.assertEqual(out["y_true"].shape, (2, 3))
        self.assertEqual(out["attention"].shape, (2, 3, n))
        self.assertEqual(out["loss"].dim(), 0)

    def test_forward_without_labels(self):
        out = self.model(text=self.batch["text"])
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (2, 3))

    def test_attention_sums_to_one_and_ignores_padding(self):
        self.model.eval()
        with torch.no_grad():
            attention = self.model(**self.batch)["attention"]
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(2, 3)))
        padded = attention.masked_select(self.mask[:, None] == 0)
        self.assertTrue(torch.all(padded == 0))

    def test_conv_output_aligned_with_tokens(self):
        for kernel_size in (1, 3, 10):
            with self.subTest(kernel_size=kernel_size):
                model = self.build(kernel_size=kernel_size)
                embeds = model.get_input_embeddings()(self.input_ids)
                hidden = model._encode(embeds, self.mask)
                self.assertEqual(hidden.shape, (*self.input_ids.shape, NUM_FILTERS))

    def test_padding_does_not_change_a_note(self):
        self.model.eval()
        with torch.no_grad():
            alone = self.model(text=[self.batch["text"][1]])["logit"]
            batched = self.model(text=self.batch["text"])["logit"][1:]
        self.assertTrue(torch.allclose(alone, batched, atol=1e-6))

    def test_forward_from_embedding_matches_forward(self):
        self.model.eval()
        out = self.model(**self.batch)
        embeds = self.model.get_input_embeddings()(self.input_ids).detach()
        embeds.requires_grad_()
        out_embed = self.model.forward_from_embedding(embeds, self.mask)
        out_embed["logit"][:, 0].sum().backward()
        self.assertTrue(torch.allclose(out_embed["logit"], out["logit"], atol=1e-6))
        self.assertEqual(embeds.grad.shape, embeds.shape)

    def test_loss_decreases(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        losses = []
        for _ in range(20):
            loss = self.model(**self.batch)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], 0.5 * losses[0])

    def test_attingrad_methods(self):
        self.model.eval()
        for method in ("attention", "inputxgrad", "attingrad"):
            with self.subTest(method=method):
                scores = AttInGrad(self.model, method=method).attribute(
                    **self.batch, target_class_idx=1
                )["text"]
                self.assertEqual(scores.shape, self.mask.shape)
                self.assertTrue(torch.all(scores[self.mask == 0] == 0))
                self.assertTrue(torch.allclose(scores.sum(-1), torch.ones(2)))

    def test_invalid_configurations(self):
        binary_dataset = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "text": "fever", "label": 0},
                {"patient_id": "p1", "text": "sepsis", "label": 1},
            ],
            input_schema={"text": "text"},
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            CAML(binary_dataset, model_name=self.tmp_dir)
        for bad in (
            {"num_filters": 0},
            {"kernel_size": 0},
            {"max_length": -1},
            {"dropout": 1.0},
            {"embed_dim": HIDDEN_DIM + 1},
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                self.build(**bad)


if __name__ == "__main__":
    unittest.main()
