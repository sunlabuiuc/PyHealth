"""Unit tests for the AttInGrad interpreter.

Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
Explainability in Healthcare Records", EMNLP 2024.

Uses the same two synthetic notes and tiny local BERT as test_plm_ca, so
nothing is downloaded.
"""

import shutil
import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import AttInGrad
from pyhealth.interpret.methods.attingrad import METHODS
from pyhealth.models import PLMCA
from tests.core.test_plm_ca import (
    CHUNK_SIZE,
    HIDDEN_DIM,
    MAX_LENGTH,
    SAMPLES,
    VOCAB,
    save_tiny_tokenizer,
)


class TestAttInGrad(unittest.TestCase):
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
            dataset_name="test_attingrad",
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
        self.model.eval()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=2)))
        _, self.mask = self.model.tokenize(self.batch["text"])

    def explain(self, method, target=1):
        return AttInGrad(self.model, method=method).attribute(
            **self.batch, target_class_idx=target
        )["text"]

    def test_shape_padding_and_normalization(self):
        for method in METHODS:
            with self.subTest(method=method):
                scores = self.explain(method)
                self.assertEqual(scores.shape, self.mask.shape)
                self.assertTrue(torch.all(scores[self.mask == 0] == 0))
                self.assertTrue(torch.all(scores >= 0))
                self.assertTrue(torch.allclose(scores.sum(-1), torch.ones(2)))

    def test_attingrad_is_product_of_parts(self):
        attention = self.explain("attention")
        input_x_grad = self.explain("inputxgrad")
        product = attention * input_x_grad
        expected = product / product.sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(self.explain("attingrad"), expected, atol=1e-6))

    def test_att_ig_is_product_of_parts(self):
        product = self.explain("attention") * self.explain("integrated_gradients")
        expected = product / product.sum(-1, keepdim=True)
        self.assertTrue(torch.allclose(self.explain("att_ig"), expected, atol=1e-6))

    def test_integrated_gradients_completeness(self):
        interpreter = AttInGrad(self.model, method="integrated_gradients", steps=64)
        input_ids, mask = self.model.tokenize(self.batch["text"])
        embed = self.model.get_input_embeddings()
        x = embed(input_ids).detach()
        baseline_ids = interpreter._baseline_ids(input_ids, mask)
        x0 = embed(baseline_ids).detach()
        targets = torch.tensor([1, 1])
        raw = interpreter.integrated_gradients(x, x0, mask, targets)
        with torch.no_grad():
            gap = (
                self.model.forward_from_embedding(x, mask)["logit"][:, 1]
                - self.model.forward_from_embedding(x0, mask)["logit"][:, 1]
            )
        self.assertTrue(torch.allclose(raw.sum(dim=(1, 2)), gap, atol=1e-2))
        # Special tokens and padding keep their ids in the baseline.
        changed = baseline_ids != input_ids
        self.assertFalse(changed[:, 0].any())
        self.assertFalse(changed[mask == 0].any())
        self.assertTrue(changed[mask == 1].any())

    def test_inputxgrad_matches_manual_gradient(self):
        input_ids, mask = self.model.tokenize(self.batch["text"][:1])
        embeds = self.model.get_input_embeddings()(input_ids).detach()
        embeds.requires_grad_(True)
        logit = self.model.forward_from_embedding(embeds, mask)["logit"][0, 1]
        (grad,) = torch.autograd.grad(logit, embeds)
        manual = (embeds.detach() * grad).norm(dim=-1)[0] * mask[0]
        manual = manual / manual.sum()

        single = {"text": self.batch["text"][:1]}
        scores = AttInGrad(self.model, method="inputxgrad").attribute(
            **single, target_class_idx=1
        )["text"][0]
        self.assertTrue(torch.allclose(scores, manual, atol=1e-6))

    def test_target_changes_explanation(self):
        self.assertFalse(
            torch.allclose(self.explain("attingrad", 0), self.explain("attingrad", 2))
        )

    def test_default_target_is_top_logit(self):
        logits = self.model(**self.batch)["logit"]
        top = logits.argmax(-1)
        default = AttInGrad(self.model).attribute(**self.batch)["text"]
        for row, target in enumerate(top.tolist()):
            chosen = self.explain("attingrad", target)[row]
            self.assertTrue(torch.allclose(default[row], chosen, atol=1e-6))

    def test_works_under_no_grad(self):
        with torch.no_grad():
            scores = self.explain("attingrad")
        self.assertTrue(torch.allclose(scores.sum(-1), torch.ones(2)))

    def test_rejects_bad_method_and_model(self):
        with self.assertRaises(ValueError):
            AttInGrad(self.model, method="saliency")
        with self.assertRaises(ValueError):
            AttInGrad(torch.nn.Linear(2, 2))


if __name__ == "__main__":
    unittest.main()
