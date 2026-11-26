"""Unit tests for the AttInGrad interpreter.

Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
Explainability in Healthcare Records", EMNLP 2024.

Uses the same two synthetic notes and tiny local BERT as test_plm_ca, so
nothing is downloaded.

Contributors: Juri Moon (jmoon326)
Paper link: https://arxiv.org/abs/2406.08958
"""

import shutil
import tempfile
import unittest

import torch

from pyhealth.datasets import get_dataloader
from pyhealth.interpret.methods import AttInGrad
from pyhealth.interpret.methods.attingrad import METHODS
from pyhealth.models import PLMCA
from tests.core.test_plm_ca import (
    CHUNK_SIZE,
    MAX_LENGTH,
    save_tiny_encoder,
    tiny_dataset,
)


class TinyModel(torch.nn.Module):
    """Embedding -> Linear -> activation -> mean over tokens -> Linear logits."""

    def __init__(self, activation):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(5, 4)
        self.hidden = torch.nn.Linear(4, 3)
        self.activation = activation
        self.out = torch.nn.Linear(3, 2)

    def tokenize(self, texts):
        raise NotImplementedError

    def get_input_embeddings(self):
        return self.embed

    def forward_from_embedding(self, inputs_embeds, attention_mask):
        h = self.activation(self.hidden(inputs_embeds)).mean(dim=1)
        return {"logit": self.out(h)}


class TestDeepLiftByHand(unittest.TestCase):
    """DeepLIFT on tiny models where the answer can be checked by hand."""

    def setUp(self):
        self.ids = torch.tensor([[1, 2, 3, 4]])
        self.mask = torch.ones_like(self.ids)
        self.targets = torch.tensor([1])

    def run_deeplift(self, model):
        x = model.embed(self.ids).detach()
        x0 = model.embed(torch.zeros_like(self.ids)).detach()
        raw = AttInGrad(model, method="deeplift").deeplift(
            x, x0, self.mask, self.targets
        )
        gap = (
            model.forward_from_embedding(x, self.mask)["logit"][0, 1]
            - model.forward_from_embedding(x0, self.mask)["logit"][0, 1]
        )
        return x, x0, raw, gap

    def test_linear_model_is_input_minus_baseline_times_gradient(self):
        model = TinyModel(torch.nn.Identity())
        x, x0, raw, gap = self.run_deeplift(model)
        # logit_1 = out_1 . (W x_n + b) / N, so dlogit/dx_n = W^T out_1 / N.
        grad = (model.hidden.weight.T @ model.out.weight[1]) / x.shape[1]
        self.assertTrue(torch.allclose(raw, (x - x0) * grad, atol=1e-6))
        self.assertTrue(torch.allclose(raw.sum(), gap, atol=1e-5))

    def test_tanh_module_gets_rescale_rule(self):
        model = TinyModel(torch.nn.Tanh())
        x, x0, raw, gap = self.run_deeplift(model)
        # Rescale: multiplier (tanh(z) - tanh(z0)) / (z - z0) per unit.
        z, z0 = model.hidden(x), model.hidden(x0)
        secant = (torch.tanh(z) - torch.tanh(z0)) / (z - z0)
        upstream = model.out.weight[1] / x.shape[1]
        expected = (x - x0) * ((secant * upstream) @ model.hidden.weight)
        self.assertTrue(torch.allclose(raw, expected, atol=1e-5))
        # Exact completeness, which the plain gradient at x does not give.
        self.assertTrue(torch.allclose(raw.sum(), gap, atol=1e-5))
        self.assertIsInstance(model.activation, torch.nn.Tanh)  # restored


class TestAttInGrad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        save_tiny_encoder(cls.tmp_dir)
        cls.dataset = tiny_dataset("test_attingrad")

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

    def test_att_deeplift_is_product_of_parts(self):
        product = self.explain("attention") * self.explain("deeplift")
        expected = product / product.sum(-1, keepdim=True)
        got = self.explain("att_deeplift")
        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_deeplift_on_plm_ca_uses_linear_rule(self):
        # PLMCA has no hooked activation on the logit path (GELU and the
        # softmaxes use the linear rule), so DeepLIFT is (x - x0) * grad(x).
        interpreter = AttInGrad(self.model, method="deeplift")
        input_ids, mask = self.model.tokenize(self.batch["text"])
        embed = self.model.get_input_embeddings()
        x = embed(input_ids).detach()
        x0 = embed(interpreter._baseline_ids(input_ids, mask)).detach()
        targets = torch.tensor([1, 1])
        _, grad = interpreter._logit_gradient(x, mask, targets)
        raw = interpreter.deeplift(x, x0, mask, targets)
        self.assertTrue(torch.allclose(raw, (x - x0) * grad, atol=1e-6))

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
        with self.assertRaises(TypeError):
            AttInGrad(torch.nn.Linear(2, 2))


if __name__ == "__main__":
    unittest.main()
