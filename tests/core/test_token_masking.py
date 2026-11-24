"""Unit tests for token-masking (TM) training.

Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
Explainability in Healthcare Records", EMNLP 2024.

Uses the tiny tokenizer and BERT from test_plm_ca, so nothing is downloaded.
"""

import shutil
import tempfile
import unittest

import torch
from torch import nn
from transformers import AutoTokenizer, BertConfig, BertModel

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import optimize_token_mask, token_masking_loss
from pyhealth.models import CAML, PLMCA
from tests.core.test_plm_ca import HIDDEN_DIM, SAMPLES, VOCAB, save_tiny_tokenizer


class OneTokenModel(nn.Module):
    """Predicts one code from the word "sepsis" alone; other tokens embed to 0."""

    def __init__(self, folder: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.embedding = nn.Embedding(len(VOCAB), 1)
        nn.init.zeros_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[VOCAB.index("sepsis")] = 1.0

    def get_input_embeddings(self):
        return self.embedding

    def forward_from_embedding(self, inputs_embeds, attention_mask):
        evidence = (inputs_embeds[..., 0] * attention_mask).sum(1, keepdim=True)
        return {"logit": 10 * evidence - 5}

    def tokenize(self, texts):
        encoded = self.tokenizer(texts, padding=True, return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]


class TestTokenMasking(unittest.TestCase):
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
            dataset_name="test_token_masking",
        )
        cls.texts = next(iter(get_dataloader(cls.dataset, batch_size=2)))["text"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def toy_inputs(self):
        model = OneTokenModel(self.tmp_dir)
        ids, mask = model.tokenize(["fever chest sepsis pain acute", "sepsis fever"])
        return model, ids, mask

    def test_mask_keeps_only_the_evidence_token(self):
        model, ids, mask = self.toy_inputs()
        baseline = model.embedding(torch.full_like(ids, model.tokenizer.mask_token_id))
        soft, iterations = optimize_token_mask(
            model, ids, mask, baseline, diet_constant=1.0, lr=0.5, max_iter=20
        )
        self.assertEqual(soft.shape, ids.shape)
        self.assertTrue(1 <= iterations <= 20)
        sepsis = ids == VOCAB.index("sepsis")
        self.assertTrue((soft[sepsis] > 0.5).all())
        self.assertTrue((soft[~sepsis] < 0.5).all())
        # The mask shrank from its start, sigmoid(2), everywhere but sepsis.
        self.assertTrue((soft[~sepsis] < torch.sigmoid(torch.tensor(2.0))).all())

    def test_authors_settings_stop_early_and_min_iter_fixes_it(self):
        # With the authors' settings (lr 0.1, start 2, 3% check two steps
        # back) the loop stops at update 3 and every mask value is still
        # above 0.5, so rounding keeps every token. min_iter lets it run on.
        model, ids, mask = self.toy_inputs()
        baseline = model.embedding(torch.full_like(ids, model.tokenizer.mask_token_id))
        soft, iterations = optimize_token_mask(
            model, ids, mask, baseline, diet_constant=0.01, lr=0.1, max_iter=20
        )
        self.assertEqual(iterations, 3)
        self.assertTrue((soft > 0.5).all())
        soft, iterations = optimize_token_mask(
            model,
            ids,
            mask,
            baseline,
            diet_constant=0.01,
            lr=1.0,
            max_iter=20,
            min_iter=20,
        )
        self.assertEqual(iterations, 20)
        self.assertTrue((soft < 0.5).any())

    def test_special_tokens_and_padding_never_masked(self):
        model, ids, mask = self.toy_inputs()
        teacher = model.forward_from_embedding(model.embedding(ids), mask)["logit"]
        out = token_masking_loss(
            model, ids, mask, teacher, diet_constant=1.0, lr=0.5, max_iter=20
        )
        special = torch.isin(ids, torch.tensor(model.tokenizer.all_special_ids))
        self.assertTrue((mask == 0).any())
        self.assertTrue((out["mask"][special | (mask == 0)] == 1).all())
        self.assertTrue((out["mask"][ids == VOCAB.index("sepsis")] == 1).all())
        self.assertTrue((out["mask"][ids == VOCAB.index("fever")] == 0).all())

    def check_real_model(self, model):
        ids, mask = model.tokenize(self.texts)
        model.train()
        with torch.no_grad():
            embeds = model.get_input_embeddings()(ids)
            teacher = model.forward_from_embedding(embeds, mask)["logit"] + 0.5

        # The inner optimisation leaves the model's gradients alone.
        embed = model.get_input_embeddings()
        baseline = embed(torch.full_like(ids, model.tokenizer.mask_token_id))
        optimize_token_mask(model, ids, mask, baseline.detach())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(model.training)

        out = token_masking_loss(model, ids, mask, teacher, lambda_1=2.0, lr=1.0)
        masked_ids = torch.where(
            out["mask"].bool(), ids, model.tokenizer.mask_token_id
        )
        with torch.no_grad():
            logits = model.forward_from_embedding(embed(ids), mask)["logit"]
            probs = torch.sigmoid(logits)
            masked = torch.sigmoid(
                model.forward_from_embedding(embed(masked_ids), mask)["logit"]
            )
        distill = (probs - torch.sigmoid(teacher)).abs().sum() / 2
        masked_term = (probs - masked).abs().sum() / 2
        # On Linux CPU, SDPA attention picks a different kernel with and without
        # autograd, so the no_grad recompute can differ by ~1e-4. A wrong
        # lambda_1 would move the loss by masked_term (~4e-2), far above 1e-3.
        self.assertTrue(
            torch.allclose(out["loss"], distill + 2.0 * masked_term, atol=1e-3)
        )
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_plm_ca(self):
        torch.manual_seed(1)
        model = PLMCA(
            self.dataset,
            model_name=self.tmp_dir,
            chunk_size=8,
            max_length=32,
            dropout=0.0,
        )
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.check_real_model(model)

    def test_caml(self):
        torch.manual_seed(1)
        model = CAML(
            self.dataset, model_name=self.tmp_dir, max_length=32, num_filters=6,
            dropout=0.0,
        )
        self.check_real_model(model)


if __name__ == "__main__":
    unittest.main()
