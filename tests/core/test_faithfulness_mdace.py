"""Unit tests for faithfulness metrics (Edin et al., EMNLP 2024).

Contributors: Juri Moon (jmoon326)
Paper: Edin et al., "An Unsupervised Approach to Achieve Supervised-Level
    Explainability in Healthcare Records", EMNLP 2024.
Paper link: https://arxiv.org/abs/2406.08958
"""

import ast
import math
import re
import unittest
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pyhealth.metrics.interpretability import (
    faithfulness,
    perturbation_curve,
    roberta_word_map,
    word_feature_map,
)

SOS, PAD, EOS, MASK = 0, 1, 2, 3  # RoBERTa ids, as the authors' code assumes.
AUTHORS = (
    Path(__file__).resolve().parents[3]
    / "explainable-medical-coding"
    / "explainable_medical_coding"
)


class CountModel(torch.nn.Module):
    """logit_j = bias_j + sum over unmasked positions of weight[token, j]."""

    def __init__(self, weight, bias):
        super().__init__()
        vocab = weight.shape[0]
        self.embedding = torch.nn.Embedding.from_pretrained(torch.eye(vocab))
        self.weight = torch.nn.Parameter(torch.as_tensor(weight, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.as_tensor(bias, dtype=torch.float))

    def get_input_embeddings(self):
        return self.embedding

    def forward_from_embedding(self, inputs_embeds, attention_mask):
        x = inputs_embeds * attention_mask[..., None]
        logit = x.sum(1) @ self.weight + self.bias
        return {"logit": logit, "y_prob": torch.sigmoid(logit)}

    def forward(self, input_ids, attention_mask):  # authors' interface
        emb = self.get_input_embeddings()(input_ids)
        return self.forward_from_embedding(emb, attention_mask.float())["logit"]


def one_token_model(vocab=12, token=7, strength=4.0, bias=-1.0):
    weight = torch.zeros(vocab, 1)
    weight[token, 0] = strength
    return CountModel(weight, [bias])


def sig(x):
    return 1 / (1 + math.exp(-x))


class TestHandComputed(unittest.TestCase):
    # <s> 5 7 6 8 9 </s>: five features, only token 7 moves the logit.
    ids = torch.tensor([SOS, 5, 7, 6, 8, 9, EOS])
    mask = torch.ones(7)

    def run_pair(self, attributions, **kw):
        return faithfulness(
            one_token_model(), self.ids, self.mask, attributions, 0, MASK, **kw
        )

    def test_good_explanation(self):
        out = self.run_pair([0.1, 0.9, 0.2, 0.3, 0.05])
        p, p0 = sig(3.0), sig(-1.0)
        drop = (p - p0) / p
        self.assertAlmostEqual(out["y_prob"], p, places=6)
        # 50 steps removing 0, 2, 4, ... top features: only step 0 keeps 7.
        self.assertAlmostEqual(out["comprehensiveness"], 49 / 50 * drop, places=6)
        # 5 < K: steps remove the 0, 2, 4, 5, ... lowest; 7 goes at step 3.
        self.assertAlmostEqual(out["sufficiency"], 47 / 50 * drop, places=6)

    def test_bad_explanation(self):
        out = self.run_pair([0.5, 0.01, 0.2, 0.3, 0.4])
        drop = (sig(3.0) - sig(-1.0)) / sig(3.0)
        self.assertAlmostEqual(out["comprehensiveness"], 47 / 50 * drop, places=6)
        self.assertAlmostEqual(out["sufficiency"], 49 / 50 * drop, places=6)

    def test_step_one_all_features(self):
        out = self.run_pair([0.1, 0.9, 0.2, 0.3, 0.05], step_size=1, max_features=None)
        drop = (sig(3.0) - sig(-1.0)) / sig(3.0)
        self.assertAlmostEqual(out["comprehensiveness"], 4 / 5 * drop, places=6)
        # Lowest-first removal: 7 is removed only at step 5, which never runs.
        self.assertAlmostEqual(out["sufficiency"], 0.0, places=6)

    def test_sufficiency_clips_increases(self):
        model = one_token_model(strength=-4.0, bias=1.0)  # token 7 lowers p
        scores = [0.1, 0.9, 0.2, 0.3, 0.05]
        out = faithfulness(model, self.ids, self.mask, scores, 0, MASK)
        self.assertEqual(out["sufficiency"], 0.0)
        self.assertLess(out["comprehensiveness"], 0.0)  # not clipped

    def test_padding_and_specials_untouched(self):
        attributions = [0.1, 0.9, 0.2, 0.3, 0.05]
        padded = torch.cat([self.ids, torch.full((5,), PAD)])
        mask = torch.cat([self.mask, torch.zeros(5)])
        a = self.run_pair(attributions)
        b = faithfulness(one_token_model(), padded, mask, attributions, 0, MASK)
        self.assertEqual(a, b)

    def test_word_level(self):
        # 7 and 6 form one word; its summed score (0.4) ranks it first.
        _, deltas = perturbation_curve(
            one_token_model(), self.ids, self.mask, [0.35, 0.2, 0.2, 0.3, 0.1],
            0, MASK, feature_map=torch.tensor([0, 1, 1, 2, 3]),
            step_size=1, max_features=None,
        )
        self.assertEqual(len(deltas), 4)
        self.assertAlmostEqual(deltas[0], 0.0)
        self.assertGreater(deltas[1], 0.5)

    def test_word_map(self):
        tokens = ["Pat", "ient", "Ġhas", "Ġ", "Ġheart", ",", "Ġfailure", "Ċ"]
        self.assertEqual(word_feature_map(tokens).tolist(), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(
            roberta_word_map(["<s>", "Ġa", "b", "</s>"]).tolist(), [0, 2, 2, 3]
        )


def load_authors():
    """Exec the authors' functions (extracted with ast) without their deps."""
    wanted = {
        "eval/faithfulness_metrics.py": {"pertubation_score"},
        "explainability/helper_functions.py": {
            "PermuteDataset", "get_feature_attributions", "create_attention_mask"
        },
        "utils/tokenizer.py": {"get_word_map_roberta", "get_tokens"},
    }
    namespace = {"np": np, "torch": torch, "math": math, "re": re,
                 "Optional": Optional, "RobertaTokenizer": object,
                 "PreTrainedTokenizerBase": object}
    for rel, names in wanted.items():
        tree = ast.parse((AUTHORS / rel).read_text())
        nodes = [n for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), rel, "exec"), namespace)
    return namespace


# Skipped in CI. To run locally, clone the authors' code next to the PyHealth
# checkout, so both folders share a parent:
#   git clone https://github.com/JoakimEdin/explainable-medical-coding ../explainable-medical-coding
# then, from PyHealth/:
#   python -m pytest tests/core/test_faithfulness_mdace.py -k Authors
@unittest.skipUnless(AUTHORS.exists(), "authors' repository not found")
class TestMatchesAuthors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_authors()

    def compare(self, n_tokens, step_size, max_features, word_level, seed):
        gen = torch.Generator().manual_seed(seed)
        vocab, classes = 15, 3
        model = CountModel(torch.randn(vocab, classes, generator=gen),
                           torch.randn(classes, generator=gen)).eval()
        body = torch.randint(4, vocab, (n_tokens,), generator=gen)
        ids = torch.cat([torch.tensor([SOS]), body, torch.tensor([EOS])])
        attributions = torch.rand(n_tokens + 2, classes, generator=gen)  # no ties

        feature_map, ours_map = None, None
        if word_level:
            names = [
                "Ġw" if t % 3 == 0 else ("Ġ" if t == 4 else "x")
                for t in body.tolist()
            ]
            tokens = ["<s>", *names, "</s>"]

            class Tok:
                def convert_ids_to_tokens(self, ids):
                    return tokens

            feature_map = self.ns["get_word_map_roberta"](ids[None], Tok())
            ours_map = word_feature_map(names)
        for descending in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # CUDA autocast off on CPU
                theirs = self.ns["pertubation_score"](
                    model, ids[None], np.arange(classes), attributions,
                    feature_map=feature_map, baseline_token_id=MASK,
                    step_size=step_size, descending=descending,
                    max_features=max_features,
                )
            for c in range(classes):
                _, ours = perturbation_curve(
                    model, ids, torch.ones(len(ids)), attributions[1:-1, c], c, MASK,
                    feature_map=ours_map, descending=descending,
                    step_size=step_size, max_features=max_features,
                )
                np.testing.assert_allclose(ours, theirs[:, c], atol=1e-6)

    def test_token_level(self):
        self.compare(30, 2, 100, False, 0)  # fewer features than K
        self.compare(250, 2, 100, False, 1)  # more features than K
        self.compare(41, 3, 10, False, 2)
        self.compare(25, 1, None, False, 3)

    def test_word_level(self):
        self.compare(60, 2, 100, True, 4)
        self.compare(300, 2, 100, True, 5)
        self.compare(40, 2, 5, True, 6)


if __name__ == "__main__":
    unittest.main()
