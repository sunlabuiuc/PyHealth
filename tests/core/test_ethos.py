"""Tests for the ETHOS model. Tiny models on synthetic timelines only."""

import inspect
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ETHOS
from pyhealth.models.ethos import _age_quantiles
from pyhealth.tasks import EthosRolloutRule
from pyhealth.trainer import Trainer

START = datetime.fromisoformat("2024-01-01")
BIRTH = datetime.fromisoformat("1980-01-01")
ADMIT = "HOSPITAL_ADMISSION//ER"
HOME = "HOSPITAL_DISCHARGE//HOME"
DEATH = "MEDS_DEATH"
LAB = "LAB//A"
RULE = EthosRolloutRule(
    stop_prefixes=(HOME, DEATH, "TIMELINE_END"),
    positive_prefixes=(DEATH,),
    time_limit_days=1.0,
)


def sample(index, rows):
    """Rows are ``(hours, code, value)``."""
    return {
        "patient_id": f"p{index}",
        "timeline": {
            "time": [START + timedelta(hours=r[0]) for r in rows],
            "code": [r[1] for r in rows],
            "numeric_value": [r[2] for r in rows],
        },
        "context": {
            "time": [BIRTH, None],
            "code": ["MEDS_BIRTH", "GENDER//F" if index % 2 else "GENDER//M"],
            "start": START,
        },
    }


def tiny_dataset(extra=()):
    samples = [
        sample(
            i,
            [
                (0.0, ADMIT, None),
                (1.0, LAB, 1.0 + i),
                (30.0, HOME if i else DEATH, None),
            ],
        )
        for i in range(3)
    ]
    samples.extend(extra)
    return create_sample_dataset(
        samples,
        input_schema={"timeline": "ethos", "context": "ethos_context"},
        output_schema={},
    )


def tiny_model(dataset, **kwargs):
    options = {"n_layer": 1, "n_head": 2, "n_embd": 8, "n_positions": 32, "dropout": 0.0}
    return ETHOS(dataset, **{**options, **kwargs})


def batch_of(dataset, size=3):
    return next(iter(get_dataloader(dataset, batch_size=size)))


def script(model, *tokens):
    """Makes the model sample these tokens in turn, then keep to the last."""
    ids = [model.timeline_processor.vocab[token] for token in tokens]
    steps = iter(ids)

    def logits(prompt):
        token = next(steps, ids[-1])
        forced = F.one_hot(torch.tensor(token), model.vocab_size).float() * 1e4
        return forced.expand(*prompt.shape, -1)

    model._logits = logits


def one_sample(batch):
    time, tokens = batch["timeline"]
    age, static = batch["context"]
    return (time[0], tokens[0]), (age[0], static[0])


class TestArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = tiny_dataset()
        cls.batch = batch_of(cls.dataset)

    def setUp(self):
        torch.manual_seed(0)
        self.model = tiny_model(self.dataset)

    def test_defaults_are_the_papers_final_model(self):
        parameters = inspect.signature(ETHOS).parameters
        defaults = {
            name: parameters[name].default
            for name in ("n_layer", "n_head", "n_embd", "n_positions", "dropout")
        }
        self.assertEqual(
            defaults,
            {
                "n_layer": 6,
                "n_head": 12,
                "n_embd": 768,
                "n_positions": 2048,
                "dropout": 0.3,
            },
        )

    def test_vocabulary_is_a_multiple_of_64(self):
        model = self.model
        self.assertEqual(model.vocab_size % 64, 0)
        self.assertGreaterEqual(
            model.vocab_size,
            model.timeline_processor.size() + model.context_processor.size(),
        )

    def test_weights_are_tied_and_have_no_bias(self):
        self.assertIs(self.model.transformer["wte"].weight, self.model.lm_head.weight)
        self.assertFalse([n for n, _ in self.model.named_parameters() if "bias" in n])

    def test_forward_pass(self):
        output = self.model(**self.batch)
        self.assertTrue(torch.isfinite(output["loss"]))
        logits, labels = output["logit"], output["y_true"]
        self.assertEqual(logits.shape[:2], labels.shape)
        self.assertEqual(logits.size(-1), self.model.vocab_size)
        self.assertIsNone(self.model.mode)

    def test_labels_shift_and_skip_the_context_and_padding(self):
        labels = self.model(**self.batch)["y_true"]
        tokens = self.batch["timeline"][1]
        size = self.model.context_size
        self.assertTrue((labels[:, :size] == -100).all())
        self.assertTrue(torch.equal(labels[:, size:], tokens[:, 1:]))

        short = sample(9, [(0.0, ADMIT, None)])
        dataset = tiny_dataset(extra=[short])
        batch = batch_of(dataset, size=4)
        model = tiny_model(dataset)
        labels = model(**batch)["y_true"]
        self.assertTrue((labels[3, size + 2 :] == -100).all())

    def test_gradients_reach_every_parameter(self):
        self.model(**self.batch)["loss"].backward()
        for name, parameter in self.model.named_parameters():
            if name != "_dummy_param":
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_attention_is_causal(self):
        ids = torch.randint(2, 10, (1, 8))
        changed = ids.clone()
        changed[0, -1] = 11
        self.model.eval()
        before, after = self.model._logits(ids), self.model._logits(changed)
        self.assertTrue(torch.allclose(before[0, :-1], after[0, :-1]))
        self.assertFalse(torch.allclose(before[0, -1], after[0, -1]))

    def test_batch_without_any_target_has_zero_loss(self):
        time, tokens = self.batch["timeline"]
        empty = {
            "timeline": (time[:, :0], tokens[:, :0]),
            "context": self.batch["context"],
        }
        loss = self.model(**empty)["loss"]
        self.assertEqual(loss.item(), 0.0)
        loss.backward()

    def test_rejects_datasets_without_ethos_fields(self):
        dataset = create_sample_dataset(
            [{"patient_id": "p", "codes": ["a"]}],
            input_schema={"codes": "sequence"},
            output_schema={},
        )
        with self.assertRaises(ValueError):
            tiny_model(dataset)

    def test_has_no_mode_and_rejects_labels(self):
        self.assertIsNone(self.model.mode)
        samples = [
            {**sample(i, [(0.0, ADMIT, None), (1.0, LAB, 1.0)]), "label": i % 2}
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples,
            input_schema={"timeline": "ethos", "context": "ethos_context"},
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            tiny_model(dataset)

    def test_rejects_sizes_that_do_not_fit(self):
        with self.assertRaises(ValueError):
            tiny_model(self.dataset, n_head=3)
        with self.assertRaises(ValueError):
            tiny_model(self.dataset, n_positions=5)


class TestContext(unittest.TestCase):
    def test_age_digits(self):
        age = torch.tensor([46.0, 0.0, 99.0, 150.0, -3.0, 45.97, math.nan])
        high, low = _age_quantiles(age, 10)
        self.assertEqual(high.tolist(), [4, 0, 9, 9, 0, 4, -1])
        self.assertEqual(low.tolist(), [6, 0, 9, 9, 0, 6, -1])

    def test_age_digits_carry(self):
        high, low = _age_quantiles(torch.tensor([49.7]), 10)
        self.assertEqual((high.item(), low.item()), (5, 0))

    def test_context_ids(self):
        dataset = tiny_dataset()
        model = tiny_model(dataset)
        batch = batch_of(dataset)
        _, static = batch["context"]
        ids = model._context_ids(static, torch.tensor([46.0, 46.0, math.nan]))
        vocab = model.timeline_processor.vocab
        position = model.context_processor.age_position
        self.assertEqual(ids[0, position].item(), vocab["Q5"])
        self.assertEqual(ids[0, position + 1].item(), vocab["Q7"])
        self.assertEqual(ids[2, position].item(), vocab.unk_id)
        others = ids[0, : position]
        self.assertTrue((others >= model.timeline_vocab_size).all())

    def test_long_timeline_uses_a_window_and_the_age_where_it_starts(self):
        rows = [(float(i * 24 * 365), LAB, None) for i in range(40)]
        dataset = tiny_dataset(extra=[sample(5, rows)])
        model = tiny_model(dataset)
        batch = batch_of(dataset, size=4)
        model.eval()
        window = model.timeline_size
        output = model(**batch)
        self.assertEqual(output["logit"].size(1), model.n_positions)

        captured = {}
        original = model._context_ids

        def spy(static, age):
            captured["age"] = age
            return original(static, age)

        model._context_ids = spy
        model(**batch)
        time, tokens = batch["timeline"]
        start = (tokens[3] != 0).sum() - (window + 1)
        self.assertGreater(start.item(), 0)
        expected = (START - BIRTH).days / 365.25
        expected += time[3, start].item() / (24 * 365.25)
        self.assertAlmostEqual(captured["age"][3].item(), expected, places=3)

    def test_training_windows_are_random_and_stay_in_the_timeline(self):
        rows = [(float(i), LAB, 1.0) for i in range(60)]
        dataset = tiny_dataset(extra=[sample(5, rows)])
        model = tiny_model(dataset)
        batch = batch_of(dataset, size=4)
        model.train()
        ages = set()
        original = model._context_ids

        def spy(static, age):
            ages.add(round(age[3].item(), 6))
            return original(static, age)

        model._context_ids = spy
        for _ in range(8):
            self.assertTrue(torch.isfinite(model(**batch)["loss"]))
        self.assertGreater(len(ages), 1)


class TestRollout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = tiny_dataset()
        cls.batch = batch_of(cls.dataset)

    def setUp(self):
        torch.manual_seed(0)
        self.model = tiny_model(self.dataset)

    def rollout(self, **kwargs):
        options = {"num_rollouts": 2, "max_new_tokens": 6, **kwargs}
        return self.model.rollout(*one_sample(self.batch), RULE, **options)

    def test_seeded_rollout_is_reproducible(self):
        def run(seed):
            generator = torch.Generator().manual_seed(seed)
            return self.rollout(generator=generator, num_rollouts=8)

        first, second, other = run(1), run(1), run(2)
        self.assertEqual(first, second)
        self.assertNotEqual(first["tokens"], other["tokens"])

    def test_stops_on_a_stop_token(self):
        script(self.model, LAB, DEATH)
        result = self.rollout()
        self.assertEqual(result["reason"], ["token_of_interest"] * 2)
        self.assertEqual(result["actual"], [DEATH] * 2)
        self.assertEqual(len(result["tokens"][0]), 2)

    def test_interval_tokens_advance_the_clock_up_to_the_limit(self):
        script(self.model, "12h-18h")
        hours = self.model.timeline_processor.interval_estimates["12h-18h"]
        result = self.rollout(max_new_tokens=8)
        self.assertEqual(result["reason"], ["time_limit"] * 2)
        steps = len(result["tokens"][0])
        self.assertEqual(steps, math.floor(24.0 / hours) + 1)
        self.assertAlmostEqual(result["days"][0], steps * hours / 24.0)

    def test_tokens_that_are_not_timeline_tokens_are_invalid(self):
        self.model._logits = lambda ids: (
            F.one_hot(
                torch.tensor(self.model.vocab_size - 1), self.model.vocab_size
            ).float()
            * 1e4
        ).expand(*ids.shape, -1)
        result = self.rollout()
        self.assertEqual(result["reason"], ["key_error"] * 2)
        self.assertEqual(result["actual"], [None] * 2)

    def test_gives_up_after_max_new_tokens(self):
        script(self.model, LAB)
        result = self.rollout(max_new_tokens=3)
        self.assertEqual(result["reason"], ["token_limit"] * 2)
        self.assertEqual(len(result["tokens"][0]), 3)

    def test_rollouts_stop_separately(self):
        vocab = self.model.timeline_processor.vocab
        calls = iter([[LAB, DEATH], [HOME]])

        def logits(ids):
            tokens = next(calls)
            forced = torch.zeros(len(tokens), 1, self.model.vocab_size)
            for row, token in enumerate(tokens):
                forced[row, 0, vocab[token]] = 1e4
            return forced

        self.model._logits = logits
        result = self.rollout()
        self.assertEqual(result["actual"], [HOME, DEATH])
        self.assertEqual([len(t) for t in result["tokens"]], [2, 1])

    def test_a_full_context_slides_and_keeps_the_static_tokens(self):
        seen = []

        def logits(ids):
            seen.append(ids.clone())
            forced = torch.zeros(*ids.shape, self.model.vocab_size)
            forced[..., self.model.timeline_processor.vocab[LAB]] = 1e4
            return forced

        self.model._logits = logits
        self.rollout(max_new_tokens=40, num_rollouts=1)
        size = self.model.context_size
        self.assertEqual(max(ids.size(1) for ids in seen), self.model.n_positions)
        self.assertTrue(torch.equal(seen[-1][:, :size], seen[0][:, :size]))

    def test_prompt_leaves_out_timeline_end(self):
        (time, tokens), (age, static) = one_sample(self.batch)
        vocab = self.model.timeline_processor.vocab
        self.assertEqual(tokens[tokens != 0][-1].item(), vocab["TIMELINE_END"])
        prompt = self.model._prompt(time, tokens, age, static)
        self.assertNotEqual(prompt[0, -1].item(), vocab["TIMELINE_END"])

    def test_restores_training_mode(self):
        self.model.train()
        self.rollout()
        self.assertTrue(self.model.training)


class TestEstimateRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = tiny_dataset()
        cls.batch = batch_of(cls.dataset)

    def setUp(self):
        torch.manual_seed(0)
        self.model = tiny_model(self.dataset)

    def test_outcome_data_must_use_the_models_processors(self):
        script(self.model, DEATH)
        new_codes = [(0.5 + k, f"LAB//NEW{k}", None) for k in range(8)]
        refitted = tiny_dataset(extra=[sample(9, [(0.0, ADMIT, None), *new_codes])])
        with self.assertRaisesRegex(ValueError, "input_processors"):
            self.model.estimate_risk(get_dataloader(refitted, batch_size=3), RULE,
                                     num_rollouts=1)
        # a plain list of batches has no dataset to compare; out-of-vocabulary
        # ids still raise
        with self.assertRaisesRegex(ValueError, "outside this model's vocabulary"):
            self.model.estimate_risk([batch_of(refitted, size=4)], RULE,
                                     num_rollouts=1)
        shared = create_sample_dataset(
            [sample(i, [(0.0, ADMIT, None), (30.0, HOME, None)]) for i in range(2)],
            input_schema={"timeline": "ethos", "context": "ethos_context"},
            output_schema={},
            input_processors=self.dataset.input_processors,
        )
        risk = self.model.estimate_risk(get_dataloader(shared, batch_size=2), RULE,
                                        num_rollouts=1)
        self.assertEqual(len(risk["y_prob"]), 2)

    def test_all_positive(self):
        script(self.model, DEATH)
        risk = self.model.estimate_risk([self.batch], RULE, num_rollouts=2)
        self.assertEqual(risk["y_prob"].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(risk["num_kept"].tolist(), [2, 2, 2])
        self.assertEqual(sorted(risk["patient_id"]), ["p0", "p1", "p2"])

    def test_all_negative(self):
        script(self.model, HOME)
        risk = self.model.estimate_risk([self.batch], RULE, num_rollouts=2)
        self.assertEqual(risk["y_prob"].tolist(), [0.0, 0.0, 0.0])

    def test_all_dropped_is_nan(self):
        script(self.model, LAB)
        risk = self.model.estimate_risk(
            [self.batch], RULE, num_rollouts=2, max_new_tokens=2
        )
        self.assertTrue(risk["y_prob"].isnan().all())
        self.assertEqual(risk["num_kept"].tolist(), [0, 0, 0])

    def test_seed_makes_it_reproducible(self):
        options = {"num_rollouts": 4, "max_new_tokens": 8, "seed": 3}
        first = self.model.estimate_risk([self.batch], RULE, **options)
        second = self.model.estimate_risk([self.batch], RULE, **options)
        self.assertTrue(first["y_prob"].nan_to_num(-1).equal(second["y_prob"].nan_to_num(-1)))

    def test_label_key(self):
        script(self.model, DEATH)
        batch = {**self.batch, "outcome": torch.tensor([[1.0], [0.0], [1.0]])}
        risk = self.model.estimate_risk(
            [batch], RULE, num_rollouts=1, label_key="outcome"
        )
        self.assertEqual(risk["y_true"].tolist(), [1.0, 0.0, 1.0])


class TestTraining(unittest.TestCase):
    def test_trainer_and_checkpoint_round_trip(self):
        dataset = tiny_dataset()
        loader = get_dataloader(dataset, batch_size=3)
        torch.manual_seed(0)
        model = tiny_model(dataset, n_embd=16)
        trainer = Trainer(model=model, enable_logging=False)
        batch = next(iter(loader))
        before = model(**batch)["loss"].item()
        trainer.train(
            train_dataloader=loader,
            epochs=6,
            optimizer_params={"lr": 0.05},
            max_grad_norm=1.0,
        )
        model.eval()
        self.assertLess(model(**batch)["loss"].item(), before)
        self.assertIn("loss", trainer.evaluate(loader))

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ethos.ckpt")
            trainer.save_ckpt(path)
            restored = tiny_model(dataset, n_embd=16)
            Trainer(model=restored, enable_logging=False).load_ckpt(path)
        restored.eval()
        self.assertTrue(
            torch.allclose(model(**batch)["logit"], restored(**batch)["logit"])
        )
        options = {"num_rollouts": 2, "max_new_tokens": 8, "seed": 1}
        self.assertEqual(
            model.estimate_risk([batch], RULE, **options)["num_kept"].tolist(),
            restored.estimate_risk([batch], RULE, **options)["num_kept"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
