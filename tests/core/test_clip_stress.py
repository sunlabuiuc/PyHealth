"""Offline tests for CLIPStress and feature-only prompt preprocessing.

Contributor: Siqi Zhu (UIUC NetID: siqizhu4).
Paper: Contrastive Pretraining for Stress Detection with Multimodal Wearable
Sensor Data and Surveys (CHIL 2025).
https://proceedings.mlr.press/v287/yang25a.html
"""

import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CLIPStress
from pyhealth.models.clip_stress_utils import build_stress_prompt, encode_stress_prompts
from pyhealth.trainer import Trainer


class TestCLIPStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.samples = [
            {
                "patient_id": f"p{i}",
                "signal": torch.randn(4, 2),
                "tabular": torch.randn(3),
                "text_embedding": torch.randn(5),
                "mask": torch.tensor([True, True, True, False]),
                "label": i % 2,
            }
            for i in range(4)
        ]
        self.schema = {
            k: "tensor" for k in ("signal", "tabular", "text_embedding", "mask")
        }
        self.dataset = create_sample_dataset(
            self.samples, self.schema, {"label": "binary"}
        )
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=4)))

    def model(self, **kwargs):
        config = {
            "hidden_dim": 8,
            "projection_dim": 6,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
        }
        config.update(kwargs)
        return CLIPStress(self.dataset, **config)

    def test_contrastive_objective_and_gradients(self):
        model = self.model()
        batch = {k: v for k, v in self.batch.items() if k != "label"}
        result = model(**batch)
        self.assertIsNone(model.mode)
        self.assertEqual(result["logit"].shape, (4, 4))
        self.assertTrue(
            torch.all(result["logit"].abs() <= 1 / model.temperature + 1e-5)
        )
        logit = result["logit"]
        expected = (logit.logsumexp(1).mean() + logit.logsumexp(0).mean()) / 2
        expected -= logit.diag().mean()
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        for module in (
            model.encoders,
            model.fusion,
            model.signal_projection,
            model.text_projection,
        ):
            grads = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in grads)
            )
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.assertTrue(all(p.grad is None for p in model.classifier.parameters()))
        changed = dict(batch, label=torch.full((4, 1), float("nan")))
        torch.testing.assert_close(model(**changed)["loss"], result["loss"])

    def test_linear_probe_freezes_values_and_dropout(self):
        model = self.model(dropout=0.5, stage="linear_probe")
        model.train()
        self.assertFalse(model.encoders.training)
        before = copy.deepcopy(model.encoders.state_dict())
        classifier_before = model.classifier.weight.detach().clone()
        first = model(**self.batch)
        torch.testing.assert_close(first["logit"], model(**self.batch)["logit"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        first["loss"].backward()
        optimizer.step()
        for key, value in before.items():
            torch.testing.assert_close(model.encoders.state_dict()[key], value)
        self.assertFalse(torch.equal(model.classifier.weight, classifier_before))
        model.set_stage("finetune")
        self.assertTrue(model.encoders.training)
        self.assertTrue(all(p.requires_grad for p in model.encoders.parameters()))
        self.assertEqual(model.mode, "binary")

    def test_downstream_inference_does_not_need_text_or_labels(self):
        model = self.model(stage="finetune")
        result = model(
            signal=self.batch["signal"], tabular=self.batch["tabular"], embed=True
        )
        self.assertEqual(result["y_prob"].shape, (4, 1))
        self.assertEqual(result["embed"].shape, (4, 8))
        self.assertNotIn("loss", result)
        self.assertNotIn("y_true", result)
        model(**self.batch)["loss"].backward()
        self.assertIsNotNone(model.encoders["signal_input"].weight.grad)

    def test_masked_padding_is_ignored(self):
        model = self.model(mask_key="mask", stage="finetune").eval()
        changed = copy.deepcopy(self.batch)
        changed["signal"][:, -1] = float("nan")
        torch.testing.assert_close(
            model(**self.batch)["logit"], model(**changed)["logit"]
        )
        shortened = dict(
            self.batch,
            signal=self.batch["signal"][:, :3],
            mask=self.batch["mask"][:, :3],
        )
        torch.testing.assert_close(
            model(**self.batch)["logit"], model(**shortened)["logit"]
        )
        changed["mask"] = torch.zeros(4, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "valid timestep"):
            model(**changed)

    def test_optional_text_does_not_change_supervised_initialization(self):
        torch.manual_seed(11)
        with_text = self.model(stage="finetune").eval()
        torch.manual_seed(11)
        without_text = self.model(stage="finetune", text_key="unused_text").eval()
        self.assertIsNone(without_text.text_projection)
        torch.testing.assert_close(
            with_text(**self.batch)["logit"], without_text(**self.batch)["logit"]
        )

    def test_zero_signal_is_valid_and_positions_encode_order(self):
        model = self.model(stage="finetune").eval()
        zero = dict(self.batch, signal=torch.zeros_like(self.batch["signal"]))
        self.assertTrue(torch.isfinite(model(**zero)["loss"]))
        reversed_batch = dict(self.batch, signal=self.batch["signal"].flip(1))
        self.assertFalse(
            torch.allclose(model.encode(**self.batch), model.encode(**reversed_batch))
        )

    def test_save_load_and_modality_ablations(self):
        for kwargs in ({}, {"signal_key": None}, {"tabular_key": None}):
            with self.subTest(kwargs=kwargs):
                model = self.model(stage="finetune", **kwargs).eval()
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                other = self.model(stage="finetune", **kwargs).eval()
                other.load_state_dict(torch.load(buffer, weights_only=True))
                torch.testing.assert_close(
                    model(**self.batch)["logit"], other(**self.batch)["logit"]
                )

    def test_multiclass_and_unlabeled_schemas(self):
        for schema, stage in (({"label": "multiclass"}, "finetune"), ({}, "pretrain")):
            dataset = create_sample_dataset(self.samples, self.schema, schema)
            batch = next(iter(get_dataloader(dataset, batch_size=4)))
            model = CLIPStress(
                dataset, hidden_dim=8, num_layers=1, num_heads=2, stage=stage
            )
            result = model(**batch)
            self.assertTrue(torch.isfinite(result["loss"]))
            if stage == "finetune":
                self.assertEqual(result["y_prob"].shape, (4, 2))
            else:
                with self.assertRaisesRegex(ValueError, "labeled dataset"):
                    model.set_stage("finetune")

    def test_invalid_configuration_and_inputs(self):
        for options in (
            {"temperature": 0},
            {"temperature": float("nan")},
            {"hidden_dim": 7},
            {"num_layers": 0},
            {"dropout": 1},
            {"signal_key": None, "tabular_key": None},
            {"tabular_key": "missing"},
            {"tabular_key": "signal"},
            {"stage": "invalid"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model(**options)
        singleton = {k: v[:1] for k, v in self.batch.items()}
        with self.assertRaisesRegex(ValueError, "at least two pairs"):
            self.model()(**singleton)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model()(
                **dict(self.batch, text_embedding=torch.full((4, 5), float("inf")))
            )
        with self.assertRaisesRegex(ValueError, "text_embedding"):
            self.model()(signal=self.batch["signal"], tabular=self.batch["tabular"])

    def test_stage_switch_clears_old_gradients(self):
        model = self.model()
        model(**self.batch)["loss"].backward()
        model.set_stage("linear_probe")
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        before = copy.deepcopy(model.encoders.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        model(**self.batch)["loss"].backward()
        optimizer.step()
        for key, value in before.items():
            torch.testing.assert_close(model.encoders.state_dict()[key], value)

    def test_transfer_from_unlabeled_dataset(self):
        dataset = create_sample_dataset(
            [
                {k: v for k, v in sample.items() if k != "label"}
                for sample in self.samples
            ],
            self.schema,
            {},
        )
        pretrained = CLIPStress(
            dataset,
            hidden_dim=8,
            projection_dim=6,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(pretrained.parameters(), lr=0.01)
        pretrained(**self.batch)["loss"].backward()
        optimizer.step()
        downstream = self.model(stage="linear_probe")
        head = copy.deepcopy(downstream.classifier.state_dict())
        downstream.load_pretrained_encoder(pretrained)
        pretrained.eval()
        torch.testing.assert_close(
            downstream.encode(**self.batch), pretrained.encode(**self.batch)
        )
        for key, value in head.items():
            torch.testing.assert_close(downstream.classifier.state_dict()[key], value)
        before = copy.deepcopy(downstream.state_dict())
        with self.assertRaisesRegex(ValueError, "architecture"):
            downstream.load_pretrained_encoder(self.model(hidden_dim=16))
        for key, value in before.items():
            torch.testing.assert_close(downstream.state_dict()[key], value)

    def test_rejects_mismatched_feature_widths(self):
        for key, value in (
            ("signal", torch.zeros(4, 4, 3)),
            ("tabular", torch.zeros(4, 2)),
            ("text_embedding", torch.zeros(4, 7)),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.model()(**dict(self.batch, **{key: value}))

    def test_transfer_rejects_different_attention_heads_without_mutation(self):
        downstream = self.model(stage="finetune")
        before = copy.deepcopy(downstream.state_dict())
        # Head counts change attention even when all weight shapes match.
        with self.assertRaisesRegex(ValueError, "attention head"):
            downstream.load_pretrained_encoder(self.model(num_heads=4))
        for key, value in before.items():
            torch.testing.assert_close(downstream.state_dict()[key], value)

    def test_trainer_in_both_stages(self):
        model = self.model()
        loader = get_dataloader(self.dataset, batch_size=4)
        for stage in ("pretrain", "linear_probe", "finetune"):
            model.set_stage(stage)
            trainer = Trainer(
                model=model, device="cpu", enable_logging=False, metrics=["accuracy"]
            )
            trainer.train(loader, epochs=1, load_best_model_at_last=False)
            scores = trainer.evaluate(loader)
            self.assertTrue(np.isfinite(scores["loss"]))
            self.assertEqual("accuracy" in scores, stage != "pretrain")


class TestStressPrompts(unittest.TestCase):
    def test_statistics_and_constant_channel(self):
        prompt = build_stress_prompt(
            np.array([[1.0, 2.0], [3.0, 2.0]]), ["steps", "heart rate"], {"age": 30}
        )
        self.assertIn("steps: mean is 2, std is 1, min is 1, max is 3", prompt)
        self.assertIn("kurtosis is 0, skewness is 0, IQR is 0", prompt)
        self.assertIn("age: 30", prompt)
        self.assertNotIn("nan", prompt)

    def test_rejects_labels_and_nonfinite_values(self):
        for key in ("label", "STRESS", "stai_stress_category", "custom_outcome"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Target"):
                build_stress_prompt(
                    np.ones((3, 1)), ["steps"], {key: 1}, target_keys=["custom_outcome"]
                )
        with self.assertRaises(ValueError):
            build_stress_prompt(np.array([[np.nan]]), ["steps"], {})

    def test_text_pooling_excludes_padding_without_downloads(self):
        class Encoder(nn.Module):
            def forward(self, **tokens):
                self.assert_frozen = not torch.is_grad_enabled() and not self.training
                return SimpleNamespace(
                    last_hidden_state=torch.tensor(
                        [[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]]
                    ).expand(tokens["attention_mask"].shape[0], -1, -1)
                )

        encoder = Encoder()

        def tokenize(prompts, **kwargs):
            if kwargs.get("return_length"):
                return {"length": [2] * len(prompts)}
            return {"attention_mask": torch.tensor([[1, 1, 0]] * len(prompts))}

        with (
            patch("transformers.AutoModel.from_pretrained", return_value=encoder),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenize),
        ):
            actual = encode_stress_prompts(["steps: 1", "steps: 2"], batch_size=1)
        torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
        self.assertTrue(encoder.assert_frozen)
        self.assertFalse(actual.requires_grad)

    def test_long_prompts_rejected_before_loading_encoder(self):
        with (
            patch("transformers.AutoTokenizer.from_pretrained") as tokenizer,
            patch("transformers.AutoModel.from_pretrained") as encoder,
        ):
            tokenizer.return_value.return_value = {"length": [512, 513]}
            with self.assertRaisesRegex(ValueError, "1 of 2 prompts exceed"):
                encode_stress_prompts(["steps: 1", "steps: 2"], truncation="error")
            encoder.assert_not_called()

    def test_long_prompts_warn_and_still_encode(self):
        def tokenize(prompts, **kwargs):
            if kwargs.get("return_length"):
                return {"length": [513] * len(prompts)}
            return {"attention_mask": torch.ones(len(prompts), 2)}

        class Encoder(nn.Module):
            def forward(self, **tokens):
                return SimpleNamespace(
                    last_hidden_state=torch.ones(len(tokens["attention_mask"]), 2, 3)
                )

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenize),
            patch("transformers.AutoModel.from_pretrained", return_value=Encoder()),
            self.assertWarnsRegex(UserWarning, "2 of 2 prompts exceed"),
        ):
            actual = encode_stress_prompts(["steps: 1", "steps: 2"], batch_size=1)
        torch.testing.assert_close(actual, torch.ones(2, 3))

    def test_chunk_pooling_preserves_order_and_weights_valid_tokens(self):
        def tokenize(prompts, **kwargs):
            if kwargs.get("return_length"):
                return {"length": [3, 2]}
            self.assertTrue(kwargs["return_overflowing_tokens"])
            return {
                "input_ids": torch.tensor([[1, 1], [7, 999], [4, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 0], [1, 1]]),
                "overflow_to_sample_mapping": torch.tensor([0, 0, 1]),
            }

        class Encoder(nn.Module):
            def forward(inner, **tokens):
                self.assertLessEqual(len(tokens["input_ids"]), 2)
                return SimpleNamespace(
                    last_hidden_state=tokens["input_ids"].float().unsqueeze(-1)
                )

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenize),
            patch("transformers.AutoModel.from_pretrained", return_value=Encoder()),
        ):
            actual = encode_stress_prompts(
                ["long", "short"], batch_size=2, max_length=2, truncation="chunk"
            )
        torch.testing.assert_close(actual, torch.tensor([[3.0], [4.0]]))


if __name__ == "__main__":
    unittest.main()
