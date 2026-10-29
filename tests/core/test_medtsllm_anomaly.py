"""Tiny offline GPT-2/Llama MedTsLLMAnomaly shape, prompt, gradient and Trainer tests.

Contributors: Thi Van Trinh Tran and Zexi Li.
All weights are initialized locally from tiny configs, never downloaded.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MedTsLLMAnomaly
from pyhealth.trainer import Trainer

SEQ, NF = 32, 2


class _Tokenizer:
    """Character tokenizer that prepends BOS on every call, like Llama-2's."""

    bos_token, eos_token, pad_token = "<s>", "</s>", None
    pad_token_id = 2

    def __call__(self, text, **kwargs):
        ids = [1] + ([1] if text == "<s>" else [3 + ord(c) % 20 for c in text])
        return SimpleNamespace(input_ids=torch.tensor([ids]))


def _dataset(n=4):
    rng = np.random.default_rng(0)
    samples = []
    for i in range(n):
        label = np.zeros(SEQ, dtype=np.int64)
        label[5:9] = i % 2
        samples.append(
            {
                "patient_id": str(i),
                "record_id": str(i),
                "signal": (3 + 2 * rng.standard_normal((SEQ, NF))).astype(np.float32),
                "label": label,
                "description": f"{60 + i} M",
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "tensor"},
        in_memory=True,
    )


def _backbone(kind):
    if kind == "gpt2":
        return GPT2Model(
            GPT2Config(
                vocab_size=24,
                n_embd=8,
                n_layer=1,
                n_head=2,
                n_positions=256,
            )
        )
    return LlamaModel(
        LlamaConfig(
            vocab_size=24,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    )


def _model(dataset, kind="gpt2", **kwargs):
    options = {
        "patch_len": 8,
        "stride": 4,
        "d_model": 4,
        "d_ff": 4,
        "n_heads": 2,
        "num_tokens": 6,
        "dataset_description": "Synthetic ECG.",
    }
    options.update(kwargs)
    return MedTsLLMAnomaly(
        dataset,
        backbone=_backbone(kind),
        tokenizer=_Tokenizer(),
        **options,
    )


class TestMedTsLLMAnomaly(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=3)))

    def test_both_backbones_output_shapes_and_loss(self):
        for kind in ("gpt2", "llama"):
            with self.subTest(kind=kind):
                out = _model(self.dataset, kind)(**self.batch)
                self.assertEqual(out["y_prob"].shape, (3, SEQ, NF))
                self.assertEqual(out["y_true"].shape, (3, SEQ, NF))
                self.assertEqual(out["anomaly_label"].shape, (3, SEQ))
                self.assertEqual(out["loss"].ndim, 0)
                self.assertTrue(torch.isfinite(out["loss"]))

    def test_half_precision_backbone_with_float32_adapters(self):
        # Llama-2 runs in bf16 (torch_dtype); the adapters stay in float32.
        for kind in ("gpt2", "llama"):
            with self.subTest(kind=kind):
                model = _model(self.dataset, kind)
                model.llm.to(torch.bfloat16)
                out = model(**self.batch)
                self.assertEqual(out["y_prob"].dtype, torch.float32)
                self.assertTrue(torch.isfinite(out["loss"]))
                out["loss"].backward()
                self.assertIsNotNone(model.mapping_layer.weight.grad)

    def test_patch_count_matches_original_formula(self):
        model = _model(self.dataset)
        self.assertEqual(model.n_patches, (SEQ - 8) // 4 + 2)
        series = torch.randn(5, SEQ)
        self.assertEqual(model.patch_embedding(series).shape[1], model.n_patches)

    def test_only_adapters_train_and_llm_is_frozen(self):
        model = _model(self.dataset, "llama")
        model(**self.batch)["loss"].backward()
        self.assertFalse(any(p.requires_grad for p in model.llm.parameters()))
        for name in (
            "mapping_layer.weight",
            "patch_embedding.token_conv.weight",
            "reprogramming_layer.query_projection.weight",
            "output_projection.weight",
            "embedding_downsample.weight",
        ):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_standardization_uses_training_statistics(self):
        model = _model(self.dataset)
        signals = torch.stack([torch.as_tensor(s["signal"]) for s in self.dataset])
        flat = signals.reshape(-1, NF).double()
        torch.testing.assert_close(model.feature_mean, flat.mean(0).float())
        torch.testing.assert_close(
            model.feature_std, flat.std(0, unbiased=False).float()
        )
        out = model(**self.batch)
        expected = (self.batch["signal"] - model.feature_mean) / model.feature_std
        torch.testing.assert_close(out["y_true"], expected)
        raw = _model(self.dataset, standardize=False)(**self.batch)
        torch.testing.assert_close(raw["y_true"], self.batch["signal"])

    def test_prompt_parts_match_original_layout(self):
        model = _model(self.dataset, prompt_patient=True)
        parts = model.prompt_parts(["69 M", ""], 2)
        task = model.task_description
        self.assertEqual(
            parts[0],
            [
                "<s>",
                "Dataset: Synthetic ECG. ",
                "Patient description: 69 M ",
                f"Task: {task} ",
                "Time series: ",
            ],
        )
        self.assertEqual(
            parts[1],
            [
                "<s>",
                "Dataset: Synthetic ECG. ",
                f"Task: {task} ",
                "Time series: ",
            ],
        )
        self.assertIn(f"past {SEQ} steps", task)

    def test_patient_prompt_changes_reconstruction_and_needs_descriptions(self):
        model = _model(self.dataset, prompt_patient=True).eval()
        x = self.batch["signal"]
        with torch.no_grad():
            a = model(signal=x, description=["60 M", "60 M", "60 M"])["y_prob"]
            b = model(signal=x, description=["99 F", "99 F", "99 F"])["y_prob"]
        self.assertFalse(torch.allclose(a, b))
        with self.assertRaisesRegex(ValueError, "description"):
            model(signal=x)

    def test_prompts_of_different_length_are_left_padded(self):
        model = _model(self.dataset, prompt_patient=True)
        emb = model.encode_prompts(model.prompt_parts(["a long description", ""], 2))
        pad = model.llm.get_input_embeddings()(torch.tensor(2))
        short_len = sum(
            model._tokens(p).shape[1] for p in model.prompt_parts([""], 1)[0]
        )
        n_pad = emb.shape[1] - short_len
        self.assertGreater(n_pad, 0)
        torch.testing.assert_close(emb[1, :n_pad], pad.expand(n_pad, -1))

    def test_native_trainer_one_step_and_inference_with_labels(self):
        model = _model(self.dataset)
        self.assertEqual(model.mode, "regression")
        trainer = Trainer(model, device="cpu", metrics=["mse"], enable_logging=False)
        loader = get_dataloader(self.dataset, batch_size=2)
        trainer.train(
            loader, epochs=1, steps_per_epoch=1, load_best_model_at_last=False
        )
        _, y_prob, loss, extra = trainer.inference(
            loader, additional_outputs=["anomaly_label"]
        )
        self.assertEqual(np.asarray(y_prob).reshape(-1).size, 4 * SEQ * NF)
        self.assertEqual(np.asarray(extra["anomaly_label"]).shape, (4, SEQ))
        self.assertTrue(np.isfinite(loss))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _model(self.dataset, d_model=0)
        with self.assertRaisesRegex(ValueError, "patch_len"):
            _model(self.dataset, patch_len=SEQ + 1)
        with self.assertRaisesRegex(ValueError, "together"):
            MedTsLLMAnomaly(self.dataset, backbone=_backbone("gpt2"))
        with self.assertRaisesRegex(ValueError, "no input"):
            _model(self.dataset, signal_key="ecg")
        with self.assertRaisesRegex(ValueError, "shape"):
            _model(self.dataset)(signal=torch.zeros(1, SEQ + 1, NF))


if __name__ == "__main__":
    unittest.main()
