"""Tests for the MIRA medical time-series foundation model (offline, tiny)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MIRA, MIRABackbone, MIRAConfig
from pyhealth.models.mira import (
    HF_REPO_ID,
    HF_REVISION,
    load_mira_checkpoint,
    normalize_time_for_ctrope,
)
from pyhealth.trainer import Trainer


def _tiny_config(**kwargs):
    values = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 2, "num_experts": 4,
        "ode_func_hidden_dims": [8],
    }
    values.update(kwargs)
    return MIRAConfig(**values)


def _reference_time_normalization(times, alpha=1.0, snap=0.1, eps=1e-4):
    """The authors' normalize_time_for_ctrope for one unpadded series."""
    t = times.float()
    length = t.shape[1]
    t_min = t.min(dim=1, keepdim=True).values
    t_max = t.max(dim=1, keepdim=True).values
    scaled = (t - t_min) / (t_max - t_min).clamp(min=1e-8) * alpha * (length - 1)
    out = torch.round(scaled / snap) * snap
    for b in range(out.shape[0]):
        for i in range(1, length):
            if out[b, i] <= out[b, i - 1]:
                out[b, i] = out[b, i - 1] + eps
    return out


def _samples(n=6, length=8, n_vars=3, steps=2, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        x = rng.normal(size=(length, n_vars)).astype("float32")
        x[rng.random(size=x.shape) < 0.4] = np.nan
        x[:, 2] = np.nan  # a variable that is never observed
        x[0, 0] = 1.0  # at least one observation per sample
        y = rng.normal(size=(steps, n_vars)).astype("float32")
        y[0, 1] = np.nan
        samples.append({
            "patient_id": f"p{i}",
            "x": x,
            "x_time": np.arange(length, dtype="float32") / 10,
            "y_time": np.arange(length, length + steps, dtype="float32") / 10,
            "y": y,
        })
    return samples


def _dataset(samples):
    return create_sample_dataset(
        samples,
        {"x": "tensor", "x_time": "tensor", "y_time": "tensor"},
        {"y": "tensor"},
    )


class TestMIRAConfig(unittest.TestCase):
    def test_defaults_are_mira_large(self):
        config = MIRAConfig()
        self.assertEqual(
            (config.hidden_size, config.num_hidden_layers, config.num_experts),
            (768, 12, 8),
        )

    def test_from_dict_ignores_hf_keys(self):
        config = MIRAConfig.from_dict(
            {"hidden_size": 32, "architectures": ["MIRAForPrediction"],
             "num_key_value_heads": None, "num_attention_heads": 4}
        )
        self.assertEqual(config.hidden_size, 32)
        self.assertEqual(config.num_key_value_heads, 4)
        self.assertEqual(MIRAConfig.from_dict(config.to_dict()), config)

    def test_rejects_bad_head_count(self):
        with self.assertRaises(ValueError):
            MIRAConfig(hidden_size=10, num_attention_heads=3)


class TestTimeNormalization(unittest.TestCase):
    def test_matches_authors_loop(self):
        torch.manual_seed(0)
        times = torch.sort(torch.rand(4, 12) * 50, dim=1).values
        times[:, 3] = times[:, 2]  # a tie
        ours = normalize_time_for_ctrope(times)
        torch.testing.assert_close(ours, _reference_time_normalization(times))

    def test_padding_gives_unpadded_times(self):
        times = torch.tensor([[0.0, 3.0, 4.0, 9.0]])
        padded = torch.cat([torch.zeros(1, 3), times], dim=1)
        mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
        out = normalize_time_for_ctrope(padded, mask)
        torch.testing.assert_close(out[:, 3:], normalize_time_for_ctrope(times))
        self.assertTrue(torch.all(out[:, :3] == 0))

    def test_single_point(self):
        out = normalize_time_for_ctrope(torch.tensor([[5.0]]))
        torch.testing.assert_close(out, torch.zeros(1, 1))


class TestMIRABackbone(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.backbone = MIRABackbone(_tiny_config()).eval()

    def test_parameter_names_follow_authors_code(self):
        names = set(self.backbone.state_dict())
        for name in [
            "model.embed_layer.emb_layer.weight",
            "model.embed_layer.gate_layer.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.ffn_layer.gate.weight",
            "model.layers.1.ffn_layer.experts.3.down_proj.weight",
            "model.layers.1.ffn_layer.shared_expert.up_proj.weight",
            "model.layers.1.ffn_layer.shared_expert_gate.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "ode_extrapolation_block.ode_func.net.0.weight",
            "ode_extrapolation_block.ode_func.net.2.bias",
            "lm_heads.0.out_layer.weight",
        ]:
            self.assertIn(name, names)
        self.assertNotIn("model.layers.0.self_attn.inv_freq", names)
        dense = MIRABackbone(_tiny_config(use_dense=True))
        self.assertIn(
            "model.layers.0.ffn_layer.gate_proj.weight", dense.state_dict()
        )

    def test_left_padding_matches_unpadded(self):
        a_vals, a_times = torch.randn(1, 7), torch.cumsum(torch.rand(1, 7), dim=1)
        b_vals, b_times = torch.randn(1, 4), torch.cumsum(torch.rand(1, 4), dim=1)
        values = torch.cat([a_vals, torch.cat([torch.zeros(1, 3), b_vals], 1)])
        times = torch.cat([a_times, torch.cat([torch.zeros(1, 3), b_times], 1)])
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[1, :3] = False
        with torch.no_grad():
            batched = self.backbone(values, times, mask)
            alone = torch.cat([self.backbone(a_vals, a_times),
                               self.backbone(b_vals, b_times)])
        self.assertTrue(torch.isfinite(batched).all())
        torch.testing.assert_close(batched, alone, atol=1e-5, rtol=1e-4)

    def test_time_shift_invariance(self):
        values, times = torch.randn(2, 6), torch.cumsum(torch.rand(2, 6), dim=1)
        with torch.no_grad():
            torch.testing.assert_close(
                self.backbone(values, times), self.backbone(values, times * 3 + 5)
            )

    def test_forecast_appends_predictions(self):
        values, times = torch.randn(2, 5), torch.arange(5.0).repeat(2, 1)
        queries = torch.tensor([[5.0, 6.0, 7.0]]).repeat(2, 1)
        with torch.no_grad():
            preds = self.backbone.forecast(values, times, queries)
            first = self.backbone(values, times)
            second = self.backbone(
                torch.cat([values, preds[:, :1]], 1),
                torch.cat([times, queries[:, :1]], 1),
            )
        self.assertEqual(preds.shape, (2, 3))
        torch.testing.assert_close(preds[:, 0], first)
        torch.testing.assert_close(preds[:, 1], second)

    def test_ode_moves_state_only_forward(self):
        values, times = torch.randn(2, 5), torch.arange(5.0).repeat(2, 1)
        with torch.no_grad():
            plain = self.backbone(values, times)
            same = self.backbone(values, times, target_times=times[:, -1])
            moved = self.backbone(values, times, target_times=times[:, -1] + 2)
            past = self.backbone(values, times, target_times=times[:, -1] - 1)
        torch.testing.assert_close(same, plain)
        torch.testing.assert_close(past, plain)
        self.assertFalse(torch.allclose(moved, plain))

    def test_rk4_matches_fine_euler(self):
        ode = self.backbone.ode_extrapolation_block
        h, dt = torch.randn(3, 16), torch.tensor([0.5, 1.0, 2.0])
        with torch.no_grad():
            rk4 = ode(h, dt)
            euler = h.clone()
            n = 4000
            for i in range(n):
                tau = torch.tensor(i / n)
                euler = euler + dt[:, None] * ode.ode_func(tau * dt, euler) / n
        torch.testing.assert_close(rk4, euler, atol=1e-3, rtol=1e-3)


class TestMIRAModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.samples = _samples()
        self.dataset = _dataset(self.samples)
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=6)))

    def test_forward_backward_and_inference(self):
        model = MIRA(self.dataset, config=_tiny_config())
        out = model(**self.batch)
        self.assertEqual(out["forecast"].shape, (6, 2, 3))
        self.assertEqual(model.mode, "regression")
        mask = torch.isfinite(self.batch["y"])
        self.assertEqual(out["y_true"].shape, (int(mask.sum()),))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        grad = model.backbone.lm_heads[0].out_layer.weight.grad
        self.assertIsNotNone(grad)
        # A never-observed variable is forecast as 0.
        self.assertTrue(torch.all(out["forecast"][:, :, 2] == 0))

        unlabeled = {k: v for k, v in self.batch.items() if k != "y"}
        no_label = model(**unlabeled)
        self.assertNotIn("loss", no_label)
        torch.testing.assert_close(no_label["y_prob"], out["forecast"])

    def test_channel_independent_and_scale_equivariant(self):
        model = MIRA(self.dataset, config=_tiny_config()).eval()
        x, xt, yt = self.batch["x"], self.batch["x_time"], self.batch["y_time"]
        with torch.no_grad():
            full = model.predict(x, xt, yt)
            alone = model.predict(x[:, :, :1], xt, yt)
            shifted = model.predict(x * 3 + 2, xt, yt)
        torch.testing.assert_close(full[:, :, :1], alone)
        observed = torch.isfinite(x).sum(1) > 1  # series with a usable std
        torch.testing.assert_close(
            (shifted[:, :, :2] - 2) / 3 * observed[:, None, :2],
            full[:, :, :2] * observed[:, None, :2],
            atol=1e-4, rtol=1e-4,
        )

    def test_length_chunks_do_not_change_forecast(self):
        model = MIRA(self.dataset, config=_tiny_config()).eval()
        with torch.no_grad():
            whole = model(**self.batch)["forecast"]
            model.chunk_size = 2
            chunked = model(**self.batch)["forecast"]
        torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-4)

    def test_use_ode_changes_forecast(self):
        torch.manual_seed(0)
        plain = MIRA(self.dataset, config=_tiny_config()).eval()
        ode = MIRA(self.dataset, config=_tiny_config(), use_ode=True,
                   ode_time_scale=10.0).eval()
        ode.load_state_dict(plain.state_dict())
        with torch.no_grad():
            a = plain(**self.batch)["forecast"][:, :, :2]
            b = ode(**self.batch)["forecast"][:, :, :2]
        self.assertFalse(torch.allclose(a, b))
        with self.assertRaises(ValueError):
            MIRA(self.dataset, config=_tiny_config(use_terminal_ode=False),
                 use_ode=True)

    def test_loads_local_checkpoint_with_every_weight(self):
        torch.manual_seed(1)
        source = MIRABackbone(_tiny_config()).eval()
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(source.config.to_dict(), architectures=["MIRAForPrediction"],
                          torch_dtype="bfloat16")
            Path(tmp, "config.json").write_text(json.dumps(config))
            state = {k: v.contiguous() for k, v in source.state_dict().items()}
            save_file(state, str(Path(tmp, "model.safetensors")))
            model = MIRA(self.dataset, pretrained=tmp).eval()
            self.assertEqual(model.backbone.config, source.config)
            values, times = torch.randn(2, 5), torch.arange(5.0).repeat(2, 1)
            with torch.no_grad():
                torch.testing.assert_close(
                    model.backbone(values, times), source(values, times)
                )

            del state["model.norm.weight"]
            save_file(state, str(Path(tmp, "model.safetensors")))
            with self.assertRaisesRegex(ValueError, "model.norm.weight"):
                MIRA(self.dataset, pretrained=tmp)

    def test_default_download_uses_pinned_revision(self):
        source = MIRABackbone(_tiny_config()).eval()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(source.config.to_dict()))
            state = {k: v.contiguous() for k, v in source.state_dict().items()}
            save_file(state, str(Path(tmp, "model.safetensors")))

            def fake_download(repo_id, filename, revision=None):
                return str(Path(tmp, filename))

            with mock.patch(
                "huggingface_hub.hf_hub_download", side_effect=fake_download
            ) as download:
                config, loaded = load_mira_checkpoint()
                model = MIRA(self.dataset, pretrained=True)
        self.assertEqual(config, source.config)
        self.assertEqual(set(loaded), set(state))
        self.assertEqual(model.backbone.config, source.config)
        calls = {c.args[1]: c for c in download.call_args_list}
        self.assertEqual(set(calls), {"config.json", "model.safetensors"})
        for call in download.call_args_list:
            self.assertEqual(call.args[0], HF_REPO_ID)
            self.assertEqual(call.kwargs["revision"], HF_REVISION)

    def test_pretrained_rejects_config(self):
        with self.assertRaisesRegex(ValueError, "pretrained"):
            MIRA(self.dataset, pretrained=True, config=_tiny_config())

    def test_trainer_scores_regression(self):
        model = MIRA(self.dataset, config=_tiny_config(num_hidden_layers=1))
        trainer = Trainer(model=model, metrics=["mse", "mae"], enable_logging=False)
        loader = get_dataloader(self.dataset, batch_size=3)
        trainer.train(train_dataloader=loader, epochs=1)
        scores = trainer.evaluate(loader)
        self.assertTrue(np.isfinite(scores["mse"]))
        self.assertTrue(np.isfinite(scores["mae"]))


if __name__ == "__main__":
    unittest.main()
