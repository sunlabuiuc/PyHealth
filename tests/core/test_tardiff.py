"""Tests for the TarDiff generator. All data is synthetic."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel, TarDiff
from pyhealth.models.generators.tardiff import UNet1D, timestep_embedding

C, L = 2, 16
SMALL = {
    "base_channels": 8,
    "mults": (1, 2),
    "num_res_blocks": 1,
    "attention_levels": (1,),
    "heads": 2,
    "n_steps": 20,
}


def _dataset(n=8, label_kind="binary", n_classes=2, length=L):
    torch.manual_seed(0)
    samples = [
        {
            "patient_id": str(i),
            "signal": torch.randn(C, length) + (i % n_classes),
            "label": i % n_classes,
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": label_kind},
    )


class LinearGuide(BaseModel):
    """Logistic regression on the flattened series, a guide with a known gradient."""

    def __init__(self, dataset):
        super().__init__(dataset)
        self.fc = nn.Linear(C * L, 1)

    def forward(self, **kwargs):
        x = kwargs["signal"].to(self.device).flatten(1)
        y = kwargs["label"].to(self.device).float().view(-1, 1)
        logit = self.fc(x)
        return {
            "loss": F.binary_cross_entropy_with_logits(logit, y),
            "logit": logit,
            "y_prob": torch.sigmoid(logit),
            "y_true": y,
        }


class TestUNet1D(unittest.TestCase):
    def test_predicts_noise_of_the_input_shape(self):
        net = UNet1D(
            C,
            3,
            base_channels=8,
            mults=(1, 2, 2),
            num_res_blocks=2,
            attention_levels=(0, 2),
            heads=2,
        )
        x = torch.randn(4, C, 32)
        self.assertEqual(
            net(x, torch.arange(4), torch.tensor([0, 1, 2, 0])).shape, x.shape
        )

    def test_rejects_lengths_that_cannot_be_downsampled(self):
        net = UNet1D(
            C,
            2,
            base_channels=8,
            mults=(1, 2, 2),
            num_res_blocks=1,
            attention_levels=(),
            heads=2,
        )
        with self.assertRaises(ValueError):
            net(torch.randn(1, C, 18), torch.tensor([0]), torch.tensor([0]))

    def test_timestep_embedding(self):
        emb = timestep_embedding(torch.tensor([0, 5]), 7)
        self.assertEqual(emb.shape, (2, 7))
        torch.testing.assert_close(emb[0, :3], torch.ones(3))  # cos(0)


class TestTarDiff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()
        cls.loader = get_dataloader(cls.dataset, batch_size=4, shuffle=False)

    def _model(self, **kwargs):
        torch.manual_seed(0)
        return TarDiff(self.dataset, **{**SMALL, **kwargs})

    def test_forward_is_the_denoising_loss(self):
        model = self._model()
        out = model(**next(iter(self.loader)))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        x0 = torch.randn(3, C, L)
        noise = torch.randn_like(x0)
        t0 = torch.zeros(3, dtype=torch.long)
        a = model.alphas_cumprod[0]
        torch.testing.assert_close(
            model.q_sample(x0, t0, noise), a.sqrt() * x0 + (1 - a).sqrt() * noise
        )

    def test_train_model_updates_the_network(self):
        model = self._model()
        before = [p.clone() for p in model.net.parameters()]
        losses = model.train_model(self.loader, iterations=3)
        self.assertEqual(len(losses), 3)
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before, model.net.parameters()))
        )
        self.assertFalse(model.training)

    def test_generate_shapes_and_seeded_determinism(self):
        model = self._model()
        labels = torch.tensor([0, 1, 1])
        a = model.generate(labels, steps=4, generator=torch.Generator().manual_seed(1))
        b = model.generate(labels, steps=4, generator=torch.Generator().manual_seed(1))
        self.assertEqual(a["signal"].shape, (3, C, L))
        self.assertEqual(a["label"].tolist(), [0, 1, 1])
        torch.testing.assert_close(a["signal"], b["signal"])
        c = model.generate(
            labels, steps=4, eta=1.0, generator=torch.Generator().manual_seed(1)
        )
        self.assertFalse(torch.equal(a["signal"], c["signal"]))

    def test_guidance_gradient_is_the_mean_loss_gradient(self):
        guide = LinearGuide(self.dataset)
        # Drop BaseModel's empty _dummy_param; then fc.weight, fc.bias.
        G = [g for g in TarDiff.guidance_gradient(guide, self.loader) if g.numel()]
        x = torch.stack([s["signal"] for s in self.dataset]).flatten(1)
        y = torch.stack([s["label"] for s in self.dataset]).float().view(-1, 1)
        err = torch.sigmoid(guide.fc(x)) - y  # dl/dlogit per sample
        torch.testing.assert_close(G[0], (err * x).mean(0, keepdim=True).detach())
        torch.testing.assert_close(G[1], err.mean(0).detach())

    def test_influence_gradient_matches_the_closed_form(self):
        model = self._model()
        guide = LinearGuide(self.dataset)
        G = TarDiff.guidance_gradient(guide, self.loader)
        Gw, Gb = [g for g in G if g.numel()]
        x = torch.randn(3, C, L)
        y = torch.tensor([0, 1, 1])
        got = model._influence_grad(guide, G, x, y)
        # dot_i = (s_i - y_i) (G_w . x_i + G_b), with s_i = sigmoid(w . x_i + b):
        # grad_x dot_i = s_i (1 - s_i) (G_w . x_i + G_b) w + (s_i - y_i) G_w.
        w, b = guide.fc.weight.detach()[0], guide.fc.bias.detach()
        xf = x.flatten(1)
        s = torch.sigmoid(xf @ w + b)
        align = xf @ Gw[0] + Gb
        expected = (s * (1 - s) * align)[:, None] * w + (s - y.float())[:, None] * Gw[0]
        torch.testing.assert_close(got, expected.view(3, C, L))

    def test_guidance_moves_samples_and_scale_zero_does_not(self):
        model = self._model()
        guide = LinearGuide(self.dataset)
        G = TarDiff.guidance_gradient(guide, self.loader)
        labels = torch.tensor([0, 1])

        def run(**kwargs):
            gen = torch.Generator().manual_seed(3)
            return model.generate(labels, steps=3, generator=gen, **kwargs)["signal"]

        plain = run()
        torch.testing.assert_close(
            run(guide=guide, guidance=G, guidance_scale=0.0), plain
        )
        self.assertFalse(
            torch.equal(run(guide=guide, guidance=G, guidance_scale=10.0), plain)
        )
        with self.assertRaises(ValueError):
            run(guide=guide)

    def test_samples_rebuild_a_dataset_with_the_same_schema(self):
        model = self._model()
        generated = model.generate(torch.tensor([0, 1]), steps=2)
        synthetic = create_sample_dataset(
            samples=model.to_samples(generated),
            input_schema={"signal": "tensor"},
            output_schema={"label": "binary"},
        )
        self.assertEqual(len(synthetic), 2)
        self.assertEqual(synthetic[0]["signal"].shape, (C, L))

    def test_raw_labels_round_trip_through_the_real_processors(self):
        # Raw labels that are not 0..K-1: strings (binary) and {1, 2, 3}.
        for kind, raw in (("binary", ["healthy", "MI"]), ("multiclass", [1, 2, 3])):
            with self.subTest(kind=kind):
                real = create_sample_dataset(
                    samples=[
                        {
                            "patient_id": str(i),
                            "signal": torch.randn(C, L),
                            "label": raw[i % len(raw)],
                        }
                        for i in range(6)
                    ],
                    input_schema={"signal": "tensor"},
                    output_schema={"label": kind},
                )
                model = TarDiff(real, **SMALL)
                indices = model.class_indices(raw)
                samples = model.to_samples(model.generate(indices, steps=2))
                self.assertEqual([s["label"] for s in samples], raw)
                synthetic = create_sample_dataset(
                    samples=samples,
                    input_schema={"signal": "tensor"},
                    output_schema={"label": kind},
                    input_processors=real.input_processors,
                    output_processors=real.output_processors,
                )
                for sample, index in zip(synthetic, indices):
                    self.assertEqual(int(sample["label"].reshape(-1)[0]), int(index))

    def test_train_model_rejects_an_empty_dataloader(self):
        with self.assertRaises(ValueError):
            self._model().train_model([], iterations=1)

    def test_generate_restores_the_guide_mode(self):
        model = self._model()
        guide = LinearGuide(self.dataset).train()
        G = TarDiff.guidance_gradient(guide, self.loader)
        model.generate(torch.tensor([0]), steps=2, guide=guide, guidance=G)
        self.assertTrue(guide.training)

    def test_save_and_load_round_trip(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tardiff.pt")
            model.save_model(path)
            other = self._model()
            for p in other.net.parameters():
                nn.init.zeros_(p)
            other.load_model(path)
            for a, b in zip(model.state_dict().values(), other.state_dict().values()):
                torch.testing.assert_close(a, b)
            with self.assertRaises(ValueError):
                self._model(base_channels=16).load_model(path)

    def test_multiclass_labels_set_the_number_of_classes(self):
        dataset = _dataset(n=9, label_kind="multiclass", n_classes=3)
        model = TarDiff(dataset, **SMALL)
        self.assertEqual(model.n_classes, 3)
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        self.assertTrue(torch.isfinite(model(**batch)["loss"]))

    def test_rejects_unsupported_datasets(self):
        regression = _dataset(label_kind="regression")
        with self.assertRaises(ValueError):
            TarDiff(regression, **SMALL)
        with self.assertRaises(ValueError):
            TarDiff(_dataset(length=17), **SMALL)  # odd, not divisible by 2
        flat = create_sample_dataset(
            samples=[
                {"patient_id": "0", "signal": torch.randn(L), "label": 0},
                {"patient_id": "1", "signal": torch.randn(L), "label": 1},
            ],
            input_schema={"signal": "tensor"},
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            TarDiff(flat, **SMALL)


if __name__ == "__main__":
    unittest.main()
