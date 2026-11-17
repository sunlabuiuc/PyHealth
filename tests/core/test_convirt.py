import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ConVIRT, ConVIRTLinearProbe
from pyhealth.models.convirt import info_nce_loss
from pyhealth.trainer import Trainer

HIDDEN = 8
EHR_DIM = 6


def _dataset(n: int = 8):
    generator = torch.Generator().manual_seed(11)
    samples = [
        {
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "ehr": torch.randn(5, EHR_DIM, generator=generator).tolist(),
            "cxr": torch.randn(3, 32, 32, generator=generator).tolist(),
            "label": i % 2,
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"ehr": "tensor", "cxr": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="convirt_test",
    )


def _convirt(dataset, **kwargs):
    options = {
        "ehr_hidden_dim": HIDDEN,
        "projection_dim": HIDDEN,
        "cxr_backbone": "resnet18",
    }
    options.update(kwargs)
    return ConVIRT(dataset, **options)


class TestInfoNCE(unittest.TestCase):
    def test_matches_a_loop_over_samples(self) -> None:
        generator = torch.Generator().manual_seed(0)
        z_ehr = torch.randn(5, 4, generator=generator)
        z_cxr = torch.randn(5, 4, generator=generator)
        temperature = 0.1
        sim = (
            torch.nn.functional.cosine_similarity(z_cxr[:, None], z_ehr[None], dim=-1)
            / temperature
        )
        total = 0.0
        for i in range(5):
            others = [j for j in range(5) if j != i]
            total += sim[i, i] - torch.logsumexp(sim[i, others], dim=0)
            total += sim[i, i] - torch.logsumexp(sim[others, i], dim=0)
        self.assertTrue(
            torch.allclose(
                info_nce_loss(z_ehr, z_cxr, temperature), -total / 5, atol=1e-5
            )
        )

    def test_is_symmetric_in_the_two_modalities(self) -> None:
        generator = torch.Generator().manual_seed(1)
        a, b = (
            torch.randn(6, 4, generator=generator),
            torch.randn(6, 4, generator=generator),
        )
        self.assertTrue(
            torch.allclose(info_nce_loss(a, b), info_nce_loss(b, a), atol=1e-5)
        )

    def test_aligned_pairs_have_lower_loss(self) -> None:
        z = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
        self.assertLess(info_nce_loss(z, z), info_nce_loss(z, z.roll(1, dims=0)))

    def test_needs_a_batch_of_two(self) -> None:
        with self.assertRaises(ValueError):
            info_nce_loss(torch.randn(1, 4), torch.randn(1, 4))


class TestConVIRT(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=8)))

    def test_forward_outputs(self) -> None:
        output = _convirt(self.dataset)(**self.batch)
        self.assertEqual(set(output), {"loss", "z_ehr", "z_cxr"})
        self.assertEqual(output["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(output["z_ehr"].shape, (8, HIDDEN))
        self.assertEqual(output["z_cxr"].shape, (8, HIDDEN))

    def test_scores_no_label(self) -> None:
        self.assertEqual(_convirt(self.dataset).label_processors, {})

    def test_all_parameters_get_gradients(self) -> None:
        model = _convirt(self.dataset)
        model(**self.batch)["loss"].backward()
        missing = [
            n
            for n, p in model.named_parameters()
            if p.grad is None and n != "_dummy_param"
        ]
        self.assertEqual(missing, [])

    def test_loss_falls_on_a_fixed_batch(self) -> None:
        torch.manual_seed(0)
        model = _convirt(self.dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()
        first = model(**self.batch)["loss"].item()
        for _ in range(15):
            optimizer.zero_grad()
            loss = model(**self.batch)["loss"]
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), first)

    def test_trainer_trains_on_the_loss_alone(self) -> None:
        model = _convirt(self.dataset)
        loader = get_dataloader(self.dataset, batch_size=4)
        trainer = Trainer(model=model, device="cpu", enable_logging=False)
        trainer.train(
            train_dataloader=loader,
            val_dataloader=loader,
            epochs=1,
            monitor="loss",
            monitor_criterion="min",
        )
        self.assertEqual(set(trainer.evaluate(loader)), {"loss"})

    def test_augmentation_runs_in_training_mode_only(self) -> None:
        model = _convirt(self.dataset, augment_cxr=True)
        model.train()
        self.assertTrue(torch.isfinite(model(**self.batch)["loss"]))
        model.eval()
        first = model(**self.batch)["z_cxr"]
        second = model(**self.batch)["z_cxr"]
        self.assertTrue(torch.equal(first, second))

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            _convirt(self.dataset, temperature=0.0)
        with self.assertRaises(ValueError):
            _convirt(self.dataset, ehr_feature_key="missing")
        with self.assertRaises(ValueError):
            _convirt(self.dataset, cxr_mean=(0.5, 0.5, 0.5))

    def test_images_are_used_as_given_by_default(self) -> None:
        self.assertIsNone(_convirt(self.dataset).encoders.cxr_mean)

    def test_images_added_to_the_batch_by_the_training_loop(self) -> None:
        dataset = create_sample_dataset(
            samples=[
                {
                    "patient_id": f"p{i}",
                    "ehr": torch.randn(5, EHR_DIM).tolist(),
                    "label": i % 2,
                }
                for i in range(4)
            ],
            input_schema={"ehr": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="convirt_no_image_input",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        batch["cxr"] = torch.rand(4, 3, 32, 32)
        self.assertTrue(torch.isfinite(_convirt(dataset)(**batch)["loss"]))


class TestConVIRTLinearProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _dataset()
        cls.loader = get_dataloader(cls.dataset, batch_size=4)
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=8)))

    def _probe(self):
        return ConVIRTLinearProbe(self.dataset, _convirt(self.dataset))

    def test_only_the_linear_layer_is_trained(self) -> None:
        probe = self._probe()
        trainable = [
            n
            for n, p in probe.named_parameters()
            if p.requires_grad and n != "_dummy_param"
        ]
        self.assertEqual(trainable, ["linear.weight", "linear.bias"])
        self.assertEqual(probe.linear.weight.numel() + 1, HIDDEN + 512 + 1)

    def test_forward_outputs(self) -> None:
        output = self._probe()(**self.batch)
        self.assertEqual(set(output), {"logit", "y_prob", "loss", "y_true"})
        self.assertEqual(output["logit"].shape, (8, 1))

    def test_encoders_are_frozen_and_stay_in_eval_mode(self) -> None:
        probe = self._probe()
        before = probe.encoders.cxr_encoder.bn1.running_mean.clone()
        probe.train()
        probe(**self.batch)["loss"].backward()
        self.assertFalse(probe.encoders.training)
        self.assertTrue(
            torch.equal(before, probe.encoders.cxr_encoder.bn1.running_mean)
        )
        self.assertTrue(all(p.grad is None for p in probe.encoders.parameters()))

    def test_probe_does_not_share_weights_with_the_pretrained_model(self) -> None:
        convirt = _convirt(self.dataset)
        probe = ConVIRTLinearProbe(self.dataset, convirt)
        with torch.no_grad():
            probe.encoders.ehr_encoder.weight_ih_l0.add_(1.0)
        self.assertTrue(convirt.encoders.ehr_encoder.weight_ih_l0.requires_grad)
        self.assertFalse(
            torch.equal(
                probe.encoders.ehr_encoder.weight_ih_l0,
                convirt.encoders.ehr_encoder.weight_ih_l0,
            )
        )

    def test_cached_features_match_the_forward_pass(self) -> None:
        probe = self._probe()
        features, labels = probe.extract_features(self.loader)
        self.assertEqual(features.shape, (8, HIDDEN + 512))
        self.assertEqual(labels.shape[0], 8)
        full = next(iter(get_dataloader(self.dataset, batch_size=8, shuffle=False)))
        probe.eval()
        with torch.no_grad():
            expected = probe(**full)["logit"]
            cached = probe.linear(features)
        self.assertTrue(torch.allclose(expected, cached, atol=1e-5))

    def test_state_dict_round_trip(self) -> None:
        source = self._probe()
        target = self._probe()
        target.load_state_dict(source.state_dict())
        source.eval()
        target.eval()
        self.assertTrue(
            torch.equal(source(**self.batch)["logit"], target(**self.batch)["logit"])
        )


if __name__ == "__main__":
    unittest.main()
