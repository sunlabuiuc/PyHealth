"""ECGJEPA and ECGJEPAEncoder on random signals."""

import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import ECGJEPA, ECGJEPAEncoder
from pyhealth.models.ecg_jepa import cross_pattern_mask, sincos_pos_embed
from pyhealth.trainer import Trainer

SMALL = {"dim": 16, "depth": 2, "num_heads": 2, "num_patches": 4, "patch_size": 5}


def _dataset(n_leads=3, n=8, mode="multilabel"):
    samples = []
    for i in range(n):
        label = ["a", "b"][: i % 3] if mode == "multilabel" else i % 2
        samples.append({"patient_id": str(i), "signal": torch.randn(n_leads, 20), "label": label})
    return create_sample_dataset(samples, {"signal": "tensor"}, {"label": mode})


class TestParts(unittest.TestCase):
    def test_pos_embed_halves_encode_patch_then_lead(self):
        pos = sincos_pos_embed(8, 3, 4)
        self.assertEqual(pos.shape, (12, 8))
        # Same patch, other lead: the first half (patch) is equal, the second is not.
        torch.testing.assert_close(pos[1, :4], pos[5, :4])
        self.assertFalse(torch.allclose(pos[1, 4:], pos[5, 4:]))
        # Same lead, other patch: the second half (lead) is equal.
        torch.testing.assert_close(pos[4, 4:], pos[7, 4:])
        with self.assertRaises(ValueError):
            sincos_pos_embed(6, 2, 2)

    def test_cross_pattern_mask(self):
        mask = cross_pattern_mask(3, 4)
        self.assertTrue(mask[0, 3])  # same lead
        self.assertTrue(mask[0, 8])  # same patch, lead 2
        self.assertFalse(mask[0, 9])  # other lead and patch
        self.assertTrue(torch.equal(mask, mask.T))

    def test_encoder_attends_only_through_the_mask(self):
        torch.manual_seed(0)
        encoder = ECGJEPAEncoder(num_leads=2, num_patches=4, patch_size=5, dim=16, depth=1, num_heads=2)
        encoder.eval()
        x = torch.randn(1, 2, 20)
        with torch.no_grad():
            tokens = encoder.W_P(x.reshape(1, 8, 5))
            pos = encoder.pos_embed
            block = encoder.encoder_blocks.blocks[0]
            mask = cross_pattern_mask(2, 4)
            out = encoder.norm(block(tokens + pos, mask)).mean(1)
            torch.testing.assert_close(encoder(x), out)
            # Changing lead 1, patch 3 does not reach token (lead 0, patch 0) in one block.
            y = x.clone()
            y[0, 1, 15:] += 5
            a = block(encoder.W_P(x.reshape(1, 8, 5)) + pos, mask)[0, 0]
            b = block(encoder.W_P(y.reshape(1, 8, 5)) + pos, mask)[0, 0]
            torch.testing.assert_close(a, b)

    def test_encoder_lead_subset_and_bad_inputs(self):
        encoder = ECGJEPAEncoder(num_leads=3, num_patches=4, patch_size=5, dim=16, depth=1, num_heads=2)
        self.assertEqual(encoder(torch.randn(2, 2, 20), leads=[0, 2]).shape, (2, 16))
        with self.assertRaises(ValueError):
            encoder(torch.randn(2, 3, 21))
        with self.assertRaises(ValueError):
            encoder(torch.randn(2, 2, 20), leads=[0])
        for leads in ([-1, 0], [0, 3]):
            with self.subTest(leads=leads), self.assertRaises(ValueError):
                encoder(torch.randn(2, 2, 20), leads=leads)


class TestECGJEPA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _dataset()
        self.batch = next(iter(get_dataloader(self.dataset, batch_size=8)))

    def test_forward_multilabel_and_binary(self):
        model = ECGJEPA(self.dataset, num_leads=3, **SMALL)
        out = model(**self.batch)
        self.assertEqual(out["y_prob"].shape, (8, 2))
        self.assertEqual(out["embed"].shape, (8, 16))
        out["loss"].backward()
        binary = _dataset(mode="binary")
        out = ECGJEPA(binary, num_leads=3, **SMALL)(**next(iter(get_dataloader(binary, batch_size=8))))
        self.assertEqual(out["y_prob"].shape, (8, 1))
        no_label = {"signal": self.batch["signal"]}
        self.assertNotIn("loss", ECGJEPA(self.dataset, num_leads=3, **SMALL)(**no_label))

    def test_checkpoint_round_trip_and_lead_indices(self):
        source = ECGJEPAEncoder(num_leads=8, **SMALL)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pth"
            torch.save({"encoder": source.state_dict(), "epoch": 100}, path)
            model = ECGJEPA(self.dataset, checkpoint=path, lead_indices=[0, 1, 6], **SMALL)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(model.encoder.state_dict()[name], value)
        model.eval()
        source.eval()
        with torch.no_grad():
            x = self.batch["signal"]
            torch.testing.assert_close(model.embed(x), source(x, [0, 1, 6]))

    def test_frozen_encoder_trains_only_the_head(self):
        model = ECGJEPA(self.dataset, num_leads=3, freeze_encoder=True, **SMALL)
        before = {n: p.clone() for n, p in model.encoder.named_parameters()}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        model.train()
        model(**self.batch)["loss"].backward()
        optimizer.step()
        for name, param in model.encoder.named_parameters():
            torch.testing.assert_close(param, before[name])
        self.assertFalse(model.encoder.training)
        self.assertIsNotNone(model.fc.weight.grad)

    def test_trainer(self):
        model = ECGJEPA(self.dataset, num_leads=3, **SMALL)
        trainer = Trainer(model=model, metrics=["roc_auc_macro"], enable_logging=False)
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=True, seed=0)
        trainer.train(train_dataloader=loader, epochs=1)
        self.assertIn("roc_auc_macro", trainer.evaluate(loader))

    def test_rejects_bad_settings(self):
        with self.assertRaises(ValueError):
            ECGJEPA(self.dataset, num_leads=2, **SMALL)  # 3 leads, 2-lead grid
        with self.assertRaises(ValueError):
            ECGJEPA(self.dataset, num_leads=8, lead_indices=[0, 1], **SMALL)
        with self.assertRaises(ValueError):
            ECGJEPA(self.dataset, num_leads=8, lead_indices=[-1, 0, 1], **SMALL)
        two_fields = create_sample_dataset(
            [{"patient_id": str(i), "signal": torch.randn(3, 20), "other": torch.randn(3, 20),
              "label": ["a"]} for i in range(2)],
            {"signal": "tensor", "other": "tensor"}, {"label": "multilabel"},
        )
        with self.assertRaises(ValueError):
            ECGJEPA(two_fields, num_leads=3, **SMALL)
        self.assertEqual(ECGJEPA(two_fields, num_leads=3, feature_key="other", **SMALL).feature_key, "other")


if __name__ == "__main__":
    unittest.main()
