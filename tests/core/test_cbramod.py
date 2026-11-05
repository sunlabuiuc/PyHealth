import tempfile
import unittest
from unittest.mock import patch

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.cbramod import (
    CBraMod,
    CBraModBackbone,
    build_classifier_head,
)


class TestCBraMod(unittest.TestCase):
    def _multiclass_dataset(self, n_time: int = 1000):
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "signal": torch.randn(16, n_time).tolist(),
                "label": i % 4,
            }
            for i in range(4)
        ]
        return create_sample_dataset(
            samples,
            {"signal": "tensor"},
            {"label": "multiclass"},
            dataset_name="test_cbramod_mc",
        )

    def _binary_dataset(self, n_time: int = 2000):
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "signal": torch.randn(16, n_time).tolist(),
                "label": i % 2,
            }
            for i in range(4)
        ]
        return create_sample_dataset(
            samples,
            {"signal": "tensor"},
            {"label": "binary"},
            dataset_name="test_cbramod_bin",
        )

    def test_forward_multiclass(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, dataset.output_processors["label"].size()))
        self.assertEqual(out["loss"].dim(), 0)

    def test_forward_binary(self):
        dataset = self._binary_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=10,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        out = model(**next(iter(loader)))
        self.assertEqual(out["logit"].shape[1], 1)

    def test_classifier_variants(self):
        dataset = self._multiclass_dataset(n_time=400)
        for name in (
            "all_patch_reps",
            "all_patch_reps_twolayer",
            "all_patch_reps_onelayer",
            "avgpooling_patch_reps",
        ):
            model = CBraMod(
                dataset=dataset,
                n_patches=2,
                classifier=name,
                n_layer=1,
                input_scale=1.0,
                pretrained=False,
            )
            loader = get_dataloader(dataset, batch_size=2, shuffle=False)
            out = model(**next(iter(loader)))
            self.assertEqual(out["logit"].shape[0], 2)

    def test_encode_and_pool(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        signal = torch.randn(2, 16, 1000)
        tokens = model.encode(signal)
        self.assertEqual(tokens.shape, (2, 16, 5, 200))
        pooled = model.pool(tokens)
        self.assertEqual(pooled.shape, (2, 200))

    def test_embed_flag(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch["embed"] = True
        out = model(**batch)
        self.assertIn("embed", out)
        self.assertIn("patch_tokens", out)

    def test_backward(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        out = model(**next(iter(get_dataloader(dataset, batch_size=2))))
        out["loss"].backward()
        has_grad = any(
            p.grad is not None for p in model.backbone.parameters() if p.requires_grad
        )
        self.assertTrue(has_grad)

    def test_label_smoothing(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            label_smoothing=0.1,
            input_scale=1.0,
            pretrained=False,
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        out_smooth = model(**batch)
        model.label_smoothing = 0.0
        out_plain = model(**batch)
        self.assertFalse(torch.allclose(out_smooth["loss"], out_plain["loss"]))

    def test_input_scale(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=1,
            n_layer=2,
            input_scale=100.0,
            pretrained=False,
        )
        scaled = model._prepare_patches(torch.full((1, 16, 200), 100.0))
        self.assertAlmostEqual(scaled.mean().item(), 1.0, places=5)
        model.input_scale = 1.0
        raw = model._prepare_patches(torch.full((1, 16, 200), 100.0))
        self.assertAlmostEqual(raw.mean().item(), 100.0, places=5)

    def test_invalid_length(self):
        dataset = self._multiclass_dataset(n_time=1001)
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        with self.assertRaises(ValueError):
            model.encode(torch.randn(1, 16, 1001))

    def test_save_load_roundtrip(self):
        dataset = self._multiclass_dataset()
        model = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        state = model.state_dict()
        model2 = CBraMod(
            dataset=dataset,
            n_patches=5,
            n_layer=2,
            input_scale=1.0,
            pretrained=False,
        )
        model2.load_state_dict(state)
        model.eval()
        model2.eval()
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        with torch.no_grad():
            a = model(**batch)["logit"]
            b = model2(**batch)["logit"]
        self.assertTrue(torch.allclose(a, b))

    def test_load_pretrained_local(self):
        dataset = self._multiclass_dataset()
        backbone = CBraModBackbone(n_layer=2)
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            wrapped = {f"module.{k}": v for k, v in backbone.state_dict().items()}
            torch.save(wrapped, tmp.name)
            model = CBraMod(
                dataset=dataset,
                n_patches=5,
                n_layer=2,
                input_scale=1.0,
                pretrained=False,
            )
            model.load_pretrained_weights(tmp.name, strict=False)

    def test_load_pretrained_hf_mock(self):
        dataset = self._multiclass_dataset()
        backbone = CBraModBackbone(n_layer=2)
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            torch.save(backbone.state_dict(), tmp.name)
            model = CBraMod(
                dataset=dataset,
                n_patches=5,
                n_layer=2,
                input_scale=1.0,
                pretrained=False,
            )
            with patch(
                "huggingface_hub.hf_hub_download",
                return_value=tmp.name,
            ):
                model.load_pretrained_weights(checkpoint_path=None)

    def test_full_backbone_state_dict_keys(self):
        backbone = CBraModBackbone(n_layer=12)
        keys = set(backbone.state_dict().keys())
        self.assertTrue(any(k.startswith("patch_embedding.") for k in keys))
        layer_ids = {
            int(parts[2])
            for parts in (k.split(".") for k in keys)
            if len(parts) > 2 and parts[0] == "encoder" and parts[1] == "layers" and parts[2].isdigit()
        }
        self.assertEqual(layer_ids, set(range(12)))
        self.assertIn("proj_out.0.weight", keys)

    def test_build_classifier_head_shapes(self):
        head = build_classifier_head(
            "all_patch_reps", 16, 5, 200, 6, dropout=0.1
        )
        feats = torch.randn(2, 16, 5, 200)
        self.assertEqual(head(feats).shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
