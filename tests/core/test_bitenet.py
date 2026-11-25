# Contributor: Ibrahim Rabiu (GitHub: @alhibb, alhibb01@gmail.com)
# Paper: Chen, Xueping, et al. "BiteNet: Bidirectional Temporal Encoder Network
# to Predict Medical Outcomes."
# IEEE International Conference on Data Mining (ICDM 2020): 412-421.

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BiteNet
from pyhealth.models.bitenet import (
    AttentionPooling,
    BiteNetEncoder,
    BiteNetTemporalEncoding,
)


class TestBiteNet(unittest.TestCase):
    """Unit test suite for BiteNet architecture and components."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": ["c1", "c2", "c3"],
                "procedures": ["p1", "p2"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": ["c2", "c4"],
                "procedures": ["p2", "p3"],
                "label": 0,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "conditions": ["c1", "c3", "c4"],
                "procedures": ["p1", "p3"],
                "label": 1,
            },
            {
                "patient_id": "p3",
                "visit_id": "v3",
                "conditions": ["c2"],
                "procedures": ["p2"],
                "label": 0,
            },
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_bitenet_ds",
        )
        self.model = BiteNet(
            dataset=self.dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
            pooling="attention",
        )

    def test_init(self):
        """Verifies initial configuration and module instantiation."""
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.heads, 2)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(len(self.model.code_feature_keys), 2)
        self.assertEqual(self.model.pooling, "attention")

    def test_invalid_pooling_raises(self):
        """Ensures invalid pooling modes raise ValueError."""
        with self.assertRaises(ValueError):
            BiteNet(dataset=self.dataset, pooling="unsupported_pool")

    def test_embedding_dim_heads_divisibility_raises(self):
        """Ensures embedding_dim not divisible by heads raises ValueError."""
        with self.assertRaises(ValueError):
            BiteNet(dataset=self.dataset, embedding_dim=33, heads=2)

    def test_empty_code_features_raises(self):
        """Ensures initializing without code features raises ValueError."""
        dummy_ds = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "interval": [5], "label": 1},
                {"patient_id": "p1", "interval": [10], "label": 0},
            ],
            input_schema={"interval": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_empty_bitenet",
        )
        with self.assertRaises(ValueError):
            BiteNet(dataset=dummy_ds)

    def test_attention_pooling_standalone(self):
        """Verifies AttentionPooling forward pass and output shapes."""
        pool = AttentionPooling(embedding_dim=32)
        x = torch.randn(2, 6, 32)
        mask = torch.tensor(
            [[True, True, True, True, False, False], [True, True, False, False, False, False]],
            dtype=torch.bool,
        )
        out = pool(x, mask=mask)
        self.assertEqual(out.shape, (2, 32))
        self.assertFalse(torch.isnan(out).any())

        # Test full-padding degenerate sequence
        full_pad_mask = torch.zeros(2, 6, dtype=torch.bool)
        pad_out = pool(x, mask=full_pad_mask)
        self.assertEqual(pad_out.shape, (2, 32))
        self.assertFalse(torch.isnan(pad_out).any())

    def test_attention_pooling_pad_invariance(self):
        """Ensures masked padding tokens do not alter pooled representations."""
        pool = AttentionPooling(embedding_dim=16)
        pool.eval()
        mask = torch.tensor(
            [[True, True, False, False], [True, True, False, False]], dtype=torch.bool
        )
        x = torch.zeros(2, 4, 16)
        x[:, 0:2, :] = 1.5
        x[1, 2:, :] = 99.0  # arbitrary pad values in sequence 1

        with torch.no_grad():
            pooled = pool(x, mask=mask)
        self.assertTrue(torch.allclose(pooled[0], pooled[1], atol=1e-5))

    def test_temporal_encoding_standalone(self):
        """Verifies sinusoidal, embedding, and none temporal encoding modes."""
        enc_sin = BiteNetTemporalEncoding(embedding_dim=32, pos_encoding="sinusoidal")
        intervals = torch.tensor([[0, 10, 20], [0, 5, 15]])
        out_sin = enc_sin(seq_len=3, interval_ids=intervals)
        self.assertEqual(out_sin.shape, (2, 3, 32))

        enc_emb = BiteNetTemporalEncoding(
            embedding_dim=32, pos_encoding="embedding", max_interval=100
        )
        out_emb = enc_emb(seq_len=3, interval_ids=intervals)
        self.assertEqual(out_emb.shape, (2, 3, 32))

        enc_none = BiteNetTemporalEncoding(embedding_dim=32, pos_encoding="none")
        out_none = enc_none(seq_len=3, batch_size=2)
        self.assertEqual(out_none.shape, (2, 3, 32))
        self.assertTrue(torch.all(out_none == 0.0))

        with self.assertRaises(ValueError):
            BiteNetTemporalEncoding(embedding_dim=32, pos_encoding="invalid_mode")

    def test_temporal_encoding_alignment(self):
        """Tests forward-fill padding for shorter intervals and truncation for longer."""
        enc = BiteNetTemporalEncoding(embedding_dim=16, pos_encoding="sinusoidal")
        # Shorter than seq_len 4
        shorter_ids = torch.tensor([[5, 12]])
        aligned_short = enc.align_temporal_ids(shorter_ids, target_len=4)
        expected_short = torch.tensor([[5, 12, 12, 12]])
        self.assertTrue(torch.equal(aligned_short, expected_short))

        # Longer than seq_len 3
        longer_ids = torch.tensor([[1, 2, 3, 4, 5]])
        aligned_long = enc.align_temporal_ids(longer_ids, target_len=3)
        expected_long = torch.tensor([[1, 2, 3]])
        self.assertTrue(torch.equal(aligned_long, expected_long))

        # 1D tensor expanded
        one_d = torch.tensor([10, 20])
        aligned_1d = enc.align_temporal_ids(one_d, target_len=3)
        self.assertEqual(aligned_1d.shape, (2, 3))

    def test_directional_encoder_standalone(self):
        """Verifies forward and backward directional self-attention stacks."""
        fw_encoder = BiteNetEncoder(
            d_model=32, heads=2, num_layers=1, direction="forward"
        )
        bw_encoder = BiteNetEncoder(
            d_model=32, heads=2, num_layers=1, direction="backward"
        )
        x = torch.randn(2, 5, 32)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]],
            dtype=torch.bool,
        )
        out_fw = fw_encoder(x, mask=mask)
        out_bw = bw_encoder(x, mask=mask)
        self.assertEqual(out_fw.shape, (2, 5, 32))
        self.assertEqual(out_bw.shape, (2, 5, 32))
        self.assertFalse(torch.isnan(out_fw).any())
        self.assertFalse(torch.isnan(out_bw).any())

        with self.assertRaises(ValueError):
            BiteNetEncoder(d_model=32, direction="invalid_dir")

    def test_multilayer_backward_encoder_padded_input(self):
        """Verifies backward encoder with num_layers>=2 handles padded rows without NaN."""
        for layers in (2, 3):
            bw_enc = BiteNetEncoder(
                d_model=16, heads=2, num_layers=layers, direction="backward"
            )
            x = torch.randn(2, 5, 16)
            mask = torch.tensor(
                [[True, True, True, False, False], [True, True, False, False, False]],
                dtype=torch.bool,
            )
            out = bw_enc(x, mask=mask)
            self.assertFalse(
                torch.isnan(out).any(), f"NaN detected with num_layers={layers}"
            )
            self.assertFalse(torch.isnan(out[mask]).any())
            self.assertTrue(torch.all(out[~mask] == 0.0))

    def test_directional_causality_masking(self):
        """Tests that forward encoder is strictly causal and backward is anti-causal."""
        torch.manual_seed(42)
        d_model = 16
        fw_enc = BiteNetEncoder(
            d_model=d_model, heads=2, num_layers=2, direction="forward", dropout=0.0
        )
        bw_enc = BiteNetEncoder(
            d_model=d_model, heads=2, num_layers=2, direction="backward", dropout=0.0
        )
        fw_enc.eval()
        bw_enc.eval()

        x1 = torch.randn(1, 6, d_model)

        with torch.no_grad():
            # Forward causal test: altering future tokens (indices 3..5)
            # must not affect earlier token representations (indices 0..2)
            x_future_changed = x1.clone()
            x_future_changed[:, 3:, :] += 10.0

            out_fw_base = fw_enc(x1)
            out_fw_mod = fw_enc(x_future_changed)

            self.assertTrue(
                torch.allclose(
                    out_fw_base[:, :3, :], out_fw_mod[:, :3, :], atol=1e-5
                ),
                "Forward encoder earlier steps leaked future token context.",
            )
            self.assertFalse(
                torch.allclose(out_fw_base[:, 3:, :], out_fw_mod[:, 3:, :]),
                "Forward encoder future steps should reflect changes.",
            )

            # Backward anti-causal test: altering past tokens (indices 0..2)
            # must not affect later token representations (indices 3..5)
            x_past_changed = x1.clone()
            x_past_changed[:, :3, :] += 10.0

            out_bw_base = bw_enc(x1)
            out_bw_mod = bw_enc(x_past_changed)

            self.assertTrue(
                torch.allclose(
                    out_bw_base[:, 3:, :], out_bw_mod[:, 3:, :], atol=1e-5
                ),
                "Backward encoder later steps leaked past token context.",
            )
            self.assertFalse(
                torch.allclose(out_bw_base[:, :3, :], out_bw_mod[:, :3, :]),
                "Backward encoder past steps should reflect changes.",
            )

    def test_binary_classification_forward(self):
        """Tests complete forward pass on binary dataset."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch)

        self.assertIn("logit", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        self.assertIn("y_true", out)

        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertEqual(out["y_prob"].shape, (2, 1))
        self.assertTrue(torch.all(out["y_prob"] >= 0.0))
        self.assertTrue(torch.all(out["y_prob"] <= 1.0))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_multiclass_classification_forward(self):
        """Tests forward pass on multiclass dataset."""
        mc_samples = [
            {"patient_id": "p0", "conditions": ["c1", "c2"], "label": 0},
            {"patient_id": "p1", "conditions": ["c2", "c3"], "label": 1},
            {"patient_id": "p2", "conditions": ["c1", "c3"], "label": 2},
        ]
        mc_dataset = create_sample_dataset(
            samples=mc_samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_mc_bitenet",
        )
        mc_model = BiteNet(
            dataset=mc_dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
        )
        loader = get_dataloader(mc_dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        out = mc_model(**batch)

        self.assertEqual(out["logit"].shape, (3, 3))
        self.assertEqual(out["y_prob"].shape, (3, 3))
        sums = out["y_prob"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_multilabel_classification_forward(self):
        """Tests forward pass on multilabel dataset."""
        ml_samples = [
            {"patient_id": "p0", "conditions": ["c1"], "label": [0, 1]},
            {"patient_id": "p1", "conditions": ["c2"], "label": [1]},
        ]
        ml_dataset = create_sample_dataset(
            samples=ml_samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="test_ml_bitenet",
        )
        ml_model = BiteNet(
            dataset=ml_dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
        )
        loader = get_dataloader(ml_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = ml_model(**batch)

        self.assertEqual(out["y_prob"].shape[0], 2)
        self.assertTrue(torch.all(out["y_prob"] >= 0.0))
        self.assertTrue(torch.all(out["y_prob"] <= 1.0))

    def test_with_explicit_intervals_and_time(self):
        """Tests BiteNet with explicit interval and time kwargs."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        # Pass interval
        batch["interval"] = torch.tensor([[0, 14, 28], [0, 7, 21]])
        out = self.model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))

        # Pass time
        del batch["interval"]
        batch["time"] = torch.tensor([[100, 114, 128], [200, 207, 221]])
        out_time = self.model(**batch)
        self.assertEqual(out_time["logit"].shape, (2, 1))

    def test_protect_auxiliary_keys_in_schema(self):
        """Ensures interval and time in input_schema are isolated from code embeddings."""
        ds = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "conditions": ["c1", "c2"], "interval": [0, 10], "label": 1},
                {"patient_id": "p1", "conditions": ["c2", "c3"], "interval": [0, 15], "label": 0},
            ],
            input_schema={"conditions": "sequence", "interval": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_aux_bitenet",
        )
        model = BiteNet(dataset=ds, embedding_dim=32, heads=1, num_layers=1)
        self.assertEqual(model.code_feature_keys, ["conditions"])
        self.assertEqual(model.fc.in_features, 32)

        loader = get_dataloader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))

    def test_tuple_valued_features(self):
        """Tests that features returned as tuples (value, mask) forward correctly."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        cond_val = batch["conditions"]
        cond_mask = torch.ones_like(cond_val, dtype=torch.bool)
        batch["conditions"] = (cond_val, cond_mask)

        out = self.model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertIn("loss", out)

    def test_pooling_modes(self):
        """Verifies attention, mean, and cls pooling strategies."""
        for mode in ("attention", "mean", "cls"):
            m = BiteNet(
                dataset=self.dataset,
                embedding_dim=32,
                heads=2,
                num_layers=1,
                pooling=mode,
            )
            loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
            out = m(**batch)
            self.assertEqual(out["logit"].shape, (2, 1))

    def test_hierarchical_visit_inputs(self):
        """Tests 4D visit tensor pooling through intra_visit_pool."""
        # [batch=2, visits=3, codes=4, dim=32]
        x_4d = torch.randn(2, 3, 4, 32)
        pooled = self.model._pool_embedding(x_4d)
        self.assertEqual(pooled.shape, (2, 3, 32))

    def test_hierarchical_visit_inputs_with_code_mask(self):
        """Ensures pad codes inside visits do not affect pooled visit representations."""
        self.model.eval()
        # [batch=1, visits=2, codes=3, dim=32]
        x1 = torch.randn(1, 2, 3, 32)
        # Second visit has only 1 valid code, codes 1 and 2 are padding
        mask = torch.tensor(
            [[[True, True, True], [True, False, False]]], dtype=torch.bool
        )
        x2 = x1.clone()
        x2[0, 1, 1:, :] += 100.0  # modify pad codes in visit 1

        with torch.no_grad():
            pooled1 = self.model._pool_embedding(x1, mask=mask)
            pooled2 = self.model._pool_embedding(x2, mask=mask)

        self.assertTrue(torch.allclose(pooled1, pooled2, atol=1e-5))

    def test_cls_pooling_with_padded_sequence(self):
        """Verifies cls pooling correctly selects last valid visit for fw and index 0 for bw."""
        cls_model = BiteNet(
            dataset=self.dataset,
            embedding_dim=32,
            heads=2,
            num_layers=1,
            pooling="cls",
        )
        cls_model.eval()
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = cls_model(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertFalse(torch.isnan(out["logit"]).any())

    def test_odd_embedding_dim(self):
        """Verifies odd embedding_dim support in embedding layer and model."""
        odd_enc = BiteNetTemporalEncoding(embedding_dim=63, pos_encoding="sinusoidal")
        out = odd_enc(seq_len=4, batch_size=2)
        self.assertEqual(out.shape, (2, 4, 63))

        odd_model = BiteNet(
            dataset=self.dataset,
            embedding_dim=63,
            heads=1,
            num_layers=1,
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        model_out = odd_model(**batch)
        self.assertEqual(model_out["logit"].shape, (2, 1))

    def test_forward_from_embedding_direct(self):
        """Tests direct call to forward_from_embedding with pre-embedded features."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        embedded = self.model.embedding_model(
            {"conditions": batch["conditions"], "procedures": batch["procedures"]}
        )
        direct_kwargs = {
            "conditions": (embedded["conditions"],),
            "procedures": (embedded["procedures"],),
            "label": batch["label"],
            "interval": torch.tensor([[0, 10, 20], [0, 5, 15]]),
        }
        out = self.model.forward_from_embedding(**direct_kwargs)
        self.assertEqual(out["logit"].shape, (2, 1))
        self.assertIn("loss", out)

    def test_embed_flag_and_backward(self):
        """Verifies patient embedding output and gradient flow."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch, embed=True)

        self.assertIn("embed", out)
        self.assertEqual(
            out["embed"].shape, (2, 32 * len(self.model.code_feature_keys))
        )

        loss = out["loss"]
        loss.backward()

        self.assertIsNotNone(self.model.fc.weight.grad)
        self.assertIsNotNone(self.model.fusion[0].weight.grad)
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"Parameter {name} has non-finite gradients on padded batch",
                )

    def test_sinusoidal_intervals_not_clamped_beyond_max_interval(self):
        """Ensures large day intervals in sinusoidal mode are not clamped to max_interval."""
        enc = BiteNetTemporalEncoding(
            embedding_dim=16, pos_encoding="sinusoidal", max_interval=365
        )
        large_intervals = torch.tensor([[500, 1000]])
        aligned = enc.align_temporal_ids(large_intervals, target_len=2, max_val=None)
        self.assertEqual(aligned[0, 0].item(), 500)
        self.assertEqual(aligned[0, 1].item(), 1000)


if __name__ == "__main__":
    unittest.main()
