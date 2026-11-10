"""Author: Tamjit Kumar, Yi Wang
NetID: tkumar70, ywang4715
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
       in the Intensive Care Unit
Paper Link: https://doi.org/10.1145/3450439.3451860
Description: Unit tests for TPC model architecture, forward-filling & decay indicators,
             eICU length of stay task logic, and unconfounded batch sampling.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

# Ensure repository root is on sys.path regardless of execution directory
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pyhealth.datasets import (
    collate_fn_dict_with_padding,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.datasets.splitter import split_by_patient, split_by_visit
from pyhealth.models.tpc import (
    TPC,
    MyBatchNorm1d,
    TPCBlock,
    _align_mask,
    _align_target,
    _create_batchnorm_layers,
)
from pyhealth.tasks import LengthOfStayTPCeICU
from pyhealth.tasks.length_of_stay_tpc import (
    StayDataMap,
    compute_cache_fingerprint,
)
from pyhealth.tasks.length_of_stay_tpc_eicu import (
    CORE_VITALS_MAP,
    DEFAULT_LAB_NAMES,
    DEFAULT_VITALS_NORMALIZATION,
    BucketBatchSampler,
    FastReservoirSampler,
    MapDataset,
    compute_empirical_training_stats,
    compute_los_metrics,
    generate_tpc_tensors,
    get_valid_eicu_stay_ids,
    load_eicu_vitals_cache,
    select_qualifying_labs,
)
from pyhealth.trainer import Trainer

LengthOfStayTask = LengthOfStayTPCeICU


class MockDataset:
    """Mock PyHealth dataset to validate BaseModel compatibility."""

    def __init__(self):
        self.tokens = {"static": ["token1", "token2", "token3"]}
        self.input_schema = {
            "static": "tensor",
            "vitals": "tensor",
            "vitals_decay": "tensor",
            "mask": "tensor",
        }
        self.output_schema = {
            "los": "tensor",
            "mortality": "tensor",
        }

    def get_all_tokens(self, key):
        return self.tokens.get(key, [])

    def __getitem__(self, idx):
        return {
            "vitals": torch.randn(24, 16),
            "vitals_decay": torch.rand(24, 16),
            "static": torch.randn(44),
            "los": torch.ones(24, 1),
            "mortality": torch.tensor([0.0]),
        }


class MockPatient:
    """Minimal mock patient for unit testing eICU LengthOfStayTask logic."""

    def __init__(
        self,
        patient_id: str = "p0",
        age: float = 50.0,
        stay_hours: int = 24,
        has_lab: bool = True,
        unittype: str = "Med-Surg ICU",
        unitadmitsource: str = "Emergency Department",
        unitstaytype: str = "admit",
        ethnicity: str = "Caucasian",
    ):
        self.patient_id = patient_id
        discharge_offset = float(stay_hours * 60)  # minutes

        self.events = {
            "patient": [
                type(
                    "Event",
                    (),
                    {
                        "patientunitstayid": "s0",
                        "unitdischargeoffset": discharge_offset,
                        "gender": "Female",
                        "age": str(age),
                        "ethnicity": ethnicity,
                        "admissionheight": 165.0,
                        "admissionweight": 65.0,
                        "unittype": unittype,
                        "unitadmittime24": "14:00:00",
                        "unitadmitsource": unitadmitsource,
                        "unitvisitnumber": 1,
                        "unitstaytype": unitstaytype,
                        "hospitaldischargestatus": "Alive",
                        "unitdischargestatus": "Alive",
                    },
                )()
            ],
            "lab": [
                type(
                    "Event",
                    (),
                    {
                        "patientunitstayid": "s0",
                        "labresultoffset": 120.0,  # 2 hours into stay
                        "labname": "potassium",
                        "labresult": 4.2,
                    },
                )()
            ]
            if has_lab
            else [],
        }

    def get_events(self, event_type: str):
        return self.events.get(event_type, [])


class TestTPCModel(unittest.TestCase):
    """Unit tests for the TPC model architecture and forward pass."""

    def test_tpc_sequence_forward_and_loss(self):
        """Tests sequence-level hourly LoS prediction and MSLE loss."""
        for batch_size, seq_len, num_vitals, num_static in [
            (2, 24, 12, 2),
            (4, 48, 12, 2),
        ]:
            mock_dataset: Any = MockDataset()
            vitals = torch.randn(batch_size, seq_len, num_vitals)
            vitals_time_delta = torch.rand(batch_size, seq_len, num_vitals) * 5.0
            static = torch.randn(batch_size, num_static)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
            mock_labels = torch.rand(batch_size, seq_len, 1) * 10.0 + 0.1

            model = TPC(
                dataset=mock_dataset,
                continuous_dim=num_vitals,
                static_dim=num_static,
                n_layers=3,
                temp_dim=8,
                point_dim=16,
                kernel_size=3,
                loss_fn="msle",
            )

            output = model(
                static=static,
                vitals=vitals,
                vitals_time_delta=vitals_time_delta,
                mask=mask,
                y_true=mock_labels,
            )

            self.assertIn("loss", output)
            self.assertIn("y_prob", output)
            self.assertEqual(output["y_prob"].shape, (batch_size, seq_len, 1))
            self.assertFalse(torch.isnan(output["loss"]))
            self.assertTrue((output["y_prob"] > 0).all())

            # Test transposed labels (B, 1, T)
            transposed_labels = mock_labels.transpose(1, 2)
            out_trans = model(
                vitals=vitals,
                vitals_time_delta=vitals_time_delta,
                static=static,
                y_true=transposed_labels,
                mask=mask,
            )
            self.assertIn("loss", out_trans)
            self.assertFalse(torch.isnan(out_trans["loss"]))

    def test_tpc_causality(self):
        """Verifies strict causal convolution without future lookahead."""
        batch_size = 1
        seq_len = 24
        num_vitals = 12

        for bn in ["default", "none"]:
            model = TPC(
                dataset=None,
                continuous_dim=num_vitals,
                static_dim=0,
                n_layers=3,
                kernel_size=3,
                batchnorm=bn,
            )
            model.eval()

            vitals1 = torch.randn(batch_size, seq_len, num_vitals)
            decay1 = torch.rand(batch_size, seq_len, num_vitals)
            vitals2 = vitals1.clone()
            decay2 = decay1.clone()

            # Alter future timesteps starting at t = 16
            perturb_step = 16
            vitals2[:, perturb_step:, :] += 50.0
            decay2[:, perturb_step:, :] = 0.0

            with torch.no_grad():
                out1 = model(vitals=vitals1, vitals_decay=decay1)["y_prob"]
                out2 = model(vitals=vitals2, vitals_decay=decay2)["y_prob"]

            past_diff = (
                (out1[:, :perturb_step, :] - out2[:, :perturb_step, :])
                .abs()
                .max()
                .item()
            )
            future_diff = (
                (out1[:, perturb_step:, :] - out2[:, perturb_step:, :])
                .abs()
                .max()
                .item()
            )

            self.assertLess(past_diff, 1e-6)
            self.assertGreater(future_diff, 0.0)

    def test_tpc_padding_invariance(self):
        """Verifies that zero-padding steps do not contaminate loss."""
        num_vitals = 12
        model = TPC(dataset=None, continuous_dim=num_vitals)
        model.eval()

        # Patient 1: real stay of 10 hours
        T_valid = 10
        vitals_valid = torch.randn(1, T_valid, num_vitals)
        decay_valid = torch.rand(1, T_valid, num_vitals)
        y_true_valid = torch.rand(1, T_valid, 1) + 0.5
        mask_valid = torch.ones(1, T_valid, dtype=torch.bool)

        loss_unpadded = model(
            vitals=vitals_valid,
            vitals_decay=decay_valid,
            y_true=y_true_valid,
            mask=mask_valid,
        )["loss"].item()

        # Patient 1 padded with 10 zero-padding hours (total T = 20)
        T_padded = 20
        vitals_padded = torch.zeros(1, T_padded, num_vitals)
        vitals_padded[:, :T_valid, :] = vitals_valid

        decay_padded = torch.zeros(1, T_padded, num_vitals)
        decay_padded[:, :T_valid, :] = decay_valid

        y_true_padded = torch.zeros(1, T_padded, 1)
        y_true_padded[:, :T_valid, :] = y_true_valid

        mask_padded = torch.zeros(1, T_padded, dtype=torch.bool)
        mask_padded[:, :T_valid] = True

        with torch.no_grad():
            out_padded = model(
                vitals=vitals_padded,
                vitals_decay=decay_padded,
                y_true=y_true_padded,
                mask=mask_padded,
            )
            loss_padded = model._compute_loss(
                out_padded["y_prob"], y_true_padded, mask=mask_padded
            ).item()

        self.assertFalse(torch.isnan(torch.tensor(loss_padded)))
        self.assertGreater(loss_padded, 0.0)
        self.assertAlmostEqual(loss_padded, loss_unpadded, places=5)

    def test_generate_tpc_tensors(self):
        """Tests forward-filling and exponential decay indicator calculation."""
        raw = torch.tensor(
            [
                [10.0, 0.0],
                [0.0, 5.0],
                [0.0, 0.0],
                [20.0, 0.0],
            ]
        )
        mask = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        )

        vitals_ff, decay = generate_tpc_tensors(raw, mask)

        self.assertEqual(vitals_ff[0, 0].item(), 10.0)
        self.assertEqual(vitals_ff[1, 0].item(), 10.0)
        self.assertEqual(vitals_ff[2, 0].item(), 10.0)
        self.assertEqual(vitals_ff[3, 0].item(), 20.0)

        self.assertAlmostEqual(decay[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(decay[1, 0].item(), 0.75, places=5)
        self.assertAlmostEqual(decay[2, 0].item(), 0.5625, places=5)
        self.assertAlmostEqual(decay[3, 0].item(), 1.0, places=5)

        self.assertEqual(decay[0, 1].item(), 0.0)
        self.assertEqual(decay[1, 1].item(), 1.0)

    def test_tpc_collate_and_training_padding(self):
        """Tests PyHealth collation padding and model training stability."""
        num_vitals = 12
        T1, T2 = 8, 16

        s1 = {
            "vitals": torch.randn(T1, num_vitals),
            "vitals_decay": torch.rand(T1, num_vitals),
            "static": torch.tensor([0.5, 1.0]),
            "mask": torch.ones(T1, dtype=torch.bool),
            "los": torch.rand(T1, 1) + 0.1,
        }
        s2 = {
            "vitals": torch.randn(T2, num_vitals),
            "vitals_decay": torch.rand(T2, num_vitals),
            "static": torch.tensor([0.6, 0.0]),
            "mask": torch.ones(T2, dtype=torch.bool),
            "los": torch.rand(T2, 1) + 0.1,
        }

        batch = collate_fn_dict_with_padding([s1, s2])

        self.assertEqual(batch["mask"].shape, (2, T2))
        self.assertEqual(batch["mask"].sum(dim=1).tolist(), [T1, T2])
        self.assertTrue((batch["mask"][0, T1:] == False).all())
        self.assertTrue((batch["los"][0, T1:] == 0.0).all())
        self.assertTrue((batch["vitals"][0, T1:] == 0.0).all())

        model = TPC(dataset=None, continuous_dim=num_vitals, static_dim=2)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        out = model(**batch)
        self.assertIn("loss", out)
        self.assertFalse(torch.isnan(out["loss"]))
        out["loss"].backward()
        optimizer.step()

    def test_tpc_full_gradient_flow(self):
        """Verifies that gradients flow to all model parameters without NaN."""
        batch_size = 2
        seq_len = 16
        num_vitals = 6
        static_dim = 4

        model = TPC(
            dataset=None,
            continuous_dim=num_vitals,
            static_dim=static_dim,
            n_layers=2,
            aux_label_key="mortality",
        )
        model.train()

        vitals = torch.randn(batch_size, seq_len, num_vitals)
        vitals_decay = torch.rand(batch_size, seq_len, num_vitals)
        static = torch.randn(batch_size, static_dim)
        y_true = torch.rand(batch_size, seq_len, 1) + 0.1
        mortality = torch.randint(0, 2, (batch_size, 1)).float()

        out = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            static=static,
            y_true=y_true,
            mortality=mortality,
        )
        self.assertIn("loss", out)
        out["loss"].backward()

        for name, param in model.named_parameters():
            if name != "_dummy_param" and param.requires_grad:
                self.assertIsNotNone(
                    param.grad, f"Parameter '{name}' received no gradient."
                )
                assert param.grad is not None
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"Parameter '{name}' received NaN gradient.",
                )

    def test_tpc_multitask(self):
        """Tests multitask learning with both LoS and in-hospital mortality."""
        batch_size = 2
        seq_len = 24
        num_vitals = 12

        model = TPC(
            dataset=None,
            continuous_dim=num_vitals,
            aux_label_key="mortality",
            aux_weight=0.5,
        )

        vitals = torch.randn(batch_size, seq_len, num_vitals)
        vitals_decay = torch.rand(batch_size, seq_len, num_vitals)
        y_true = torch.rand(batch_size, seq_len, 1) * 5.0 + 0.1
        mortality = torch.randint(0, 2, (batch_size, 1)).float()

        output = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            y_true=y_true,
            mortality=mortality,
        )

        self.assertIn("loss", output)
        self.assertIn("y_prob", output)
        self.assertIn("aux_prob", output)
        self.assertIn("aux_logit", output)
        self.assertNotIn("mortality_prob", output)
        self.assertEqual(output["aux_prob"].shape, (batch_size, seq_len, 1))
        self.assertTrue(
            ((output["aux_prob"] >= 0.0) & (output["aux_prob"] <= 1.0)).all()
        )

        # Test with 3D mask (B, T, 1)
        mask_3d = torch.ones(batch_size, seq_len, 1, dtype=torch.bool)
        output_3d = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            y_true=y_true,
            mortality=mortality,
            mask=mask_3d,
        )
        self.assertIn("loss", output_3d)
        self.assertFalse(torch.isnan(output_3d["loss"]))

        # Test with sequence mortality (B, T) and scalar y_true (B,)
        mort_seq = torch.randint(0, 2, (batch_size, seq_len)).float()
        y_scalar = torch.tensor([2.5, 3.5])
        out_seq = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            y_true=y_scalar,
            mortality=mort_seq,
        )
        self.assertIn("loss", out_seq)
        self.assertFalse(torch.isnan(out_seq["loss"]))
        self.assertEqual(out_seq["y_prob"].shape, (batch_size, seq_len, 1))

    def test_tpc_aux_default_weight_and_unique_state_dict(self):
        """Tests that aux_weight defaults to 100.0 and state_dict has unique keys."""
        batch_size = 2
        seq_len = 24
        num_vitals = 6

        model = TPC(
            dataset=None,
            continuous_dim=num_vitals,
            aux_label_key="mortality",
        )
        self.assertEqual(model.aux_weight, 100.0)

        # Ensure state_dict has strictly unique keys and no duplicate module aliases
        state_keys = list(model.state_dict().keys())
        self.assertEqual(len(state_keys), len(set(state_keys)))
        self.assertTrue(any(k.startswith("aux_") for k in state_keys))
        self.assertFalse(any("mort" in k for k in state_keys))

        # Test with explicit aux_weight override
        model_custom = TPC(
            dataset=None,
            continuous_dim=num_vitals,
            aux_label_key="mortality",
            aux_weight=50.0,
        )
        self.assertEqual(model_custom.aux_weight, 50.0)

        # Test forward loss computation with default aux_weight=100.0
        vitals = torch.randn(batch_size, seq_len, num_vitals)
        vitals_decay = torch.rand(batch_size, seq_len, num_vitals)
        y_true = torch.rand(batch_size, seq_len, 1) * 3.0 + 0.1
        mortality = torch.randint(0, 2, (batch_size, 1)).float()

        out = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            y_true=y_true,
            mortality=mortality,
        )
        self.assertIn("loss", out)
        self.assertIn("aux_prob", out)
        self.assertNotIn("mortality_prob", out)
        self.assertFalse(torch.isnan(out["loss"]))

    def test_tpc_no_decay_ablation(self):
        """Tests TPC ablation variant without decay channels (use_decay=False)."""
        batch_size = 2
        seq_len = 16
        num_vitals = 12

        model = TPC(dataset=None, continuous_dim=num_vitals, use_decay=False)
        vitals = torch.randn(batch_size, seq_len, num_vitals)
        y_true = torch.rand(batch_size, seq_len, 1) + 0.1
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        self.assertEqual(model.blocks[0].C_n, 1)
        self.assertEqual(model.blocks[0].P_n, num_vitals * 1)

        out = model(vitals=vitals, y_true=y_true, mask=mask)
        self.assertEqual(out["y_prob"].shape, (batch_size, seq_len, 1))
        self.assertFalse(torch.isnan(out["loss"]))

    def test_tpc_save_and_load_state(self):
        """Verifies that model state can be saved and loaded accurately."""
        model = TPC(dataset=None, continuous_dim=12, static_dim=5)
        model.eval()
        vitals = torch.randn(2, 24, 12)
        vitals_decay = torch.rand(2, 24, 12)
        static = torch.randn(2, 5)

        with torch.no_grad():
            out1 = model(vitals=vitals, vitals_decay=vitals_decay, static=static)[
                "y_prob"
            ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            weight_path = Path(tmp_dir) / "tpc_weights.pt"
            torch.save(model.state_dict(), weight_path)

            loaded_model = TPC(dataset=None, continuous_dim=12, static_dim=5)
            loaded_model.load_state_dict(torch.load(weight_path, weights_only=True))
            loaded_model.eval()

            with torch.no_grad():
                out2 = loaded_model(
                    vitals=vitals, vitals_decay=vitals_decay, static=static
                )["y_prob"]

        torch.testing.assert_close(out1, out2)

    def test_tpc_dataset_dimension_inference(self):
        """Verifies input dimensions are inferred from dataset and logit is returned."""
        ds: Any = MockDataset()
        model = TPC(dataset=ds)
        self.assertEqual(model.continuous_dim, 16)
        self.assertEqual(model.static_dim, 44)
        self.assertIn("vitals", model.feature_keys)
        self.assertIn("vitals_decay", model.feature_keys)
        self.assertIn("static", model.feature_keys)

        out = model(
            vitals=torch.randn(2, 24, 16),
            vitals_decay=torch.rand(2, 24, 16),
            static=torch.randn(2, 44),
        )
        self.assertIn("y_prob", out)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (2, 24, 1))

    def test_tpc_static_1d_broadcast(self):
        """Verifies 1D static tensor is safely broadcast across mini-batch."""
        model = TPC(dataset=None, continuous_dim=12, static_dim=5)
        vitals = torch.randn(3, 20, 12)
        vitals_decay = torch.rand(3, 20, 12)
        static_1d = torch.randn(5)
        out = model(vitals=vitals, vitals_decay=vitals_decay, static=static_1d)
        self.assertEqual(out["y_prob"].shape, (3, 20, 1))

    def test_tpc_input_validation_raises_type_error(self):
        """Verifies that non-Tensor vitals input raises TypeError."""
        model = TPC(continuous_dim=12, static_dim=0)
        with self.assertRaises(TypeError):
            model(vitals="invalid_type")

    def test_tpc_loss_fn_validation(self):
        """Verifies unsupported loss_fn raises ValueError."""
        with self.assertRaises(ValueError):
            TPC(dataset=None, continuous_dim=6, static_dim=0, loss_fn="invalid_loss")
        with self.assertRaises(ValueError):
            TPC(dataset=None, continuous_dim=6, static_dim=0, loss_fn="mae")
        m1 = TPC(dataset=None, continuous_dim=6, static_dim=0, loss_fn="msle")
        self.assertEqual(m1.loss_fn_type, "msle")
        m2 = TPC(dataset=None, continuous_dim=6, static_dim=0, loss_fn="mse")
        self.assertEqual(m2.loss_fn_type, "mse")

    def test_tpc_base_model_schema_initialization(self):
        """Verifies feature_keys, label_keys, and mode with dataset=None."""
        m_default = TPC(dataset=None, continuous_dim=6, static_dim=4, time_dim=2)
        self.assertEqual(
            m_default.feature_keys,
            ["vitals", "vitals_decay", "static", "time_features"],
        )
        self.assertEqual(m_default.label_keys, ["los"])
        self.assertEqual(m_default.label_key, "los")
        self.assertEqual(m_default.mode, "regression")

        m_custom = TPC(
            dataset=None,
            continuous_dim=6,
            static_dim=0,
            use_decay=False,
            feature_keys=["custom_vitals"],
            label_key="target_los",
        )
        self.assertEqual(m_custom.feature_keys, ["custom_vitals"])
        self.assertEqual(m_custom.label_keys, ["target_los"])
        self.assertEqual(m_custom.label_key, "target_los")

    def test_tpc_warmup_mask_mixed_and_toy_batches(self):
        """Tests that short stays in mixed batches are not unmasked on warmup hours,
        while all-short toy batches safely trigger the valid_mask fallback."""
        from pyhealth.models.tpc import MyBatchNorm1d

        self.assertTrue(issubclass(MyBatchNorm1d, torch.nn.Module))

        model = TPC(dataset=None, continuous_dim=4, static_dim=0, min_time=4)
        vitals = torch.randn(2, 10, 4)
        vitals_decay = torch.rand(2, 10, 4)
        # stay 0: 3 hours (all warmup), stay 1: 10 hours (6 post-warmup)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, :3] = True
        mask[1, :10] = True
        yt = torch.rand(2, 10, 1) + 0.1

        out = model(vitals=vitals, vitals_decay=vitals_decay, mask=mask, y_true=yt)
        eval_mask = out["mask"].squeeze(-1)
        # stay 0 should have 0 eval hours; stay 1 should have 6 eval hours
        self.assertEqual(int(eval_mask[0].sum().item()), 0)
        self.assertEqual(int(eval_mask[1].sum().item()), 6)
        self.assertFalse(torch.isnan(out["loss"]))

        # Toy batch where ALL stays are <= min_time (length 4)
        mask_toy = torch.zeros(2, 4, dtype=torch.bool)
        mask_toy[0, :3] = True
        mask_toy[1, :4] = True
        out_toy = model(
            vitals=vitals[:, :4, :],
            vitals_decay=vitals_decay[:, :4, :],
            mask=mask_toy,
            y_true=yt[:, :4, :],
        )
        eval_mask_toy = out_toy["mask"].squeeze(-1)
        # Batch-level fallback should unmask valid hours for toy tests
        self.assertEqual(int(eval_mask_toy[0].sum().item()), 3)
        self.assertEqual(int(eval_mask_toy[1].sum().item()), 4)
        self.assertFalse(torch.isnan(out_toy["loss"]))

    def test_tpc_batchnorm_variants(self):
        """Tests all batchnorm variants from Rocheteau et al. (tpc_model.py)."""
        vitals = torch.randn(2, 16, 6)
        vitals_decay = torch.rand(2, 16, 6)
        static = torch.randn(2, 4)
        for bn in [
            "mybatchnorm",
            "default",
            "pointonly",
            "temponly",
            "low_momentum",
            "none",
        ]:
            model = TPC(
                dataset=None,
                continuous_dim=6,
                static_dim=4,
                batchnorm=bn,
                aux_label_key="mortality",
            )
            model.train()
            out_tr = model(vitals=vitals, vitals_decay=vitals_decay, static=static)
            self.assertEqual(out_tr["y_prob"].shape, (2, 16, 1))

            model.eval()
            out_ev = model(vitals=vitals, vitals_decay=vitals_decay, static=static)
            self.assertEqual(out_ev["y_prob"].shape, (2, 16, 1))

    def test_tpc_time_features(self):
        """Tests forward pass with non-convolved temporal context features."""
        vitals = torch.randn(2, 20, 6)
        vitals_decay = torch.rand(2, 20, 6)
        time_feat = torch.rand(2, 20, 2)
        model = TPC(
            dataset=None,
            continuous_dim=6,
            static_dim=0,
            time_dim=2,
            n_layers=2,
        )
        out = model(vitals=vitals, vitals_decay=vitals_decay, time_features=time_feat)
        self.assertEqual(out["y_prob"].shape, (2, 20, 1))


class TestTPCTask(unittest.TestCase):
    """Unit tests for the LengthOfStayTPCeICU task pipeline and batching."""

    def test_task_filters_and_target(self):
        """Tests task filters (age > 18, stay >= 5h, has obs) and target format."""
        task = LengthOfStayTask(lab_names=["potassium"])

        # Adult with valid stay and observation
        p_valid = MockPatient(age=50, stay_hours=10, has_lab=True)
        samples = task(p_valid)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["los"].shape, (10, 1))
        self.assertEqual(samples[0]["mask"].shape, (10,))
        self.assertEqual(samples[0]["vitals"].shape, (10, 7))
        self.assertEqual(task.continuous_dim, 7)
        self.assertEqual(len(task.vital_keys), 6)
        # Target remaining LoS should be strictly descending
        self.assertGreater(samples[0]["los"][0].item(), samples[0]["los"][-1].item())

        # Child (< 18) should be skipped
        p_child = MockPatient(age=16, stay_hours=10, has_lab=True)
        self.assertEqual(len(task(p_child)), 0)

        # Short stay (< 5h) should be skipped
        p_short = MockPatient(age=50, stay_hours=3, has_lab=True)
        self.assertEqual(len(task(p_short)), 0)

        # Stay with no observations should be skipped
        p_nolab = MockPatient(age=50, stay_hours=10, has_lab=False)
        self.assertEqual(len(task(p_nolab)), 0)

        # Missing or empty age string should be skipped
        p_empty_age = MockPatient(age=50, stay_hours=10, has_lab=True)
        p_empty_age.events["patient"][0].age = ""
        self.assertEqual(len(task(p_empty_age)), 0)

        p_null_age = MockPatient(age=50, stay_hours=10, has_lab=True)
        p_null_age.events["patient"][0].age = None
        self.assertEqual(len(task(p_null_age)), 0)

    def test_static_features_deterministic_and_one_hot(self):
        """Tests that static feature extraction is deterministic and valid one-hot."""
        patient = MockPatient(
            patient_id="p1",
            age=65.0,
            stay_hours=24,
            unittype="MICU",
            unitadmitsource="Emergency Department",
            unitstaytype="admit",
            ethnicity="Caucasian",
        )

        task = LengthOfStayTask()

        # 1. Deterministic across two task calls
        res1 = task(patient)
        res2 = task(patient)
        self.assertEqual(len(res1), 1)
        self.assertEqual(len(res2), 1)
        static1 = res1[0]["static"]
        static2 = res2[0]["static"]
        self.assertTrue(torch.equal(static1, static2))
        self.assertEqual(static1.shape[0], task.static_dim)

        # 2. Known category lands in exact assigned slot
        known_cats = {
            "unittype": "MICU",
            "unitadmitsource": "Emergency Department",
            "unitstaytype": "admit",
            "ethnicity": "Caucasian",
        }
        offset = 11  # continuous & indicator features come first
        for cat_name, vocab in task.cat_vocabs.items():
            block_len = len(vocab) + 1  # includes unknown slot
            block = static1[offset : offset + block_len]
            self.assertAlmostEqual(block.sum().item(), 1.0, places=5)
            self.assertTrue(((block == 0.0) | (block == 1.0)).all())
            val = known_cats[cat_name]
            self.assertEqual(static1[offset + vocab.index(val)].item(), 1.0)
            self.assertEqual(static1[offset + len(vocab)].item(), 0.0)
            offset += block_len
        self.assertEqual(offset, task.static_dim)

        # 3. Unknown category sets unknown slot and leaves known slots 0.0
        patient_unk = MockPatient(
            patient_id="p_unk",
            age=65.0,
            stay_hours=24,
            unittype="UNSEEN_UNIT",
            unitadmitsource="UNSEEN_SOURCE",
            unitstaytype="UNSEEN_STAYTYPE",
            ethnicity="UNSEEN_ETHNICITY",
        )
        res_unk = task(patient_unk)
        static_unk = res_unk[0]["static"]

        offset = 11
        for cat_name, vocab in task.cat_vocabs.items():
            block_len = len(vocab) + 1
            block = static_unk[offset : offset + block_len]
            self.assertAlmostEqual(block.sum().item(), 1.0, places=5)
            self.assertEqual(block[-1].item(), 1.0)
            self.assertTrue((block[:-1] == 0.0).all())
            offset += block_len

    def test_baseline_values_outside_window_ignored(self):
        """Tests that baseline measurements outside [-24h, +5h] are ignored."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_content = (
                "patientunitstayid,physicalexamoffset,physicalexampath,"
                "physicalexamtext\n"
                "stay_100,-1560,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/2,2\n"
                "stay_100,-120,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/3,3\n"
                "stay_100,60,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/4,4\n"
                "stay_100,360,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Eyes Score/1,1\n"
            )
            pe_file = tmp_path / "physicalExam.csv"
            pe_file.write_text(csv_content)

            _, baseline_static_by_stay = load_eicu_vitals_cache(str(tmp_path))

            self.assertIn("stay_100", baseline_static_by_stay)
            base = baseline_static_by_stay["stay_100"]
            self.assertIn("eyes", base)
            self.assertEqual(base["eyes"], 4.0)

    def test_select_qualifying_labs_respects_stay_window(self):
        """Tests that select_qualifying_labs counts labs within ICU window."""
        stay_minutes = 48 * 60  # 48 hours = 2880 minutes
        p = MockPatient(patient_id="p_labs", age=40.0, stay_hours=48, has_lab=False)

        def ev(name, minutes):
            return type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": minutes,
                    "labname": name,
                    "labresult": 1.0,
                },
            )()

        labs = []
        for item in ["A1", "A2", "A3", "A4", "A5"]:
            labs += [ev(item, m) for m in (60, 300, 600, 1200)]
        labs += [ev("X", 600)]  # 1 in-window obs
        labs += [ev("X", -m) for m in (60, 120, 180, 240, 300)]
        labs += [ev("X", stay_minutes + m) for m in (60, 120, 180, 240, 300)]
        p.events["lab"] = labs

        ds: Any = type("MockDS", (), {"iter_patients": lambda self: [p]})()
        selected, met = select_qualifying_labs(
            ds,
            train_pids={"p_labs"},
            min_prevalence=0.25,
            min_avg_obs=3.0,
            max_labs=45,
        )
        self.assertTrue(met)
        self.assertEqual(set(selected), {"A1", "A2", "A3", "A4", "A5"})
        self.assertNotIn("X", selected)

    def test_select_qualifying_labs_fallback(self):
        """Tests that select_qualifying_labs triggers fallback when needed."""
        p = MockPatient(patient_id="p_fallback", age=40.0, stay_hours=48, has_lab=False)
        p.events["lab"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 600.0,
                    "labname": "potassium",
                    "labresult": 4.0,
                },
            )()
        ]

        mock_dataset: Any = type("MockDS", (), {"iter_patients": lambda self: [p]})()

        selected, threshold_met = select_qualifying_labs(
            mock_dataset,
            train_pids={"p_fallback"},
            min_prevalence=0.25,
            min_avg_obs=3.0,
            max_labs=45,
        )
        self.assertFalse(threshold_met)
        self.assertEqual(selected, DEFAULT_LAB_NAMES)

    def test_compute_empirical_training_stats_non_finite_resilience(self):
        """Tests that compute_empirical_training_stats handles non-finite values."""
        p = MockPatient(patient_id="p_finite", age=45.0, stay_hours=24, has_lab=False)
        # Add lab events with valid, NaN, and Inf values
        p.events["lab"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 120.0,
                    "labname": "glucose",
                    "labresult": 100.0,
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 180.0,
                    "labname": "glucose",
                    "labresult": float("nan"),
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 240.0,
                    "labname": "glucose",
                    "labresult": float("inf"),
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 300.0,
                    "labname": "glucose",
                    "labresult": 110.0,
                },
            )(),
        ]
        mock_ds: Any = type("MockDS", (), {"iter_patients": lambda self: [p]})()

        vitals = {
            "s0": [
                (1, "heart_rate", 80.0),
                (2, "heart_rate", float("nan")),
                (3, "heart_rate", float("inf")),
                (4, "heart_rate", 90.0),
            ]
        }

        ts_stats, _, static_stats, _ = compute_empirical_training_stats(
            mock_ds,
            train_pids={"p_finite"},
            lab_names=["glucose"],
            vitals_by_stay=vitals,
        )

        for k, stat in ts_stats.items():
            self.assertEqual(len(stat), 4)
            mean, std, p1, p99 = stat
            self.assertFalse(np.isnan(mean), f"NaN mean in {k}")
            self.assertFalse(np.isinf(mean), f"Inf mean in {k}")
            self.assertFalse(np.isnan(std), f"NaN std in {k}")
            self.assertFalse(np.isinf(std), f"Inf std in {k}")
            self.assertLessEqual(p1, p99)

        for k, (mean, std, med) in static_stats.items():
            self.assertFalse(np.isnan(mean), f"NaN static mean in {k}")
            self.assertFalse(np.isnan(std), f"NaN static std in {k}")
            self.assertFalse(np.isnan(med), f"NaN static med in {k}")

    def test_empirical_stats_clipping_consistency_with_polars(self):
        """Tests that mean/std match across Polars and fallback on clipped values."""
        vals = [4.0] * 98 + [0.1, 1000.0]

        # 1. Fallback path (iter_patients)
        p = MockPatient(patient_id="p1", age=50.0, stay_hours=24, has_lab=False)
        p.events["lab"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 60.0,
                    "labname": "potassium",
                    "labresult": v,
                },
            )()
            for v in vals
        ]
        ds_fallback = type("MockDS", (), {"iter_patients": lambda self: [p]})()
        stats_fallback, counts_fallback, _, _ = compute_empirical_training_stats(
            ds_fallback, train_pids={"p1"}, lab_names=["potassium"]
        )

        # 2. Polars path (global_event_df)
        import polars as pl

        p_rows = [
            {
                "event_type": "patient",
                "patient_id": "p1",
                "patient/patientunitstayid": "s0",
                "patient/unitdischargeoffset": 1440.0,
                "patient/age": "50",
                "patient/unittype": "MICU",
                "patient/unitadmitsource": "Floor",
                "patient/unitstaytype": "admit",
                "patient/ethnicity": "Caucasian",
                "patient/admissionheight": 170.0,
                "patient/admissionweight": 70.0,
                "lab/patientunitstayid": None,
                "lab/labresultoffset": None,
                "lab/labname": None,
                "lab/labresult": None,
            }
        ]
        l_rows = [
            {
                "event_type": "lab",
                "patient_id": "p1",
                "patient/patientunitstayid": "s0",
                "patient/unitdischargeoffset": 1440.0,
                "patient/age": "50",
                "patient/unittype": "MICU",
                "patient/unitadmitsource": "Floor",
                "patient/unitstaytype": "admit",
                "patient/ethnicity": "Caucasian",
                "patient/admissionheight": 170.0,
                "patient/admissionweight": 70.0,
                "lab/patientunitstayid": "s0",
                "lab/labresultoffset": 60.0,
                "lab/labname": "potassium",
                "lab/labresult": v,
            }
            for v in vals
        ]
        ds_polars = type(
            "MockDS", (), {"global_event_df": pl.LazyFrame(p_rows + l_rows)}
        )()
        stats_polars, counts_polars, _, _ = compute_empirical_training_stats(
            ds_polars, train_pids={"p1"}, lab_names=["potassium"]
        )

        # Unclipped std would be ~99.6; clipped std should be ~0.996
        raw_std = float(np.std(vals, ddof=1))
        self.assertGreater(raw_std, 90.0)

        # Both paths compute mean and std on the clipped column
        fb_mean, fb_std, fb_p1, fb_p99 = stats_fallback["potassium"]
        pl_mean, pl_std, pl_p1, pl_p99 = stats_polars["potassium"]

        self.assertLess(fb_std, 2.0)
        self.assertLess(pl_std, 2.0)

        self.assertAlmostEqual(fb_mean, pl_mean, places=5)
        self.assertAlmostEqual(fb_std, pl_std, places=5)
        self.assertAlmostEqual(fb_p1, pl_p1, places=5)
        self.assertAlmostEqual(fb_p99, pl_p99, places=5)
        self.assertEqual(counts_fallback["potassium"], 100)
        self.assertEqual(counts_polars["potassium"], 100)

    def test_compute_empirical_training_stats_warns_on_zero_vitals(self):
        """Tests that a warning is logged when no bedside vitals are found."""
        p = MockPatient(patient_id="p1", age=50.0, stay_hours=24, has_lab=True)
        ds = type("MockDS", (), {"iter_patients": lambda self: [p]})()
        with self.assertLogs(
            "pyhealth.tasks.length_of_stay_tpc_eicu", level="WARNING"
        ) as cm:
            stats, counts, _, _ = compute_empirical_training_stats(
                ds,
                train_pids={"p1"},
                lab_names=["potassium"],
                vitals_by_stay=None,
            )
        self.assertTrue(any("No bedside vital signs found" in log for log in cm.output))
        # Vitals fall back to default priors
        for vkey in CORE_VITALS_MAP:
            self.assertEqual(counts[vkey], 0)
            self.assertIn(vkey, stats)
            self.assertEqual(stats[vkey], DEFAULT_VITALS_NORMALIZATION[vkey])

    def test_fast_reservoir_sampler_unbiased(self):
        """Tests that FastReservoirSampler uniformly samples without order bias."""
        sampler = FastReservoirSampler(capacity=10_000, seed=42)
        n_items = 100_000
        for i in range(n_items):
            sampler.add(float(i))

        self.assertEqual(sampler.total_count, n_items)
        self.assertEqual(len(sampler.samples), 10_000)

        true_mean = (n_items - 1) / 2.0  # 49,999.5
        sample_mean = float(np.mean(sampler.samples))
        # Unbiased reservoir sample mean is within 1% of true mean (biased head: 4999.5)
        self.assertAlmostEqual(sample_mean, true_mean, delta=1000.0)

        # Uniform span across the entire stream
        self.assertLess(min(sampler.samples), 5_000.0)
        self.assertGreater(max(sampler.samples), 95_000.0)

        # Reproducibility with the same seed
        sampler2 = FastReservoirSampler(capacity=10_000, seed=42)
        for i in range(1000):
            sampler2.add(float(i))
        sampler3 = FastReservoirSampler(capacity=10_000, seed=42)
        for i in range(1000):
            sampler3.add(float(i))
        self.assertEqual(sampler2.samples, sampler3.samples)

        # Filtering non-finite values (NaN / Inf)
        sampler_nan = FastReservoirSampler(capacity=100, seed=42)
        sampler_nan.add(float("nan"))
        sampler_nan.add(float("inf"))
        sampler_nan.add(float("-inf"))
        sampler_nan.add(42.0)
        self.assertEqual(sampler_nan.total_count, 1)
        self.assertEqual(sampler_nan.samples, [42.0])

    def test_compute_empirical_training_stats_reservoir_sampling(self):
        """Tests that stats use reservoir sampling to accurately track counts."""
        # Create 50 mock patients / stays with vital observations
        patients = []
        train_pids = set()
        vitals_by_stay = {}
        for i in range(50):
            pid = f"p_{i}"
            sid = f"s_{i}"
            train_pids.add(pid)
            p = MockPatient(patient_id=pid, age=50.0 + i, stay_hours=24, has_lab=False)
            p.events["patient"][0].patientunitstayid = sid
            patients.append(p)
            # 20 observations per stay -> 1,000 total observations
            vitals_by_stay[sid] = [
                (h, "heart_rate", 70.0 + (i % 20) + (h % 5)) for h in range(20)
            ]

        mock_ds = type("MockDS", (), {"iter_patients": lambda self: patients})()
        stats, counts, _s_stats, _ = compute_empirical_training_stats(
            mock_ds,
            train_pids=train_pids,
            lab_names=[],
            vitals_by_stay=vitals_by_stay,
            max_stat_samples=100,  # Bound reservoir capacity to 100
            seed=42,
        )

        # Observation count is TRUE total across all 50 stays (1000), not capped at 100
        self.assertEqual(counts["heart_rate"], 1000)
        # Stats are computed on the uniform reservoir sample
        mean, std, p1, p99 = stats["heart_rate"]
        self.assertGreater(mean, 65.0)
        self.assertLess(mean, 95.0)
        self.assertGreater(std, 0.5)
        self.assertLessEqual(p1, p99)

    def test_outlier_percentile_clipping_in_task(self):
        """Tests that 1st/99th percentile clipping bounds are enforced on raw values."""
        norm_stats: dict[str, tuple[float, ...]] = {
            "glucose": (100.0, 10.0, 80.0, 120.0),
        }
        task = LengthOfStayTPCeICU(lab_names=["glucose"], norm_stats=norm_stats)
        p = MockPatient(patient_id="p_outlier", age=45.0, stay_hours=24, has_lab=False)
        p.events["lab"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 120.0,
                    "labname": "glucose",
                    "labresult": 500.0,
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": 180.0,
                    "labname": "glucose",
                    "labresult": 10.0,
                },
            )(),
        ]
        samples = task(p)
        self.assertEqual(len(samples), 1)
        vitals = samples[0]["vitals"]
        glucose_idx = task.feature_map["glucose"]
        # 500.0 clipped to 120.0 -> normalized = (120 - 100) / 10 = 2.0
        self.assertAlmostEqual(vitals[2, glucose_idx].item(), 2.0, places=4)
        # 10.0 clipped to 80.0 -> normalized = (80 - 100) / 10 = -2.0
        self.assertAlmostEqual(vitals[3, glucose_idx].item(), -2.0, places=4)

    def test_bucket_batch_sampler(self):
        """Tests BucketBatchSampler indexing, shuffling, and reset replay."""
        lengths = [12, 14, 15, 20, 22, 25, 48, 50, 52, 100, 105, 110]
        batch_size = 3
        sampler = BucketBatchSampler(
            lengths, batch_size=batch_size, shuffle=True, seed=123
        )

        # 1. Epoch 0: All indices yielded exactly once
        batches_ep0 = list(sampler)
        all_indices_ep0 = [idx for batch in batches_ep0 for idx in batch]
        self.assertEqual(sorted(all_indices_ep0), list(range(len(lengths))))
        self.assertEqual(len(batches_ep0), 4)
        for b in batches_ep0:
            self.assertEqual(len(b), batch_size)

        # 2. Epoch 1: Yields all indices with shuffled batch order
        batches_ep1 = list(sampler)
        all_indices_ep1 = [idx for batch in batches_ep1 for idx in batch]
        self.assertEqual(sorted(all_indices_ep1), list(range(len(lengths))))
        self.assertNotEqual(batches_ep0, batches_ep1)

        # 3. Reset: Replays epoch 0 sequence identically
        sampler.reset()
        batches_ep0_replay = list(sampler)
        self.assertEqual(batches_ep0_replay, batches_ep0)

        # 4. Error handling and empty sequence checks
        with self.assertRaises(ValueError):
            BucketBatchSampler([10, 20], batch_size=0)
        empty_sampler = BucketBatchSampler([], batch_size=4)
        self.assertEqual(len(empty_sampler), 0)
        self.assertEqual(list(empty_sampler), [])

    def test_map_dataset_lazy_indexing(self):
        """Tests that MapDataset supports both eager lists and lazy index mapping."""
        mock_data = [{"id": f"s_{i}"} for i in range(10)]

        # Eager mode
        ds_eager = MapDataset(mock_data)
        self.assertEqual(len(ds_eager), 10)
        self.assertEqual(ds_eager[0]["id"], "s_0")
        self.assertEqual(ds_eager[9]["id"], "s_9")

        # Lazy subset indexing mode
        subset_indices = [2, 5, 8]
        ds_lazy = MapDataset(mock_data, indices=subset_indices)
        self.assertEqual(len(ds_lazy), 3)
        self.assertEqual(ds_lazy[0]["id"], "s_2")
        self.assertEqual(ds_lazy[1]["id"], "s_5")
        self.assertEqual(ds_lazy[2]["id"], "s_8")

    def test_example_utilities(self):
        """Tests that BucketBatchSampler, MapDataset, and metrics re-export cleanly."""
        import pyhealth.tasks as tasks_pkg

        self.assertIs(tasks_pkg.BucketBatchSampler, BucketBatchSampler)
        self.assertIs(tasks_pkg.MapDataset, MapDataset)
        self.assertIs(tasks_pkg.compute_los_metrics, compute_los_metrics)

        example_path = Path(repo_root) / "examples" / "eicu_los_tpc.py"
        if example_path.exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "eicu_los_tpc_example", example_path
            )
            if spec is not None and spec.loader is not None:
                example_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(example_mod)
                self.assertIs(example_mod.BucketBatchSampler, BucketBatchSampler)
                self.assertIs(example_mod.MapDataset, MapDataset)
                self.assertIs(example_mod.compute_los_metrics, compute_los_metrics)

    def test_compute_los_metrics_unknown_metric(self):
        """Tests that compute_los_metrics raises ValueError for unknown metrics."""
        with self.assertRaises(ValueError):
            compute_los_metrics([1.0], [1.0], metrics=["invalid_metric_xyz"])

    def test_static_feature_names_and_include_mortality(self):
        """Tests static_feature_names matches static_dim and include_mortality flag."""
        task = LengthOfStayTPCeICU(include_mortality=True)
        self.assertEqual(len(task.static_feature_names), task.static_dim)
        self.assertIn("age", task.static_feature_names)
        self.assertIn("missing_gcs", task.static_feature_names)
        self.assertIn("mortality", task.output_schema)

        task_no_mort = LengthOfStayTPCeICU(include_mortality=False)
        self.assertNotIn("mortality", task_no_mort.output_schema)

        task_custom = LengthOfStayTPCeICU(
            min_stay_hours=2.0, min_age=21.0, decay_rate=0.5, floor_days=0.01
        )
        self.assertEqual(task_custom.min_stay_hours, 2.0)
        self.assertEqual(task_custom.min_stay_offset, 120.0)
        self.assertEqual(task_custom.min_age, 21.0)
        self.assertEqual(task_custom.decay_rate, 0.5)
        self.assertEqual(task_custom.floor_days, 0.01)

    def test_extract_patient_events_edge_cases(self):
        """Tests that extract_patient_events handles None in baseline height/weight
        and invalid offsets.
        """
        base_s_map: dict[str, dict[str, float | None]] = {
            "s0": {"height": None, "weight": None, "eyes": 4.0}
        }
        task = LengthOfStayTPCeICU(baseline_static_by_stay=base_s_map)
        p = MockPatient(patient_id="p_none_edge", age=55.0, stay_hours=24, has_lab=True)
        # Add a lab with unparseable offset to test graceful handling
        p.events["lab"].append(
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labresultoffset": "unparseable_offset",
                    "labname": "potassium",
                    "labresult": 4.5,
                },
            )()
        )
        samples = task(p)
        self.assertEqual(len(samples), 1)
        self.assertIsNotNone(samples[0]["static"])

    def test_length_of_stay_metrics_fn(self):
        """Tests compute_los_metrics on 2D and 1D inputs."""
        # 2D sequences with warmup and padding
        yt = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 3.0, 4.0, 5.0, 0.0, 0.0]]
        )
        yp = torch.tensor(
            [[1.1, 2.1, 3.1, 4.1, 5.1, 6.1], [2.1, 3.1, 4.1, 5.1, 0.0, 0.0]]
        )
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
            ]
        )

        res_2d = compute_los_metrics(yt, yp, mask=mask, min_eval_hours=4)
        for key in ["msle", "mse", "mae", "mape", "r2", "binned_kappa"]:
            self.assertIn(key, res_2d)
            self.assertFalse(np.isnan(res_2d[key]), f"NaN in {key}")
        self.assertLess(res_2d["mae"], 0.2)

        # Custom bin edges
        res_custom_bins = compute_los_metrics(
            yt, yp, mask=mask, min_eval_hours=4, bin_edges=[2.0, 4.0, 8.0]
        )
        self.assertIn("binned_kappa", res_custom_bins)

        # 1D flattened input
        res_1d = compute_los_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertAlmostEqual(res_1d["mse"], 0.0)

        # Empty array
        res_empty = compute_los_metrics(np.array([]), np.array([]))
        self.assertEqual(res_empty["mse"], 0.0)

        # List of variable-length sequence batches
        yt_list = [
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        ]
        yp_list = [
            torch.tensor([[1.1, 2.1, 3.1]]),
            torch.tensor([[1.1, 2.1, 3.1, 4.1, 5.1]]),
        ]
        res_list = compute_los_metrics(yt_list, yp_list, min_eval_hours=0)
        self.assertIn("msle", res_list)
        self.assertLess(res_list["mae"], 0.2)

        # List of unbatched (T, 1) sequence tensors with min_eval_hours=4
        # Two stays of length 6; identical after warmup (t >= 4), differing
        # only at t < 4
        yt_stay1 = torch.tensor([[10.0], [20.0], [30.0], [40.0], [5.0], [6.0]])
        yp_stay1 = torch.tensor([[99.0], [99.0], [99.0], [99.0], [5.0], [6.0]])
        res_unbatched = compute_los_metrics([yt_stay1], [yp_stay1], min_eval_hours=4)
        self.assertAlmostEqual(res_unbatched["mae"], 0.0, places=5)

        # Single-bin agreement edge case where denominator is zero
        res_single_bin = compute_los_metrics([2.5, 2.5], [2.5, 2.5], min_eval_hours=0)
        self.assertEqual(res_single_bin["binned_kappa"], 1.0)

    def test_load_eicu_vitals_cache_chunked_and_filtered(self):
        """Tests that load_eicu_vitals_cache chunks and filters stays."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            p_prefix = "notes/Progress Notes/Physical Exam/Physical Exam/Vital Signs/"
            csv_content = (
                "patientunitstayid,physicalexamoffset,physicalexampath,"
                "physicalexamtext\n"
                f"stay_1,60,{p_prefix}HR/HR Current,80\n"
                f"stay_2,120,{p_prefix}HR/HR Current,90\n"
                f"stay_1,180,{p_prefix}HR/HR Current,85\n"
                f"stay_3,240,{p_prefix}HR/HR Current,95\n"
            )
            pe_file = tmp_path / "physicalExam.csv"
            pe_file.write_text(csv_content)

            # Test chunksize=2 with filter on stay_1 only
            vitals, _ = load_eicu_vitals_cache(
                str(tmp_path), chunksize=2, valid_stay_ids={"stay_1"}
            )

            self.assertIn("stay_1", vitals)
            self.assertNotIn("stay_2", vitals)
            self.assertNotIn("stay_3", vitals)
            self.assertEqual(len(vitals["stay_1"]), 2)
            # Ordered chronologically: hour 1 (60m) then hour 3 (180m)
            self.assertEqual(vitals["stay_1"][0], (1, "heart_rate", 80.0))
            self.assertEqual(vitals["stay_1"][1], (3, "heart_rate", 85.0))

    def test_load_eicu_vitals_cache_periodic_and_aperiodic_tables(self):
        """Tests that periodic and aperiodic tables are parsed and merged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            p_content = (
                "PatientUnitStayID,ObservationOffset,heartRate,"
                "systemicSystolic,temperature\n"
                "stay_1,30,72,120,36.8\n"
                "stay_2,60,80,110,37.0\n"
            )
            (tmp_path / "vitalPeriodic.csv").write_text(p_content)

            ap_content = (
                "patientunitstayid,observationoffset,noninvasivesystolic,"
                "temperature\n"
                "stay_1,90,125,98.6\n"
                "stay_1,15,118,97.7\n"
            )
            (tmp_path / "vitalAperiodic.csv").write_text(ap_content)

            vitals, _ = load_eicu_vitals_cache(
                str(tmp_path), chunksize=1, valid_stay_ids={"stay_1"}
            )
            self.assertIn("stay_1", vitals)
            self.assertNotIn("stay_2", vitals)

            records = vitals["stay_1"]
            # All stay_1 records: offset 15, offset 30, offset 90
            self.assertTrue(len(records) >= 4)
            # Chronological check: hour of offset 15 is 0, hour of offset 90 is 1
            self.assertEqual(records[0][0], 0)
            self.assertEqual(records[-1][0], 1)

            # Check that 98.6 deg F was converted to approx 37.0 deg C
            temp_records = [r for r in records if r[1] == "temperature"]
            self.assertTrue(len(temp_records) >= 2)
            converted_temps = [r[2] for r in temp_records]
            # Should have ~36.5 (from 97.7 F), 36.8 (C), and ~37.0 (from 98.6 F)
            self.assertTrue(any(abs(t - 37.0) < 0.1 for t in converted_temps))
            self.assertTrue(any(abs(t - 36.5) < 0.1 for t in converted_temps))

    def test_load_eicu_vitals_physicalexam_text_and_value_columns(self):
        """Tests that load_eicu_vitals_cache extracts numeric vitals from text

        when physicalexamvalue contains attribute descriptors (as in standard eICU),
        and falls back to physicalexamvalue when numeric value is stored there.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            p_prefix = (
                "notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/"
                "Vital Sign and Physiological Data/"
            )
            csv_content = (
                "patientunitstayid,physicalexamoffset,physicalexampath,"
                "physicalexamvalue,physicalexamtext\n"
                # Standard eICU: physicalexamtext has numeric measurement ('82')
                f"stay_1,60,{p_prefix}HR/HR Current,HR Current,82\n"
                # Fallback: physicalexamvalue has numeric measurement ('120')
                f"stay_1,120,{p_prefix}BP (systolic)/BP (systolic) Current,"
                "120,recorded\n"
                # Both numeric (e.g. GCS score)
                "stay_1,-60,notes/Progress Notes/Physical Exam/Physical Exam/"
                "Neurologic/GCS/Motor Score/6,6,6\n"
            )
            pe_file = tmp_path / "physicalExam.csv"
            pe_file.write_text(csv_content)

            vitals, baselines = load_eicu_vitals_cache(str(tmp_path))

            self.assertIn("stay_1", vitals)
            self.assertEqual(len(vitals["stay_1"]), 2)
            self.assertEqual(vitals["stay_1"][0], (1, "heart_rate", 82.0))
            self.assertEqual(vitals["stay_1"][1], (2, "systolic_bp", 120.0))

            self.assertIn("stay_1", baselines)
            self.assertEqual(baselines["stay_1"].get("motor"), 6.0)

    def test_get_valid_eicu_stay_ids(self):
        """Tests get_valid_eicu_stay_ids filtering on adult and >=5h criteria."""

        class MockPatientObj:
            def __init__(self, sid, offset, age):
                self.patientunitstayid = sid
                self.unitdischargeoffset = offset
                self.age = age

        p_adult = type(
            "P",
            (),
            {
                "get_events": lambda self, event_type: [
                    MockPatientObj("stay_valid", 360, "55"),
                    MockPatientObj("stay_short", 120, "55"),
                    MockPatientObj("stay_child", 400, "16"),
                ]
            },
        )()
        mock_ds = type("DS", (), {"iter_patients": lambda self: [p_adult]})()
        valid_ids = get_valid_eicu_stay_ids(mock_ds)
        self.assertEqual(valid_ids, {"stay_valid"})

    def test_length_of_stay_tpc_eicu_fn(self):
        """Tests length_of_stay_tpc_eicu_fn functional task wrapper."""
        from pyhealth.tasks import length_of_stay_tpc_eicu_fn

        p_empty = type("P", (), {"get_events": lambda self, event_type: []})()
        samples = length_of_stay_tpc_eicu_fn(p_empty)
        self.assertEqual(samples, [])

    def test_task_fit_method(self):
        """Tests LengthOfStayTPCeICU.fit factory method."""
        p = MockPatient(patient_id="p_fit", age=50.0, stay_hours=24, has_lab=True)
        ds: Any = type("MockDS", (), {"iter_patients": lambda self: [p]})()
        vitals = {"s0": [(1, "heart_rate", 75.0)]}
        task = LengthOfStayTask.fit(
            dataset=ds,
            train_pids={"p_fit"},
            vitals_by_stay=vitals,
        )
        self.assertIsInstance(task, LengthOfStayTask)
        self.assertIn("heart_rate", task.vital_keys)
        self.assertGreater(task.continuous_dim, 0)
        samples = task(p)
        self.assertEqual(len(samples), 1)
        self.assertIn("vitals", samples[0])

    def test_native_physicalexam_extraction_without_cache(self):
        """Tests that native physicalexam events are extracted without vitals cache."""
        p = MockPatient(patient_id="p_native", age=55.0, stay_hours=24, has_lab=False)
        p.events["physicalexam"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "physicalexamoffset": 120.0,
                    "physicalexampath": (
                        "notes/Progress Notes/Physical Exam/Physical Exam/"
                        "Vital Signs/HR/HR Current"
                    ),
                    "physicalexamtext": "82.0",
                    "physicalexamvalue": "HR Current",
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "physicalexamoffset": 180.0,
                    "physicalexampath": (
                        "notes/Progress Notes/Physical Exam/Physical Exam/"
                        "Vital Signs/Temperature"
                    ),
                    "physicalexamtext": "98.6",
                    "physicalexamvalue": "Admission",
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "physicalexamoffset": -60.0,
                    "physicalexampath": (
                        "notes/Progress Notes/Physical Exam/Physical Exam/"
                        "Neurological/GCS/Eyes Score"
                    ),
                    "physicalexamtext": "4",
                    "physicalexamvalue": "Spontaneous",
                },
            )(),
        ]

        task = LengthOfStayTask(
            lab_names=["potassium"],
            vitals_by_stay=None,
            baseline_static_by_stay=None,
        )
        self.assertEqual(len(task.vital_keys), 6)
        self.assertEqual(task.continuous_dim, 7)

        samples = task(p)
        self.assertEqual(len(samples), 1)
        sample = samples[0]

        hr_idx = task.feature_map["heart_rate"]
        temp_idx = task.feature_map["temperature"]
        pot_idx = task.feature_map["potassium"]

        self.assertEqual(sample["vitals"].shape, (24, 7))
        self.assertEqual(sample["mask"].shape, (24,))
        self.assertGreater(sample["vitals_decay"][:, hr_idx].sum().item(), 0.0)
        self.assertGreater(sample["vitals_decay"][:, temp_idx].sum().item(), 0.0)
        self.assertEqual(sample["vitals_decay"][:, pot_idx].sum().item(), 0.0)

        # 98.6 F converts to 37.0 C (default prior mean 37.0 -> normalized 0.0)
        self.assertAlmostEqual(sample["vitals"][3, temp_idx].item(), 0.0, places=2)

        # Baseline GCS Eyes was extracted (score 4)
        self.assertAlmostEqual(sample["static"][7].item(), (4.0 - 3.5) / 0.8, places=2)

        # LengthOfStayTask.fit directly from native physicalexam without cache
        ds: Any = type("MockDS", (), {"iter_patients": lambda self: [p]})()
        fitted_task = LengthOfStayTask.fit(
            dataset=ds,
            train_pids={"p_native"},
            vitals_by_stay=None,
            baseline_static_by_stay=None,
        )
        self.assertEqual(len(fitted_task.vital_keys), 6)
        self.assertIn("heart_rate", fitted_task.norm_stats)
        self.assertIn("temperature", fitted_task.norm_stats)
        self.assertEqual(fitted_task.feat_counts.get("heart_rate"), 1)
        self.assertAlmostEqual(fitted_task.norm_stats["heart_rate"][0], 82.0, places=2)
        self.assertEqual(fitted_task.feat_counts.get("systolic_bp"), 0)
        self.assertEqual(fitted_task.norm_stats["systolic_bp"][0], 120.0)

    def test_task_cache_fingerprint_decoupled(self):
        """Tests that task keeps large dicts out of vars(task) for json.dumps."""
        import json

        large_vitals = {f"stay_{i}": [(0, "heart_rate", 80.0)] for i in range(50)}
        task = LengthOfStayTask(vitals_by_stay=large_vitals)

        # 1. vars(task) does not contain vitals_by_stay directly
        v = vars(task)
        self.assertNotIn("vitals_by_stay", v)
        self.assertNotIn("baseline_static_by_stay", v)
        self.assertIn("vitals_hash", v)

        # 2. json.dumps is small and compact (< 4 KB)
        dumped = json.dumps(v, default=str)
        self.assertLess(len(dumped), 4096)

        # 3. Transparent property access returns the full dictionary
        self.assertEqual(len(task.vitals_by_stay), 50)
        self.assertIn("stay_0", task.vitals_by_stay)

    def test_base_length_of_stay_tpc_contract(self):
        """Tests BaseLengthOfStayTPC abstract contract, dimensions, and processing."""
        from pyhealth.tasks import BaseLengthOfStayTPC, RawStayRecord

        base_task = BaseLengthOfStayTPC(time_series_features=["hr", "bp"])
        self.assertEqual(base_task.continuous_dim, 2)
        self.assertEqual(base_task.time_dim, 2)
        self.assertGreater(base_task.static_dim, 0)
        self.assertIn("los", base_task.output_schema)
        self.assertIn("mortality", base_task.output_schema)

        # Calling abstract extract_patient_events raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            base_task.extract_patient_events(None)

        # Test process_stay_record on standardized RawStayRecord
        record = RawStayRecord(
            stay_id="stay_test",
            patient_id="pat_test",
            stay_hours=10,
            discharge_offset_minutes=600.0,
            admission_hour=10.0,
            hourly_vitals=[(1, "hr", 85.0), (2, "bp", 125.0)],
            static_continuous={"age": 65.0, "gender": 1.0},
            static_categorical={"unittype": "MICU"},
            is_expired=False,
        )
        sample = base_task.process_stay_record(record)
        self.assertIsNotNone(sample)
        assert sample is not None
        self.assertEqual(sample["visit_id"], "stay_test")
        self.assertEqual(sample["patient_id"], "pat_test")
        self.assertEqual(sample["vitals"].shape, (10, 2))
        self.assertEqual(sample["vitals_decay"].shape, (10, 2))
        self.assertEqual(sample["los"].shape, (10, 1))
        self.assertEqual(sample["mortality"].item(), 0.0)

        # Verify input_schema instance isolation and include_time_features=False
        base_task.input_schema["custom_key"] = "test"
        self.assertNotIn("custom_key", BaseLengthOfStayTPC.input_schema)

        no_time_task = BaseLengthOfStayTPC(
            time_series_features=["hr"], include_time_features=False
        )
        self.assertNotIn("time_features", no_time_task.input_schema)
        self.assertEqual(no_time_task.time_dim, 0)
        sample_notime = no_time_task.process_stay_record(record)
        self.assertIsNotNone(sample_notime)
        assert sample_notime is not None
        self.assertNotIn("time_features", sample_notime)

        # Verify LengthOfStayTPCEICU alias
        from pyhealth.tasks import LengthOfStayTPCEICU, LengthOfStayTPCeICU

        self.assertIs(LengthOfStayTPCEICU, LengthOfStayTPCeICU)

    def test_tpc_trainer_training_compatibility(self):
        """Tests that PyHealth Trainer can train TPC model without errors."""
        from pyhealth.trainer import Trainer

        class ToyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.samples = [
                    {
                        "vitals": torch.randn(12, 6),
                        "vitals_decay": torch.rand(12, 6),
                        "los": torch.rand(12, 1) + 0.1,
                        "mask": torch.ones(12, dtype=torch.bool),
                        "time_features": torch.rand(12, 2),
                        "static": torch.randn(4),
                    }
                    for _ in range(4)
                ]

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return self.samples[index]

        ds = ToyDataset()
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, collate_fn=collate_fn_dict_with_padding
        )

        model = TPC(
            dataset=None,
            continuous_dim=6,
            static_dim=4,
            time_dim=2,
            n_layers=2,
        )
        trainer = Trainer(model=model, device="cpu")
        trainer.train(train_dataloader=loader, epochs=1)

    def test_tpc_pyhealth_stack_variable_length_trainer(self):
        """Tests TPC compatibility with PyHealth's own stack: create_sample_dataset,

        get_dataloader, and Trainer with variable sequence lengths and dynamic padding.
        """
        from pyhealth.datasets import create_sample_dataset, get_dataloader
        from pyhealth.trainer import Trainer

        # 4 samples with variable sequence lengths: 6, 8, 10, 12 hours
        lengths = [6, 8, 10, 12]
        samples = [
            {
                "patient_id": f"p_{i}",
                "visit_id": f"v_{i}",
                "vitals": torch.randn(L, 8),
                "vitals_decay": torch.rand(L, 8),
                "static": torch.randn(5),
                "time_features": torch.rand(L, 2),
                "mask": torch.ones(L, dtype=torch.bool),
                "los": torch.rand(L, 1) + 0.5,
            }
            for i, L in enumerate(lengths)
        ]

        ds = create_sample_dataset(
            samples,
            input_schema={"vitals": "tensor"},
            output_schema={"los": "tensor"},
        )

        # Verify TPC dimension auto-inference from SampleDataset
        model = TPC(
            dataset=ds,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
            hidden_dim=8,
            min_time=4,
            seq_to_seq=False,
        )
        self.assertEqual(model.continuous_dim, 8)
        self.assertEqual(model.static_dim, 5)
        self.assertEqual(model.time_dim, 2)

        # DataLoaders with batch_size=2 ensuring variable lengths across batches
        train_loader = get_dataloader(ds, batch_size=2, shuffle=True, seed=42)
        val_loader = get_dataloader(ds, batch_size=2, shuffle=False)

        trainer = Trainer(
            model=model,
            metrics=["mse", "mae"],
            enable_logging=False,
            device="cpu",
        )

        # Train through PyHealth Trainer
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=2,
        )

        # Evaluate through PyHealth Trainer across variable-length padded batches
        scores = trainer.evaluate(val_loader)
        self.assertIn("mse", scores)
        self.assertIn("mae", scores)
        self.assertIn("loss", scores)
        self.assertFalse(np.isnan(scores["mse"]))
        self.assertFalse(np.isnan(scores["mae"]))
        self.assertGreater(scores["mse"], 0.0)

        # Inference with return_patient_ids: visit-level predictions for 4 patients
        y_true, y_prob, _loss_mean, pids = trainer.inference(
            val_loader, return_patient_ids=True
        )
        self.assertEqual(len(y_true), len(lengths))
        self.assertEqual(len(y_prob), len(lengths))
        self.assertEqual(len(pids), len(lengths))
        self.assertFalse(np.isnan(y_prob).any())

    def test_tpc_generic_single_tensor_auto_fallback(self):
        """Tests TPC compatibility with generic single-tensor PyHealth inputs.

        Verifies that:
        1. When decay keys are missing from kwargs, TPC auto-falls back to
           synthesizing decay ones without raising KeyError.
        2. When feature_keys=["x"] is provided, self.time_series_key automatically
           maps to "x".
        3. When initialized with a SampleDataset that lacks decay indicators,
           use_decay auto-falls back to False.
        """
        # 1. Forward without decay tensor when use_decay=True
        model = TPC(continuous_dim=10, n_layers=2, temp_dim=4, point_dim=4)
        x = torch.randn(2, 8, 10)
        y = torch.rand(2, 8, 1) + 0.1
        out = model(vitals=x, y_true=y)
        self.assertEqual(out["y_prob"].shape, torch.Size([2, 8, 1]))
        self.assertIn("loss", out)

        # 2. Custom feature_keys mapping
        model_custom = TPC(
            continuous_dim=6,
            feature_keys=["custom_feature"],
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        self.assertEqual(model_custom.time_series_key, "custom_feature")
        self.assertEqual(model_custom.feature_keys, ["custom_feature"])
        custom_x = torch.randn(2, 6, 6)
        out_custom = model_custom(custom_feature=custom_x, y_true=torch.rand(2, 6, 1))
        self.assertEqual(out_custom["y_prob"].shape, torch.Size([2, 6, 1]))

        # 3. SampleDataset without decay feature -> auto-fallback use_decay=False
        from pyhealth.datasets import create_sample_dataset

        samples = [
            {"patient_id": "p1", "vitals": torch.randn(6, 5), "los": torch.rand(6, 1)},
            {"patient_id": "p2", "vitals": torch.randn(8, 5), "los": torch.rand(8, 1)},
        ]
        ds = create_sample_dataset(
            samples,
            input_schema={"vitals": "tensor"},
            output_schema={"los": "tensor"},
        )
        model_ds = TPC(dataset=ds, n_layers=1, temp_dim=4, point_dim=4)
        self.assertFalse(model_ds.use_decay)
        self.assertIsNone(model_ds.decay_key)
        self.assertEqual(model_ds.feature_keys, ["vitals"])

        # 4. SampleDataset with static and no decay -> feature_keys includes static
        samples_static = [
            {
                "patient_id": "p1",
                "vitals": torch.randn(6, 5),
                "static": torch.randn(4),
                "los": torch.rand(6, 1),
            },
        ]
        ds_static = create_sample_dataset(
            samples_static,
            input_schema={"vitals": "tensor", "static": "tensor"},
            output_schema={"los": "tensor"},
        )
        model_static = TPC(dataset=ds_static, n_layers=1, temp_dim=4, point_dim=4)
        self.assertFalse(model_static.use_decay)
        self.assertEqual(model_static.feature_keys, ["vitals", "static"])

    def test_tpc_seq_to_seq_false_stay_level(self):
        """Tests seq_to_seq=False for stay/visit-level PyHealth prediction.

        Verifies that predictions and targets are pooled to shape (B, 1)
        using the last valid time-step, and gradients backpropagate smoothly.
        """
        model = TPC(
            continuous_dim=6,
            seq_to_seq=False,
            n_layers=2,
            temp_dim=4,
            point_dim=4,
            hidden_dim=8,
        )
        vitals = torch.randn(3, 10, 6)
        y_stay = torch.tensor([[1.5], [4.2], [0.8]])
        mask = torch.tensor(
            [
                [True] * 5 + [False] * 5,
                [True] * 8 + [False] * 2,
                [True] * 10,
            ]
        )

        out = model(vitals=vitals, mask=mask, y_true=y_stay)
        self.assertEqual(out["y_prob"].shape, torch.Size([3, 1]))
        self.assertEqual(out["logit"].shape, torch.Size([3, 1]))
        self.assertEqual(out["y_true"].shape, torch.Size([3, 1]))
        self.assertIn("loss", out)
        self.assertEqual(out["loss"].dim(), 0)

        out["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad and name != "_dummy_param":
                self.assertIsNotNone(p.grad)

    def test_tpc_binary_mode(self):
        """Tests first-class mode='binary' support for single-task mortality.

        Matches Table 4 Row 1 (Mortality only TPC) in Rocheteau et al.
        Verifies sigmoid outputs, BCE loss computation, and backpropagation.
        """
        # Sequence-level binary prediction
        model_seq = TPC(
            continuous_dim=6,
            mode="binary",
            seq_to_seq=True,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        vitals = torch.randn(2, 6, 6)
        y_binary_seq = torch.randint(0, 2, (2, 6, 1)).float()
        out_seq = model_seq(vitals=vitals, y_true=y_binary_seq)
        self.assertEqual(out_seq["y_prob"].shape, torch.Size([2, 6, 1]))
        self.assertTrue((out_seq["y_prob"] >= 0.0).all())
        self.assertTrue((out_seq["y_prob"] <= 1.0).all())
        self.assertIn("loss", out_seq)
        out_seq["loss"].backward()

        # Stay-level binary prediction
        model_stay = TPC(
            continuous_dim=6,
            mode="binary",
            seq_to_seq=False,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        y_binary_stay = torch.tensor([[0.0], [1.0]])
        out_stay = model_stay(vitals=vitals, y_true=y_binary_stay)
        self.assertEqual(out_stay["y_prob"].shape, torch.Size([2, 1]))
        self.assertTrue((out_stay["y_prob"] >= 0.0).all())
        self.assertTrue((out_stay["y_prob"] <= 1.0).all())
        out_stay["loss"].backward()

    def test_tpc_ablation_flags(self):
        """Tests architectural ablation flags from Rocheteau et al. Table 3.

        Verifies:
        1. share_weights=True sets groups=1 in temporal convolutions (Temp. only WS).
        2. use_skip=False bypasses residual skips and sets C_n = temp_dim.
        """
        # 1. Temporal weight sharing (share_weights=True)
        model_ws = TPC(
            continuous_dim=8,
            share_weights=True,
            n_layers=2,
            temp_dim=4,
            point_dim=4,
        )
        block0 = cast(TPCBlock, model_ws.blocks[0])
        block1 = cast(TPCBlock, model_ws.blocks[1])
        conv0 = block0.temporal_conv
        conv1 = block1.temporal_conv
        self.assertEqual(conv0.groups, 1)
        self.assertEqual(conv0.in_channels, block0.C_n)
        self.assertEqual(conv0.out_channels, block0.temp_dim)
        self.assertEqual(conv1.groups, 1)
        self.assertEqual(conv1.in_channels, block1.C_n)
        self.assertEqual(conv1.out_channels, block1.temp_dim)
        vitals = torch.randn(2, 6, 8)
        out_ws = model_ws(vitals=vitals, y_true=torch.rand(2, 6, 1))
        self.assertEqual(out_ws["y_prob"].shape, torch.Size([2, 6, 1]))
        out_ws["loss"].backward()

        # 2. TPC without skip connections (use_skip=False)
        model_noskip = TPC(
            continuous_dim=8,
            use_skip=False,
            n_layers=2,
            temp_dim=4,
            point_dim=4,
        )
        # For n > 1 with no skip, C_n = temp_dim instead of temp_dim + 1
        block_noskip1 = cast(TPCBlock, model_noskip.blocks[1])
        self.assertEqual(block_noskip1.C_n, 4)
        out_noskip = model_noskip(vitals=vitals, y_true=torch.rand(2, 6, 1))
        self.assertEqual(out_noskip["y_prob"].shape, torch.Size([2, 6, 1]))
        out_noskip["loss"].backward()
        block_noskip0 = cast(TPCBlock, model_noskip.blocks[0])
        self.assertIsNotNone(block_noskip0.temporal_conv.weight.grad)
        self.assertTrue(torch.any(block_noskip0.temporal_conv.weight.grad != 0))


class TestTPCEdgeCasesAndStress(unittest.TestCase):
    """Exhaustive edge case, exception handling, and stress tests for TPC."""

    def test_batchnorm_edge_cases_and_momentum_none(self):
        """Tests MyBatchNorm1d dimensional checks, momentum=None, and invalid modes."""
        bn = MyBatchNorm1d(4)
        # 1. Dimension checks: 1D or 4D tensors must raise ValueError
        with self.assertRaises(ValueError):
            bn(torch.randn(4))
        with self.assertRaises(ValueError):
            bn(torch.randn(2, 4, 6, 8))

        # 2. momentum=None running average factor
        bn_nomom = MyBatchNorm1d(4, momentum=None, track_running_stats=True)
        bn_nomom.train()
        x1 = torch.randn(2, 4, 5)
        _ = bn_nomom(x1)
        self.assertIsNotNone(bn_nomom.num_batches_tracked)
        assert bn_nomom.num_batches_tracked is not None
        self.assertEqual(bn_nomom.num_batches_tracked.item(), 1)
        x2 = torch.randn(2, 4, 5)
        _ = bn_nomom(x2)
        assert bn_nomom.num_batches_tracked is not None
        self.assertEqual(bn_nomom.num_batches_tracked.item(), 2)

        # In eval mode with momentum=None, exponential_average_factor=0.0
        bn_nomom.eval()
        _ = bn_nomom(torch.randn(2, 4, 5))

        # 3. Invalid batchnorm mode in factory
        with self.assertRaises(ValueError):
            _create_batchnorm_layers("unsupported_mode", 4, 4)

    def test_align_target_and_align_mask_variations(self):
        """Tests shape standardization helpers across non-standard dimensionalities."""
        device = torch.device("cpu")
        B, T = 3, 5

        # 1. Target alignment variations
        # 1D target (B,)
        y1 = _align_target(torch.tensor([1.0, 2.0, 3.0]), B, T, device)
        self.assertEqual(y1.shape, torch.Size([B, T, 1]))

        # 2D target (B, 1) -> expanded
        y2 = _align_target(torch.ones(B, 1), B, T, device)
        self.assertEqual(y2.shape, torch.Size([B, T, 1]))

        # 2D target (B, T)
        y3 = _align_target(torch.ones(B, T), B, T, device)
        self.assertEqual(y3.shape, torch.Size([B, T, 1]))

        # 2D target with unexpected T2 (B, 2)
        y4 = _align_target(torch.ones(B, 2), B, T, device)
        self.assertEqual(y4.shape, torch.Size([B, 2, 1]))

        # 3D target (B, 1, T) -> transposed
        y5 = _align_target(torch.ones(B, 1, T), B, T, device)
        self.assertEqual(y5.shape, torch.Size([B, T, 1]))

        # 3D target (B, 1, 1) -> expanded
        y6 = _align_target(torch.ones(B, 1, 1), B, T, device)
        self.assertEqual(y6.shape, torch.Size([B, T, 1]))

        # 4D target (B, T, 1, 1) -> viewed
        y7 = _align_target(torch.ones(B, T, 1, 1), B, T, device)
        self.assertEqual(y7.shape, torch.Size([B, T, 1]))

        # 2. Mask alignment variations
        # None -> all True
        m_none = _align_mask(None, B, T, device)
        self.assertEqual(m_none.shape, torch.Size([B, T, 1]))
        self.assertTrue(m_none.all())

        # 1D mask (B,)
        m1 = _align_mask(torch.tensor([True, False, True]), B, T, device)
        self.assertEqual(m1.shape, torch.Size([B, T, 1]))

        # 2D mask (B, 1) -> expanded
        m2 = _align_mask(torch.ones(B, 1, dtype=torch.bool), B, T, device)
        self.assertEqual(m2.shape, torch.Size([B, T, 1]))

        # 3D mask (B, 1, T) -> transposed
        m3 = _align_mask(torch.ones(B, 1, T, dtype=torch.bool), B, T, device)
        self.assertEqual(m3.shape, torch.Size([B, T, 1]))

        # 3D mask (B, T, 3) -> sliced to first channel
        m4 = _align_mask(torch.ones(B, T, 3, dtype=torch.bool), B, T, device)
        self.assertEqual(m4.shape, torch.Size([B, T, 1]))

        # 4D mask -> reshaped
        m5 = _align_mask(torch.ones(B, T, 1, 1, dtype=torch.bool), B, T, device)
        self.assertEqual(m5.shape, torch.Size([B, T, 1]))

    def test_tpc_initialization_dimension_inference_errors(self):
        """Tests initialization dimension errors, mode validation, and custom keys."""
        # continuous_dim must be provided if dataset is None
        with self.assertRaises(ValueError):
            TPC(continuous_dim=None, dataset=None)

        # Dataset missing time series key raises ValueError
        class EmptySampleDataset:
            def __init__(self) -> None:
                self.input_schema: dict[str, Any] = {}
                self.output_schema: dict[str, Any] = {}

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return {}

        with self.assertRaises(ValueError):
            TPC(dataset=EmptySampleDataset(), continuous_dim=None)

        # Unsupported prediction mode
        with self.assertRaises(ValueError):
            TPC(continuous_dim=6, mode="multiclass")

        # Custom primary time series feature key override
        model_custom = TPC(
            continuous_dim=4,
            feature_keys=["sensor_stream"],
        )
        self.assertEqual(model_custom.time_series_key, "sensor_stream")

    def test_tpc_loss_fn_mse_and_stay_level(self):
        """Tests loss_fn='mse' for sequence and stay-level, and warmup edge cases."""
        # 1. Sequence-level MSE loss
        model_seq_mse = TPC(
            continuous_dim=4,
            loss_fn="mse",
            seq_to_seq=True,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        vitals = torch.randn(2, 6, 4)
        y_true = torch.ones(2, 6, 1) * 2.0
        out_mse = model_seq_mse(vitals=vitals, y_true=y_true)
        self.assertIn("loss", out_mse)
        self.assertFalse(torch.isnan(out_mse["loss"]))
        out_mse["loss"].backward()

        # 2. Stay-level MSE loss
        model_stay_mse = TPC(
            continuous_dim=4,
            loss_fn="mse",
            seq_to_seq=False,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        out_stay_mse = model_stay_mse(
            vitals=vitals, y_true=torch.tensor([[2.0], [3.0]])
        )
        self.assertIn("loss", out_stay_mse)
        out_stay_mse["loss"].backward()

        # 3. Stay-level MSLE loss
        model_stay_msle = TPC(
            continuous_dim=4,
            loss_fn="msle",
            seq_to_seq=False,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        out_stay_msle = model_stay_msle(
            vitals=vitals, y_true=torch.tensor([[2.0], [3.0]])
        )
        self.assertIn("loss", out_stay_msle)
        out_stay_msle["loss"].backward()

        # 4. Short sequence: T <= min_time (e.g. T=3, min_time=4)
        vitals_short = torch.randn(2, 3, 4)
        y_short = torch.ones(2, 3, 1)
        out_short = model_seq_mse(vitals=vitals_short, y_true=y_short)
        self.assertIn("loss", out_short)

        # 5. Warmup fallback: valid hours exist only before min_time
        mask_early = torch.zeros(2, 6, 1, dtype=torch.bool)
        mask_early[:, :2, :] = True  # only hours 0 and 1 are valid (< min_time=4)
        out_early = model_seq_mse(vitals=vitals, y_true=y_true, mask=mask_early)
        self.assertIn("loss", out_early)
        self.assertFalse(torch.isnan(out_early["loss"]))

        # 6. Entire mask is False (zero valid hours)
        mask_all_zero = torch.zeros(2, 6, 1, dtype=torch.bool)
        out_zero = model_seq_mse(vitals=vitals, y_true=y_true, mask=mask_all_zero)
        self.assertIn("loss", out_zero)
        self.assertFalse(torch.isnan(out_zero["loss"]))

    def test_tpc_feature_validation_and_broadcasting(self):
        """Tests input validation exceptions and tensor dimension broadcasting."""
        # Static feature validation
        model_static = TPC(continuous_dim=4, static_dim=3)
        vitals = torch.randn(2, 5, 4)
        # Missing static feature key
        with self.assertRaises(KeyError):
            model_static(vitals=vitals)
        # Static feature not a tensor
        with self.assertRaises(TypeError):
            model_static(vitals=vitals, static="invalid")
        # Static feature wrong dimension (4 != 3)
        with self.assertRaises(ValueError):
            model_static(vitals=vitals, static=torch.randn(2, 4))
        # Static feature (1, 3) broadcast to B=2
        out_stat = model_static(vitals=vitals, static=torch.randn(1, 3))
        self.assertEqual(out_stat["y_prob"].shape, torch.Size([2, 5, 1]))

        # Temporal context feature validation (key defaults to 'time_features')
        model_time = TPC(continuous_dim=4, time_dim=2)
        # Missing time feature key
        with self.assertRaises(KeyError):
            model_time(vitals=vitals)
        # Time feature not a tensor
        with self.assertRaises(TypeError):
            model_time(vitals=vitals, time_features="invalid")
        # Time feature wrong dimension (3 != 2)
        with self.assertRaises(ValueError):
            model_time(vitals=vitals, time_features=torch.randn(2, 5, 3))
        # Time feature 2D (5, 2) broadcast to B=2
        out_time = model_time(vitals=vitals, time_features=torch.randn(5, 2))
        self.assertEqual(out_time["y_prob"].shape, torch.Size([2, 5, 1]))

    def test_tpc_auxiliary_task_stay_level_and_fallback(self):
        """Tests stay-level auxiliary classification with and without padding masks."""
        # Stay-level multi-task model
        model = TPC(
            continuous_dim=4,
            seq_to_seq=False,
            aux_label_key="mortality",
            aux_weight=0.5,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        vitals = torch.randn(2, 6, 4)
        y_stay = torch.tensor([[3.0], [5.0]])
        aux_stay = torch.tensor([[0.0], [1.0]])

        # 1. With mask (evaluates at last valid index)
        mask = torch.tensor(
            [
                [[True], [True], [True], [False], [False], [False]],
                [[True], [True], [True], [True], [True], [False]],
            ]
        )
        out_mask = model(vitals=vitals, mask=mask, los=y_stay, mortality=aux_stay)
        self.assertIn("aux_loss", out_mask)
        self.assertIn("aux_prob", out_mask)
        self.assertEqual(out_mask["aux_prob"].shape, torch.Size([2, 1]))
        out_mask["loss"].backward()

        # 2. Without mask (evaluates at last index T-1)
        out_nomask = model(vitals=vitals, los=y_stay, mortality=aux_stay)
        self.assertIn("aux_loss", out_nomask)
        self.assertEqual(out_nomask["aux_prob"].shape, torch.Size([2, 1]))

        # 3. Sequence-level multi-task with all-zero mask fallback
        model_seq = TPC(
            continuous_dim=4,
            seq_to_seq=True,
            aux_label_key="mortality",
            n_layers=1,
            temp_dim=4,
            point_dim=4,
        )
        zero_mask = torch.zeros(2, 6, 1, dtype=torch.bool)
        out_seq_zero = model_seq(
            vitals=vitals,
            mask=zero_mask,
            los=torch.ones(2, 6, 1),
            mortality=torch.zeros(2, 6, 1),
        )
        self.assertIn("aux_loss", out_seq_zero)
        self.assertFalse(torch.isnan(out_seq_zero["aux_loss"]))

    def test_stay_data_map_protocol(self):
        """Tests collections.abc.Mapping protocol and dictionary methods of
        StayDataMap.
        """
        # Empty init
        sdm_empty = StayDataMap()
        self.assertEqual(len(sdm_empty), 0)
        self.assertEqual(list(sdm_empty), [])

        # Non-empty init
        data = {"stay_1": [1, 2], "stay_2": [3, 4]}
        sdm = StayDataMap(data)
        self.assertEqual(len(sdm), 2)
        self.assertEqual(list(sdm), ["stay_1", "stay_2"])
        self.assertEqual(list(sdm.keys()), ["stay_1", "stay_2"])
        self.assertEqual(list(sdm.values()), [[1, 2], [3, 4]])
        self.assertEqual(list(sdm.items()), [("stay_1", [1, 2]), ("stay_2", [3, 4])])
        self.assertEqual(repr(sdm), "<StayDataMap: 2 stays>")
        self.assertIn("stay_1", sdm)
        self.assertNotIn("nonexistent", sdm)
        self.assertNotIn(999, sdm)  # non-string key
        self.assertEqual(sdm.get("stay_1"), [1, 2])
        self.assertIsNone(sdm.get("nonexistent"))

    def test_compute_cache_fingerprint_variants(self):
        """Tests cache fingerprint calculation on None, files, paths, and dicts."""
        # None
        self.assertEqual(compute_cache_fingerprint(None), "none")
        # Empty
        self.assertEqual(compute_cache_fingerprint([]), "empty")
        self.assertEqual(compute_cache_fingerprint({}), "empty")

        # Existing file path
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
            tmp.write(b"payload")
            tmp.flush()
            fp = compute_cache_fingerprint(tmp.name)
            self.assertEqual(len(fp), 16)

        # Nonexistent file path string
        fp_nonexistent = compute_cache_fingerprint("/path/to/missing_cache.pkl")
        self.assertEqual(len(fp_nonexistent), 16)

        # Dictionary with > 3 keys triggers boundary sampling
        d_large = {f"stay_{i:02d}": [i] for i in range(10)}
        fp_dict = compute_cache_fingerprint(d_large)
        self.assertEqual(len(fp_dict), 16)

    def test_compute_los_metrics_tensor_lists_and_shapes(self):
        """Tests compute_los_metrics with variable length lists, 3D masks,
        and short stays.
        """
        # 1. Variable length 1D tensors
        y_true_list = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.5, 2.5])]
        y_pred_list = [torch.tensor([1.1, 2.1, 2.9]), torch.tensor([1.4, 2.6])]
        m1 = compute_los_metrics(y_true_list, y_pred_list, min_eval_hours=1)
        self.assertIn("mse", m1)
        self.assertIn("stay_msle", m1)

        # 2. Variable length 2D tensors with explicit mask list
        yt_2d_list = [
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[1.5], [2.5]]),
        ]
        yp_2d_list = [
            torch.tensor([[1.1], [2.1], [2.9]]),
            torch.tensor([[1.4], [2.6]]),
        ]
        mask_list = [
            torch.tensor([[True], [True], [True]]),
            torch.tensor([[True], [False]]),
        ]
        m2 = compute_los_metrics(
            yt_2d_list, yp_2d_list, mask=mask_list, min_eval_hours=1
        )
        self.assertIn("mae", m2)

        # 3. 3D mask array and mask=None with 2D array
        yt_np = np.array([[1.0, 2.0], [3.0, 4.0]])
        yp_np = np.array([[1.1, 2.1], [2.9, 3.9]])
        m_3d = np.ones((2, 2, 1), dtype=bool)
        m3 = compute_los_metrics(yt_np, yp_np, mask=m_3d, min_eval_hours=0)
        self.assertIn("msle", m3)

        # mask=None fallback to yt > 0
        m4 = compute_los_metrics(yt_np, yp_np, mask=None, min_eval_hours=0)
        self.assertIn("msle", m4)

    def test_polars_global_event_df_lab_selection_and_stay_filtering(self):
        """Tests accelerated Polars LazyFrame paths and graceful fallback to
        iter_patients.
        """
        # Build a single strongly-typed DataFrame matching eICU schema
        events_data: dict[str, list[Any]] = {
            "event_type": ["patient", "patient", "patient"] + ["lab"] * 12,
            "patient_id": ["p0", "p1", "p2"] + ["p0"] * 6 + ["p1"] * 6,
            "patient/patientunitstayid": ["s0", "s1", "s2"] + [None] * 12,
            "patient/unitdischargeoffset": [1440.0, 2880.0, 100.0] + [None] * 12,
            "patient/age": ["55", "> 89", "17"] + [None] * 12,
            "lab/patientunitstayid": [None, None, None] + ["s0"] * 6 + ["s1"] * 6,
            "lab/labresultoffset": [None, None, None]
            + [60.0, 120.0, 180.0, 240.0, 300.0, 360.0] * 2,
            "lab/labname": [None, None, None]
            + ["glucose", "potassium", "sodium", "chloride", "creatinine", "BUN"] * 2,
            "lab/labresult": [None, None, None]
            + [100.0, 4.0, 140.0, 100.0, 1.0, 15.0] * 2,
        }
        global_df = pl.DataFrame(events_data).lazy()

        class MockDatasetWithPolars:
            def __init__(self, lf: pl.LazyFrame) -> None:
                self.global_event_df = lf

            def iter_patients(self) -> list[Any]:
                return []

        ds = MockDatasetWithPolars(global_df)

        # 1. Test get_valid_eicu_stay_ids via Polars
        valid_sids = get_valid_eicu_stay_ids(ds, min_stay_hours=5.0, min_age=18.0)
        self.assertIn("s0", valid_sids)
        self.assertIn("s1", valid_sids)
        self.assertNotIn("s2", valid_sids)  # offset 100 < 300 and age 17 <= 18

        # 2. Test select_qualifying_labs via Polars
        labs, success = select_qualifying_labs(
            ds,
            train_pids={"p0", "p1"},
            min_prevalence=0.5,
            min_avg_obs=0.5,
            max_labs=10,
        )
        self.assertTrue(success)
        self.assertGreaterEqual(len(labs), 5)
        self.assertIn("glucose", labs)

        # 3. Test select_qualifying_labs with fewer than 5 qualifying labs
        # -> default fallback
        labs_strict, success_strict = select_qualifying_labs(
            ds,
            train_pids={"p0", "p1"},
            min_prevalence=0.99,
            min_avg_obs=100.0,  # none will pass
            max_labs=10,
        )
        self.assertFalse(success_strict)
        self.assertEqual(labs_strict, DEFAULT_LAB_NAMES)

    def test_load_eicu_vitals_cache_corrupt_and_valid(self):
        """Tests automatic detection and deletion of corrupt cache and valid loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = os.path.join(tmp_dir, "corrupt_cache.pkl")
            with open(corrupt_path, "wb") as f:
                f.write(b"truncated invalid pickle data...")

            # Calling with corrupt cache catches exception, removes corrupt file,
            # and recomputes
            vitals, baseline = load_eicu_vitals_cache(
                eicu_root=tmp_dir,
                output_path=corrupt_path,
            )
            self.assertEqual(vitals, {})
            self.assertEqual(baseline, {})
            # Verify the file at corrupt_path was recreated with valid pickle data
            self.assertTrue(os.path.exists(corrupt_path))
            with open(corrupt_path, "rb") as f:
                recomputed = pickle.load(f)
            self.assertEqual(recomputed, ({}, {}))

            # Valid cache loading
            valid_path = os.path.join(tmp_dir, "valid_cache.pkl")
            expected_payload = (
                {"s0": [(1.0, "heartrate", 80.0)]},
                {"s0": {"height": 175.0}},
            )
            with open(valid_path, "wb") as f:
                pickle.dump(expected_payload, f)

            loaded_vitals, loaded_base = load_eicu_vitals_cache(
                eicu_root=tmp_dir,
                output_path=valid_path,
            )
            self.assertEqual(loaded_vitals, expected_payload[0])
            self.assertEqual(loaded_base, expected_payload[1])

    def test_task_lazy_cache_loading_properties(self):
        """Tests LengthOfStayTPCeICU property-based lazy loading of decoupled
        cache files.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_file = os.path.join(tmp_dir, "decoupled_cache.pkl")
            expected_vitals = {"s0": [(1.0, "heartrate", 75.0)]}
            expected_base = {"s0": {"height": 180.0, "weight": 70.0}}
            with open(cache_file, "wb") as f:
                pickle.dump((expected_vitals, expected_base), f)

            task = LengthOfStayTPCeICU(
                vitals_by_stay=cache_file,
                baseline_static_by_stay=cache_file,
            )
            # Before property access, dictionaries are empty StayDataMap
            self.assertEqual(len(task._vitals_data), 0)
            self.assertEqual(len(task._baseline_data), 0)
            self.assertEqual(task.vitals_count, -1)
            self.assertEqual(task.baseline_count, -1)

            # Access .vitals_by_stay property
            vitals = task.vitals_by_stay
            self.assertEqual(len(vitals), 1)
            self.assertIn("s0", vitals)

            # Access .baseline_static_by_stay property
            base = task.baseline_static_by_stay
            self.assertEqual(len(base), 1)
            self.assertIn("s0", base)

    def test_extract_patient_events_malformed_labs_and_offsets(self):
        """Tests handling of null, negative, and unparseable lab result offsets."""
        task = LengthOfStayTPCeICU()
        patient = MockPatient("p_malformed", stay_hours=24)
        # Inject malformed labs into patient events
        patient.events["lab"] = [
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labname": "unknown_lab",
                    "labresultoffset": 100.0,
                    "labresult": 1.0,
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labname": "glucose",
                    "labresultoffset": None,
                    "labresult": 100.0,
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labname": "glucose",
                    "labresultoffset": "unparseable_offset",
                    "labresult": 100.0,
                },
            )(),
            type(
                "Event",
                (),
                {
                    "patientunitstayid": "s0",
                    "labname": "glucose",
                    "labresultoffset": 120.0,
                    "labresult": 95.0,
                },
            )(),
        ]
        records = task.extract_patient_events(patient)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].hourly_vitals), 1)
        self.assertEqual(records[0].hourly_vitals[0][1], "glucose")

    def test_extreme_inputs_and_stress(self):
        """Stress tests TPC with extreme sequence lengths, batch sizes, and values."""
        # 1. Maximum eICU time horizon (T = 336 hours = 14 days)
        model = TPC(
            continuous_dim=8,
            seq_to_seq=True,
            n_layers=3,
            temp_dim=8,
            point_dim=8,
        )
        T_max = 336
        vitals_long = torch.randn(2, T_max, 8)
        y_long = torch.linspace(14.0, 0.0, T_max).view(1, T_max, 1).expand(2, T_max, 1)
        out_long = model(vitals=vitals_long, y_true=y_long)
        self.assertIn("loss", out_long)
        self.assertFalse(torch.isnan(out_long["loss"]))
        out_long["loss"].backward()
        self.assertIsNotNone(model.head_fc.weight.grad)
        assert model.head_fc.weight.grad is not None
        self.assertFalse(torch.isnan(model.head_fc.weight.grad).any())

        # 2. Single-step stay (T = 1 hour)
        model.zero_grad()
        vitals_1 = torch.randn(2, 1, 8)
        out_1 = model(vitals=vitals_1, y_true=torch.ones(2, 1, 1))
        self.assertIn("loss", out_1)
        out_1["loss"].backward()

        # 3. Large batch (B = 64, T = 48)
        model.zero_grad()
        vitals_large = torch.randn(64, 48, 8)
        y_large = torch.ones(64, 48, 1)
        out_large = model(vitals=vitals_large, y_true=y_large)
        self.assertEqual(out_large["y_prob"].shape, torch.Size([64, 48, 1]))
        out_large["loss"].backward()

        # 4. Extreme values (+10^5, -10^5) without NaN
        model.zero_grad()
        vitals_extreme = torch.full((2, 10, 8), 1e5)
        y_extreme = torch.full((2, 10, 1), 1e5)
        out_extreme = model(vitals=vitals_extreme, y_true=y_extreme)
        self.assertFalse(torch.isnan(out_extreme["loss"]))
        self.assertFalse(torch.isinf(out_extreme["loss"]))
        out_extreme["loss"].backward()
        self.assertIsNotNone(model.head_fc.weight.grad)
        assert model.head_fc.weight.grad is not None
        self.assertFalse(torch.isnan(model.head_fc.weight.grad).any())


class TestPyHealthTrainerIntegration(unittest.TestCase):
    """End-to-end integration tests with PyHealth Trainer, GPU, splits, and
    multi-workers.
    """

    def test_tpc_cuda_device_portability_and_mixed_precision(self):
        """Tests TPC execution on real CUDA hardware with FP16 and BF16 autocast."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA hardware is not available on this machine.")

        device = torch.device("cuda:0")
        model = TPC(
            continuous_dim=4,
            static_dim=2,
            time_dim=2,
            n_layers=1,
            temp_dim=4,
            point_dim=4,
            seq_to_seq=False,
        ).to(device)

        vitals = torch.randn(2, 4, 4, device=device)
        vitals_decay = torch.rand(2, 4, 4, device=device)
        static = torch.randn(2, 2, device=device)
        time_feat = torch.rand(2, 4, 2, device=device)
        mask = torch.ones(2, 4, 1, dtype=torch.bool, device=device)
        y_true = torch.rand(2, 1, device=device) + 0.1

        # 1. Full CUDA Forward & Backward
        out = model(
            vitals=vitals,
            vitals_decay=vitals_decay,
            static=static,
            time_features=time_feat,
            mask=mask,
            los=y_true,
        )
        self.assertEqual(out["loss"].device.type, "cuda")
        self.assertFalse(torch.isnan(out["loss"]))
        out["loss"].backward()
        self.assertIsNotNone(model.head_fc.weight.grad)
        assert model.head_fc.weight.grad is not None
        self.assertFalse(torch.isnan(model.head_fc.weight.grad).any())

        # 2. Mixed Precision FP16
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out_fp16 = model(
                vitals=vitals,
                vitals_decay=vitals_decay,
                static=static,
                time_features=time_feat,
                mask=mask,
                los=y_true,
            )
        self.assertFalse(torch.isnan(out_fp16["loss"]))
        out_fp16["loss"].backward()
        self.assertIsNotNone(model.head_fc.weight.grad)
        assert model.head_fc.weight.grad is not None
        self.assertFalse(torch.isnan(model.head_fc.weight.grad).any())

        # 3. Mixed Precision BF16
        model.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_bf16 = model(
                vitals=vitals,
                vitals_decay=vitals_decay,
                static=static,
                time_features=time_feat,
                mask=mask,
                los=y_true,
            )
        self.assertFalse(torch.isnan(out_bf16["loss"]))
        out_bf16["loss"].backward()
        self.assertIsNotNone(model.head_fc.weight.grad)
        assert model.head_fc.weight.grad is not None
        self.assertFalse(torch.isnan(model.head_fc.weight.grad).any())

        # 4. Trainer on CUDA
        samples = [
            {
                "patient_id": f"p_{i}",
                "visit_id": f"v_{i}",
                "vitals": torch.randn(4, 4),
                "static": torch.randn(2),
                "time_features": torch.rand(4, 2),
                "los": torch.rand(1) + 0.5,
            }
            for i in range(2)
        ]
        ds = create_sample_dataset(
            samples,
            input_schema={
                "vitals": "tensor",
                "static": "tensor",
                "time_features": "tensor",
            },
            output_schema={"los": "tensor"},
        )
        loader = get_dataloader(ds, batch_size=2, shuffle=False)
        trainer = Trainer(model=model, device="cuda:0", enable_logging=False)
        trainer.train(train_dataloader=loader, epochs=1)
        scores = trainer.evaluate(loader)
        self.assertIn("loss", scores)

    def test_multi_worker_dataloader_integration(self):
        """Tests DataLoader multi-processing worker safety with BucketBatchSampler."""
        samples = [
            {
                "patient_id": f"p_{i}",
                "vitals": torch.randn(10 + (i % 5), 6),
                "mask": torch.ones(10 + (i % 5), dtype=torch.bool),
                "los": torch.ones(10 + (i % 5), 1),
            }
            for i in range(16)
        ]
        lengths = [s["vitals"].shape[0] for s in samples]
        map_ds = MapDataset(samples)
        batch_sampler = BucketBatchSampler(lengths, batch_size=4, shuffle=True, seed=42)

        loader = DataLoader(
            map_ds,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_dict_with_padding,
            num_workers=2,
            persistent_workers=True,
        )

        batches = list(loader)
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertEqual(batch["vitals"].shape[0], 4)
            self.assertIn("mask", batch)
            self.assertEqual(batch["vitals"].shape[0], batch["mask"].shape[0])

    def test_pyhealth_dataset_splitter_integration(self):
        """Tests split_by_patient and split_by_visit on TPC sample datasets."""
        samples = [
            {
                "patient_id": f"p_{i % 3}",
                "visit_id": f"v_{i}",
                "vitals": torch.randn(24, 6),
                "los": torch.rand(24, 1) + 0.5,
            }
            for i in range(12)
        ]
        ds = create_sample_dataset(
            samples,
            input_schema={"vitals": "tensor"},
            output_schema={"los": "tensor"},
        )

        # Split by patient (no patient overlap between splits)
        train_ds, val_ds, test_ds = split_by_patient(ds, [0.6, 0.2, 0.2], seed=42)
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 12)
        train_pids = {sample["patient_id"] for sample in train_ds}
        val_pids = {sample["patient_id"] for sample in val_ds}
        self.assertEqual(len(train_pids.intersection(val_pids)), 0)

        # Split by visit
        train_v, val_v, test_v = split_by_visit(ds, [0.6, 0.2, 0.2], seed=42)
        self.assertEqual(len(train_v) + len(val_v) + len(test_v), 12)

    def test_trainer_checkpoint_monitor_and_early_stopping(self):
        """Tests checkpoint saving, best model selection, and identical reload."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            samples = [
                {"vitals": torch.randn(24, 6), "los": torch.ones(1)} for _ in range(8)
            ]
            ds = create_sample_dataset(
                samples,
                input_schema={"vitals": "tensor"},
                output_schema={"los": "tensor"},
            )
            loader = get_dataloader(ds, batch_size=4)

            model = TPC(
                dataset=ds,
                continuous_dim=6,
                seq_to_seq=False,
                n_layers=1,
                temp_dim=4,
                point_dim=4,
            )
            trainer = Trainer(
                model=model,
                output_path=tmp_dir,
                exp_name="test_exp",
                device="cpu",
            )

            # Train with early stopping and monitor
            trainer.train(
                train_dataloader=loader,
                val_dataloader=loader,
                epochs=3,
                monitor="loss",
                monitor_criterion="min",
                patience=2,
                load_best_model_at_last=True,
            )

            best_ckpt = os.path.join(tmp_dir, "test_exp", "best.ckpt")
            last_ckpt = os.path.join(tmp_dir, "test_exp", "last.ckpt")
            self.assertTrue(os.path.exists(best_ckpt))
            self.assertTrue(os.path.exists(last_ckpt))

            # Verify loading into fresh model produces identical outputs
            fresh_model = TPC(
                dataset=ds,
                continuous_dim=6,
                seq_to_seq=False,
                n_layers=1,
                temp_dim=4,
                point_dim=4,
            )
            _ = Trainer(
                model=fresh_model,
                checkpoint_path=best_ckpt,
                device="cpu",
                enable_logging=False,
            )

            x = torch.randn(2, 24, 6)
            model.eval()
            fresh_model.eval()
            with torch.no_grad():
                out_orig = model(vitals=x)
                out_loaded = fresh_model(vitals=x)
            diff = (out_orig["y_prob"] - out_loaded["y_prob"]).abs().max().item()
            self.assertEqual(diff, 0.0)


class TestTPCPaperMathematicalFidelity(unittest.TestCase):
    """Rigorous mathematical fidelity and paper equivalence tests.

    Verifies exact conformance with:
    'Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit' (Rocheteau et al., ACM CHIL 2021).
    """

    def test_causal_dilated_convolution_zero_lookahead_eq1(self):
        """Verifies strict causality (zero lookahead) across all layers (Eq. 1)."""
        torch.manual_seed(42)
        model = TPC(
            continuous_dim=6,
            static_dim=4,
            time_dim=2,
            n_layers=3,
            temp_dim=8,
            point_dim=16,
            kernel_size=3,
            seq_to_seq=True,
        )
        model.eval()

        T = 40
        t_perturb = 20
        vitals1 = torch.randn(1, T, 6)
        vitals2 = vitals1.clone()
        # Perturb only time steps >= t_perturb
        vitals2[:, t_perturb:, :] += torch.randn(1, T - t_perturb, 6)

        static = torch.randn(1, 4)
        time_feats = torch.randn(1, T, 2)

        with torch.no_grad():
            out1 = model(vitals=vitals1, static=static, time_features=time_feats)
            out2 = model(vitals=vitals2, static=static, time_features=time_feats)

        # Before perturbation time t_perturb, predictions MUST be identical
        diff_prior = (
            (out1["y_prob"][:, :t_perturb, :] - out2["y_prob"][:, :t_perturb, :])
            .abs()
            .max()
            .item()
        )
        self.assertEqual(
            diff_prior,
            0.0,
            "Causality violated! Future perturbations leaked into past predictions.",
        )

        # At and after t_perturb, predictions SHOULD diverge
        diff_post = (
            (out1["y_prob"][:, t_perturb:, :] - out2["y_prob"][:, t_perturb:, :])
            .abs()
            .max()
            .item()
        )
        self.assertGreater(diff_post, 1e-4)

    def test_channel_accumulation_and_tensor_dimensions_eq2_to_5(self):
        """Verifies channel accumulation R_n = F + (n-1)*Z and pointwise
        broadcast (Eq. 2-5).
        """
        F_feat = 12
        Z_point = 16
        Y_temp = 8
        S_static = 5
        k_size = 3
        T = 24
        B = 2

        # Layer 1
        block1 = TPCBlock(
            layer_idx=1,
            num_features=F_feat,
            point_dim=Z_point,
            temp_dim=Y_temp,
            static_dim=S_static,
            kernel_size=k_size,
            use_decay=True,
            use_skip=True,
        )
        self.assertEqual(block1.R_n, F_feat)
        self.assertEqual(block1.C_n, 2)  # raw + decay

        x_raw = torch.randn(B, F_feat, T)
        x_decay = torch.rand(B, F_feat, T)
        h1 = torch.stack([x_raw, x_decay], dim=2)
        s_exp = torch.randn(B, S_static, T)

        h2, p1 = block1(h1, x_raw, x_decay, s_exp, prev_pointwise=[])
        # After layer 1: R_2 = F + Z = 12 + 16 = 28; Channels = Y + 1 = 9
        self.assertEqual(h2.shape, (B, F_feat + Z_point, Y_temp + 1, T))
        self.assertEqual(p1.shape, (B, Z_point, 1, T))

        # Layer 2
        block2 = TPCBlock(
            layer_idx=2,
            num_features=F_feat,
            point_dim=Z_point,
            temp_dim=Y_temp,
            static_dim=S_static,
            kernel_size=k_size,
            use_decay=True,
            use_skip=True,
        )
        self.assertEqual(block2.R_n, F_feat + Z_point)  # 28
        self.assertEqual(block2.C_n, Y_temp + 1)  # 9

        h3, p2 = block2(h2, x_raw, x_decay, s_exp, prev_pointwise=[p1])
        # After layer 2: R_3 = F + 2*Z = 12 + 32 = 44; Channels = Y + 1 = 9
        self.assertEqual(h3.shape, (B, F_feat + 2 * Z_point, Y_temp + 1, T))
        self.assertEqual(p2.shape, (B, Z_point, 1, T))

    def test_two_stage_per_stay_msle_loss_eq6(self):
        """Verifies per-stay averaging before cohort averaging (Eq. 6)."""
        model = TPC(
            continuous_dim=6,
            loss_fn="msle",
            min_time=4,
            floor_days=1.0 / 48.0,
            seq_to_seq=True,
        )

        # Batch of 2 patients: Patient 0 has 10 hours, Patient 1 has 20 hours
        # Hours [0..3] are warmup (min_time=4).
        # Patient 0 has 6 valid evaluation hours [4..9].
        # Patient 1 has 16 valid evaluation hours [4..19].
        y_pred = torch.tensor(
            [
                [[1.0]] * 10 + [[1.0]] * 10,  # Patient 0 (padded with dummy)
                [[2.0]] * 20,  # Patient 1
            ],
            dtype=torch.float32,
        )

        y_true = torch.tensor(
            [
                [[2.0]] * 10 + [[2.0]] * 10,  # Patient 0
                [[8.0]] * 20,  # Patient 1
            ],
            dtype=torch.float32,
        )

        # Mask: patient 0 valid for 10 hours, patient 1 valid for 20 hours
        mask = torch.zeros(2, 20, dtype=torch.bool)
        mask[0, :10] = True
        mask[1, :20] = True

        loss = model._compute_loss(y_pred, y_true, mask=mask)

        # Hand calculation:
        # Patient 0: for hours 4..9 (6 hours), pred=1.0, true=2.0
        # log(pred)=0, log(true)=ln(2). Squared error = (ln 2)^2 approx 0.480453
        # Average per stay for patient 0 = (ln 2)^2.
        # Patient 1: for hours 4..19 (16 hours), pred=2.0, true=8.0
        # log(pred)=ln 2, log(true)=ln 8 = 3 ln 2. Diff = -2 ln 2.
        # Squared error = 4 * (ln 2)^2.
        # Average per stay for patient 1 = 4 * (ln 2)^2.
        # Final loss = 0.5 * ((ln 2)^2 + 4 * (ln 2)^2) = 2.5 * (ln 2)^2
        # approx 1.201132
        ln2 = np.log(2.0)
        expected_loss = 2.5 * (ln2**2)

        self.assertAlmostEqual(loss.item(), expected_loss, places=5)

    def test_ablation_matrix_behaviors_table3_and_4(self):
        """Verifies weight sharing, skip removal, decay removal, and batchnorm
        variants.
        """
        # 1. Weight Sharing (share_weights=True)
        block_ws = TPCBlock(
            layer_idx=1,
            num_features=6,
            point_dim=8,
            temp_dim=4,
            static_dim=2,
            share_weights=True,
        )
        # Groups = 1, in_channels = C_n (2), out_channels = temp_dim (4)
        self.assertEqual(block_ws.temporal_conv.groups, 1)
        self.assertEqual(block_ws.temporal_conv.in_channels, 2)
        self.assertEqual(block_ws.temporal_conv.out_channels, 4)

        # 2. Independent Weights (share_weights=False, default TPC)
        block_ind = TPCBlock(
            layer_idx=1,
            num_features=6,
            point_dim=8,
            temp_dim=4,
            static_dim=2,
            share_weights=False,
        )
        # Groups = R_n (6), in_channels = 6 * 2 = 12, out_channels = 6 * 4 = 24
        self.assertEqual(block_ind.temporal_conv.groups, 6)
        self.assertEqual(block_ind.temporal_conv.in_channels, 12)
        self.assertEqual(block_ind.temporal_conv.out_channels, 24)

        # 3. No Skip Connections (use_skip=False)
        block_noskip = TPCBlock(
            layer_idx=2,
            num_features=6,
            point_dim=8,
            temp_dim=4,
            static_dim=2,
            use_skip=False,
        )
        # R_n does NOT accumulate previous pointwise features
        self.assertEqual(block_noskip.R_n, 6)
        self.assertEqual(block_noskip.C_n, 4)  # temp_dim only

        # 4. No Decay (use_decay=False)
        block_nodecay = TPCBlock(
            layer_idx=1,
            num_features=6,
            point_dim=8,
            temp_dim=4,
            static_dim=2,
            use_decay=False,
        )
        self.assertEqual(block_nodecay.C_n, 1)  # only raw vitals

    def test_metrics_fidelity_against_rocheteau_reference(self):
        """Verifies Cohen linear weighted kappa and MAPE floor matching metrics.py."""
        # Test Cohen's linear weighted kappa with custom 10-bin edges
        y_true = np.array([0.5, 1.5, 2.5, 3.5, 7.5, 15.0])
        y_pred = np.array([0.8, 1.2, 2.8, 3.1, 7.1, 16.0])

        metrics_res = compute_los_metrics(
            y_true=y_true,
            y_pred=y_pred,
            metrics=["binned_kappa", "mape", "msle", "mse", "mae"],
            min_eval_hours=0,
        )
        # Predictions fall into exact same bins -> perfect kappa = 1.0
        self.assertAlmostEqual(metrics_res["binned_kappa"], 1.0, places=4)

        # Verify MAPE floor: denom = max(yt, 4/24)
        yt_small = np.array([0.05])  # < 4/24 = 0.166667
        yp_small = np.array([0.15])
        res_small = compute_los_metrics(
            y_true=yt_small,
            y_pred=yp_small,
            metrics=["mape"],
            min_eval_hours=0,
            mape_floor_days=4.0 / 24.0,
        )
        # |0.15 - 0.05| / (4/24) * 100 = 0.10 / 0.166667 * 100 = 60.0%
        self.assertAlmostEqual(res_small["mape"], 60.0, places=2)


class TestTPCCheckpointResume(unittest.TestCase):
    """Verifies that checkpoint/resume in example runner matches uninterrupted training."""

    def test_resume_matches_uninterrupted_training(self):
        """Validates that training with interruption and --resume matches uninterrupted run."""
        import json

        from examples.eicu_los_tpc import run_ablation_study
        from pyhealth.tasks.length_of_stay_tpc import BucketBatchSampler

        torch.set_num_threads(1)

        def sample(index: int):
            values = torch.full((8, 2), 0.1 * (index + 1))
            return {
                "vitals": values,
                "vitals_decay": torch.ones_like(values),
                "static": torch.tensor([index / 10.0]),
                "time_features": torch.tensor(
                    [[(t + 1) / 24.0, t / 23.0] for t in range(8)]
                ),
                "mask": torch.ones(8, dtype=torch.bool),
                "los": torch.tensor([[2.0 - t / 24.0] for t in range(8)]),
                "mortality": torch.tensor([float(index % 2)]),
            }

        train = [sample(i) for i in range(4)]
        val = [sample(i) for i in range(4, 6)]
        test = [sample(i) for i in range(6, 8)]

        def run(directory: Path, epochs: int, resume: bool = False):
            sampler = BucketBatchSampler([8] * len(train), batch_size=2, seed=7)
            train_loader = DataLoader(MapDataset(train), batch_sampler=sampler)
            val_loader = DataLoader(MapDataset(val), batch_size=2)
            test_loader = DataLoader(MapDataset(test), batch_size=2)
            return run_ablation_study(
                train_loader,
                val_loader,
                test_loader,
                sampler,
                continuous_dim=2,
                static_dim=1,
                time_dim=2,
                n_layers=1,
                kernel_size=2,
                temp_dim=2,
                point_dim=2,
                hidden_dim=3,
                dropout=0.1,
                temp_dropout=0.1,
                batchnorm="none",
                epochs=epochs,
                lr=0.001,
                device="cpu",
                seed=7,
                checkpoint_dir=directory,
                resume=resume,
                data_signature="synthetic-v1",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            full = temp_path / "full"
            resumed = temp_path / "resumed"
            run(full, epochs=2)
            run(resumed, epochs=1)
            run(resumed, epochs=2, resume=True)

            for subdir in sorted(full.iterdir()):
                self.assertTrue((subdir / "latest.pt").is_file())
                self.assertTrue((subdir / "best.pt").is_file())
                self.assertTrue((subdir / "metrics.json").is_file())
                self.assertTrue(bool(json.loads((subdir / "metrics.json").read_text())))
                uninterrupted = torch.load(subdir / "latest.pt", weights_only=False)
                continued = torch.load(
                    resumed / subdir.name / "latest.pt", weights_only=False
                )
                self.assertEqual(uninterrupted["completed_epoch"], 2)
                self.assertEqual(continued["completed_epoch"], 2)
                for key in uninterrupted["model_state"]:
                    torch.testing.assert_close(
                        uninterrupted["model_state"][key],
                        continued["model_state"][key],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
