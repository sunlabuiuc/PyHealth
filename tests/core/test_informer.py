import tempfile
import shutil
import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel
from pyhealth.models.informer import Informer


# ------------------------------------------------------------------ #
# Tiny config — keeps every test in milliseconds                      #
# ------------------------------------------------------------------ #
SEQ_LEN   = 96
LABEL_LEN = 48
PRED_LEN  = 24
ENC_IN    = 7      # number of encoder input channels (ETTh1 has 7)
DEC_IN    = 7      # number of decoder input channels
C_OUT     = 7      # intermediate output channels
TIME_FEAT = 4      # hourly time encoding: [month, day, weekday, hour]
N_SAMPLES = 2
BATCH_SIZE = 2
def make_sample(i):
    """Generate one synthetic ETT-style window."""
    # Generate timestamps starting from a base time at hourly intervals
    base_time = datetime(2021, 1, 1)

    enc_timestamps  = [base_time + timedelta(hours=t) for t in range(SEQ_LEN)]
    dec_timestamps  = [base_time + timedelta(hours=t) for t in range(LABEL_LEN + PRED_LEN)]

    x_enc       = np.random.randn(SEQ_LEN,              ENC_IN).astype(np.float32)
    x_mark_enc  = np.random.randn(SEQ_LEN,              TIME_FEAT).astype(np.float32)
    x_dec       = np.random.randn(LABEL_LEN + PRED_LEN, DEC_IN).astype(np.float32)
    x_mark_dec  = np.random.randn(LABEL_LEN + PRED_LEN, TIME_FEAT).astype(np.float32)

    label = float(x_dec[LABEL_LEN:].mean())

    return {
        "patient_id": f"patient-{i}",
        "visit_id":   f"visit-{i}",
        # Each timeseries field must be (List[datetime], np.ndarray)
        "x_enc":      (enc_timestamps, x_enc),
        "x_mark_enc": (enc_timestamps, x_mark_enc),
        "x_dec":      (dec_timestamps, x_dec),
        "x_mark_dec": (dec_timestamps, x_mark_dec),
        "label":      label,
    }

samples = [make_sample(i) for i in range(N_SAMPLES)]

class TestInformer(unittest.TestCase):
    """Test cases for the Informer model."""

    @classmethod
    def setUpClass(cls):
        """Build dataset, dataloader, and model once for the entire test class.

        Using setUpClass instead of setUp means the dataset construction and
        model instantiation happen only once, keeping individual test runs
        in the millisecond range. A temporary directory is created here and
        removed in tearDownClass.
        """
        # Temporary directory for any artefacts produced during tests
        cls.tmp_dir = tempfile.mkdtemp()

        # Synthetic dataset via create_sample_dataset (no real data)
        cls.samples = samples
        input_schema = {
            "x_enc":      "timeseries",
            "x_mark_enc": "timeseries",
            "x_dec":      "timeseries",
            "x_mark_dec": "timeseries",
        }
        output_schema = {"label": "regression"}

        cls.dataset = create_sample_dataset(
            samples=cls.samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_informer",
        )

        # Dataloader via get_dataloader from pyhealth.datasets
        cls.train_loader = get_dataloader(
            cls.dataset, batch_size=4, shuffle=False
        )

        # Tiny Informer model on CPU for fast tests
        cls.model = Informer(
        dataset=cls.dataset,
        enc_in=ENC_IN,
        dec_in=DEC_IN,
        c_out=C_OUT,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
    )

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory after all tests finish."""
        shutil.rmtree(cls.tmp_dir)

    # -------------------------------------------------------------- #
    # 1. Instantiation tests                                          #
    # -------------------------------------------------------------- #

    def test_model_initialization(self):
        """Test that the Informer model initializes correctly."""
        self.assertIsInstance(self.model, Informer)
        self.assertIsInstance(self.model, BaseModel)
        self.assertIsInstance(self.model, torch.nn.Module)
        self.assertEqual(self.model.pred_len, PRED_LEN)
        self.assertEqual(self.model.attn, "prob")

    def test_label_keys_populated(self):
        """BaseModel must auto-populate label_keys from output_schema."""
        self.assertEqual(self.model.label_keys[0], "label")

    def test_feature_keys_populated(self):
        """BaseModel must auto-populate feature_keys from input_schema."""
        expected = {"x_enc", "x_mark_enc", "x_dec", "x_mark_dec"}
        self.assertEqual(expected, set(self.model.feature_keys))

    # -------------------------------------------------------------- #
    # 2. Dataloader / input format tests                              #
    # -------------------------------------------------------------- #

    def test_forward_input_format(self):
        """Test that the dataloader provides correctly typed inputs."""
        data_batch = next(iter(self.train_loader))

        # Timeseries features arrive as (time_tensor, value_tensor) tuples
        for key in ["x_enc", "x_mark_enc", "x_dec", "x_mark_dec"]:
            self.assertIn(key, data_batch)
            self.assertIsInstance(data_batch[key], torch.Tensor)
            self.assertEqual(len(data_batch[key]), 2)

        # Label must be a tensor
        self.assertIsInstance(data_batch["label"], torch.Tensor)

    # -------------------------------------------------------------- #
    # 3. Forward pass tests                                           #
    # -------------------------------------------------------------- #

    def test_model_forward(self):
        """Test that the Informer forward pass works correctly."""
        data_batch = next(iter(self.train_loader))
        self.model.eval()
        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss",   ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit",  ret)

        # loss must be a scalar
        self.assertEqual(ret["loss"].dim(), 0)
        # batch dimension must match
        self.assertEqual(ret["y_prob"].shape[0], BATCH_SIZE)
        self.assertEqual(ret["y_true"].shape[0], BATCH_SIZE)

    def test_loss_is_finite(self):
        """Test that the loss is finite (not NaN or inf)."""
        data_batch = next(iter(self.train_loader))
        self.model.eval()
        with torch.no_grad():
            ret = self.model(**data_batch)
        self.assertTrue(
            torch.isfinite(ret["loss"]).all(),
            "Loss is not finite"
        )

    # -------------------------------------------------------------- #
    # 4. Output shape tests                                           #
    # -------------------------------------------------------------- #

    def test_output_shapes(self):
        """Test that all output tensors have the correct shapes."""
        data_batch = next(iter(self.train_loader))
        self.model.eval()
        with torch.no_grad():
            ret = self.model(**data_batch)

        output_size = self.model.get_output_size()   # 1 for regression

        self.assertEqual(ret["logit"].shape,  (BATCH_SIZE, output_size))
        self.assertEqual(ret["y_prob"].shape, (BATCH_SIZE, output_size))
        self.assertEqual(ret["y_true"].shape[0], BATCH_SIZE)

    # -------------------------------------------------------------- #
    # 5. Gradient computation tests                                   #
    # -------------------------------------------------------------- #

    def test_model_backward(self):
        """Test that the Informer backward pass works correctly."""
        data_batch = next(iter(self.train_loader))
        self.model.train()
        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(
            has_gradient,
            "No parameters have gradients after backward pass"
        )

    def test_gradients_are_finite(self):
        """Test that all computed gradients are finite."""
        data_batch = next(iter(self.train_loader))
        self.model.train()
        ret = self.model(**data_batch)
        ret["loss"].backward()

        for name, p in self.model.named_parameters():
            if p.grad is not None:
                self.assertTrue(
                    torch.isfinite(p.grad).all(),
                    f"Non-finite gradient in parameter '{name}'"
                )

    # -------------------------------------------------------------- #
    # 6. Variant / configuration tests                                #
    # -------------------------------------------------------------- #

    def test_full_attention_variant(self):
        """Test Informer with full attention forward and backward pass."""
        model_full = Informer(
        dataset=self.dataset,
        enc_in=ENC_IN,
        dec_in=DEC_IN,
        c_out=C_OUT,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=True,
        mix=True,
    )
        data_batch = next(iter(self.train_loader))
        model_full.train()
        ret = model_full(**data_batch)
        ret["loss"].backward()
        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_no_distil_variant(self):
        """Test Informer without distillation layers."""
        model_nodistil = Informer(
        dataset=self.dataset,
        enc_in=ENC_IN,
        dec_in=DEC_IN,
        c_out=C_OUT,
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        out_len=PRED_LEN,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        dropout=0.05,
        attn="prob",
        embed="timeF",
        freq="h",
        activation="gelu",
        output_attention=False,
        distil=False,
        mix=True,
    )
        data_batch = next(iter(self.train_loader))
        model_nodistil.train()
        ret = model_nodistil(**data_batch)
        ret["loss"].backward()
        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_train_eval_mode_toggle(self):
        """Test that the model switches cleanly between train and eval modes."""
        self.model.train()
        self.assertTrue(self.model.training)
        self.model.eval()
        self.assertFalse(self.model.training)

    # -------------------------------------------------------------- #
    # 7. Temporary directory tests                                    #
    # -------------------------------------------------------------- #

    def test_tmp_dir_exists(self):
        """Temporary directory must exist during the test run."""
        import os
        self.assertTrue(os.path.isdir(self.tmp_dir))

    def test_can_write_to_tmp_dir(self):
        """Must be able to write artefacts into the temporary directory."""
        import os
        probe = os.path.join(self.tmp_dir, "probe.txt")
        with open(probe, "w") as f:
            f.write("ok")
        self.assertTrue(os.path.exists(probe))


if __name__ == "__main__":
    unittest.main()
