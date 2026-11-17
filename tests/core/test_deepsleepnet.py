"""Tests for the DeepSleepNet model. All data is synthetic."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepSleepNet
from pyhealth.models.deepsleepnet import _same_padding
from pyhealth.processors import MultiClassLabelProcessor
from pyhealth.trainer import Trainer

SEQ_LEN = 4
EPOCH_LEN = 3000  # 30 s at 100 Hz


def _dataset(n_samples=4, shape=(SEQ_LEN, 1, EPOCH_LEN)):
    rng = np.random.default_rng(0)
    samples = [
        {
            "patient_id": str(i),
            "signal": rng.standard_normal(shape).astype(np.float32),
            "label": rng.integers(0, 5, size=shape[0]),
        }
        for i in range(n_samples)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": ("tensor", {"dtype": torch.long})},
    )


class TestDeepSleepNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = _dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=2)))

    def _model(self, **kwargs):
        kwargs.setdefault("n_conv", 2)
        kwargs.setdefault("hidden_dim", 8)
        return DeepSleepNet(dataset=self.dataset, **kwargs)

    def test_same_padding_matches_tensorflow(self):
        # TF "SAME": ceil(length / stride) outputs, extra padding on the right.
        self.assertEqual(_same_padding(3000, 50, 6), (22, 22))  # 500 outputs
        self.assertEqual(_same_padding(63, 4, 4), (0, 1))  # 16 outputs
        self.assertEqual(_same_padding(16, 2, 2), (0, 0))

    def test_feature_sizes_of_the_reference_and_sleepkd_student(self):
        # Reference: small branch 16 x 128, large branch 8 x 128.
        reference = self._model(n_conv=4)
        self.assertEqual(reference.epoch_dim, 16 * 128 + 8 * 128)
        # SleepKD student: one 64-filter convolution per branch, 128 LSTM units.
        student = self._model(n_conv=1, hidden_dim=128)
        self.assertEqual(student.epoch_dim, 16 * 64 + 8 * 64)
        self.assertEqual(student.sequence_dim, 256)

    def test_sleepkd_layout_has_the_parameters_of_sleepkd_table_5(self):
        # Counted as Keras does: BatchNorm moving statistics included, one
        # LSTM bias vector per gate instead of PyTorch's two.
        def keras_count(model):
            count = sum(p.numel() for p in model.parameters())
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    count += m.running_mean.numel() + m.running_var.numel()
            return count - sum(
                p.numel() for n, p in model.lstm.named_parameters() if "bias_hh" in n
            )

        for (n_conv, hidden_dim), expected in {
            (4, 512): 5_502_474,
            (1, 128): 1_552_906,
        }.items():
            model = self._model(
                n_conv=n_conv, hidden_dim=hidden_dim, n_rnn_layers=1, layout="sleepkd"
            )
            self.assertEqual(keras_count(model), expected)

    def test_sleepkd_layout_features(self):
        model = self._model(layout="sleepkd").eval()
        out = model.encode(self.batch["signal"])
        self.assertEqual(model.epoch_dim, 1024)
        self.assertEqual(out["epoch_feature"].shape, (2, SEQ_LEN, 1024))
        self.assertEqual(out["sequence_feature"].shape, (2, SEQ_LEN, 16))
        self.assertEqual(out["logit"].shape, (2, SEQ_LEN, 5))
        # The sequence features are the LSTM output alone.
        lstm_input = model.epoch_head(out["epoch_feature"])
        lstm_out, _ = model.lstm(lstm_input)
        torch.testing.assert_close(out["sequence_feature"], lstm_out)

    def test_encode_returns_features_of_the_announced_sizes(self):
        model = self._model()
        out = model.encode(self.batch["signal"])
        self.assertEqual(out["epoch_feature"].shape, (2, SEQ_LEN, model.epoch_dim))
        self.assertEqual(out["sequence_feature"].shape, (2, SEQ_LEN, 16))
        self.assertEqual(out["logit"].shape, (2, SEQ_LEN, 5))

    def test_forward_flattens_epochs_and_backpropagates(self):
        model = self._model()
        out = model(**self.batch, embed=True)
        self.assertEqual(out["y_prob"].shape, (2 * SEQ_LEN, 5))
        self.assertEqual(out["y_true"].shape, (2 * SEQ_LEN,))
        self.assertEqual(out["embed"].shape, (2 * SEQ_LEN, 16))
        torch.testing.assert_close(out["y_prob"].sum(1), torch.ones(2 * SEQ_LEN))
        out["loss"].backward()
        self.assertIsNotNone(model.small.layers[0].conv.weight.grad)

    def test_later_epochs_inform_earlier_predictions(self):
        # The LSTM is bidirectional: the first epoch's output sees the last.
        model = self._model().eval()
        x = self.batch["signal"].clone()
        before = model.encode(x)["logit"][:, 0]
        x[:, -1] += 1.0
        after = model.encode(x)["logit"][:, 0]
        self.assertFalse(torch.allclose(before, after))

    def test_trainer_scores_every_epoch_as_multiclass(self):
        model = self._model()
        self.assertIsInstance(model.label_processors["label"], MultiClassLabelProcessor)
        trainer = Trainer(
            model=model, metrics=["accuracy", "f1_macro"], enable_logging=False
        )
        loader = get_dataloader(self.dataset, batch_size=2)
        trainer.train(train_dataloader=loader, epochs=1)
        y_true, y_prob, _ = trainer.inference(loader)
        self.assertEqual(y_true.shape, (len(self.dataset) * SEQ_LEN,))
        self.assertEqual(y_prob.shape, (len(self.dataset) * SEQ_LEN, 5))
        scores = trainer.evaluate(loader)
        self.assertEqual(set(scores), {"accuracy", "f1_macro", "loss"})

    def test_batchnorm_never_sees_dropout(self):
        # BatchNorm's running statistics must not depend on dropout, or they
        # are wrong in evaluation, where dropout is off.
        def running_stats(dropout_on, layout):
            torch.manual_seed(1)
            model = self._model(n_conv=4, layout=layout)
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout) and not dropout_on:
                    m.eval()
            model(**self.batch)
            return [
                m.running_var.clone()
                for m in model.modules()
                if isinstance(m, torch.nn.BatchNorm1d)
            ]

        for layout in ("supratak", "sleepkd"):
            with self.subTest(layout=layout):
                for on, off in zip(
                    running_stats(True, layout), running_stats(False, layout)
                ):
                    torch.testing.assert_close(on, off)

    def test_rejects_inputs_without_an_epoch_axis(self):
        with self.assertRaises(ValueError):
            DeepSleepNet(dataset=_dataset(shape=(SEQ_LEN, EPOCH_LEN)))

    def test_rejects_no_convolution(self):
        with self.assertRaises(ValueError):
            self._model(n_conv=0)

    def test_rejects_unknown_layout(self):
        with self.assertRaises(ValueError):
            self._model(layout="keras")


if __name__ == "__main__":
    unittest.main()
