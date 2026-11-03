"""Trainer mixed precision and gradient accumulation.

``amp_dtype`` used to fall through to fp16 for any unknown spelling, and the
autocast context was hardcoded to CUDA. Accumulation lets small synthetic
batches simulate a larger effective batch size.

With ``use_amp=False`` the step matches ``upstream/master`` exactly: clip only
when ``max_grad_norm`` is set, no unconditional grad-norm pass, no skipping.
``accumulation_steps=1`` is bit-identical in effect to upstream.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/core/test_trainer_amp.py -q
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TinyModel(nn.Module):
    mode = None

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x, y):
        logit = self.fc(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        return {"loss": loss}


def _loader(n=16, batch_size=4, seed=0):
    torch.manual_seed(seed)
    xs = torch.randn(n, 4)
    ys = (xs.sum(dim=1) > 0).float()
    ds = TensorDataset(xs, ys)

    def collate(batch):
        xs_b, ys_b = zip(*batch)
        return {"x": torch.stack(list(xs_b)), "y": torch.stack(list(ys_b))}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _upstream_reference_train(model, loader, epochs, max_grad_norm, lr=1e-3):
    """The upstream/master batch loop, copied for equivalence checking."""
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param = list(model.named_parameters())
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in param if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
    )
    data_iterator = iter(loader)
    steps_per_epoch = len(loader)
    for _ in range(epochs):
        model.zero_grad()
        model.train()
        for _ in range(steps_per_epoch):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                data = next(data_iterator)
            output = model(**data)
            loss = output["loss"]
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()


class TestTrainerAmp(unittest.TestCase):
    def test_known_spellings_map_to_the_named_dtype(self):
        from pyhealth.trainer import resolve_amp_dtype

        self.assertIs(resolve_amp_dtype("bf16"), torch.bfloat16)
        self.assertIs(resolve_amp_dtype("bfloat16"), torch.bfloat16)
        self.assertIs(resolve_amp_dtype("fp16"), torch.float16)
        self.assertIs(resolve_amp_dtype("float16"), torch.float16)

    def test_unknown_spelling_raises(self):
        from pyhealth.trainer import resolve_amp_dtype

        for bad in ("bfloat_16", "f16", "int8", ""):
            with self.assertRaises(ValueError):
                resolve_amp_dtype(bad)

    def test_train_rejects_unknown_amp_dtype(self):
        from pyhealth.trainer import Trainer

        trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
        with self.assertRaises(ValueError):
            trainer.train(_loader(), epochs=1, amp_dtype="f16")

    def test_invalid_accumulation_steps_raises(self):
        from pyhealth.trainer import Trainer

        trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
        for bad in (0, -2):
            with self.assertRaises(ValueError):
                trainer.train(_loader(), epochs=1, accumulation_steps=bad)

    def test_autocast_device_follows_the_trainer_device(self):
        from pyhealth.trainer import autocast_device_type

        self.assertEqual(autocast_device_type("cuda:0"), "cuda")
        self.assertEqual(autocast_device_type("cpu"), "cpu")
        self.assertEqual(autocast_device_type("mps"), "mps")
        self.assertEqual(autocast_device_type("xpu"), "cpu")

    def test_default_path_clips_only_when_configured(self):
        from pyhealth.trainer import Trainer

        with mock.patch(
            "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
        ) as clip:
            trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
            trainer.train(_loader(), epochs=1, max_grad_norm=None)
            clip.assert_not_called()
        with mock.patch(
            "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
        ) as clip:
            trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
            trainer.train(_loader(), epochs=1, max_grad_norm=1.0)
            self.assertEqual(clip.call_count, len(_loader()))

    def test_accumulation_one_matches_upstream_bit_identical(self):
        from pyhealth.trainer import Trainer

        torch.manual_seed(123)
        model_new = _TinyModel()
        torch.manual_seed(123)
        model_ref = _TinyModel()
        loader = _loader()
        Trainer(model_new, enable_logging=False, device="cpu").train(
            loader, epochs=2, accumulation_steps=1, max_grad_norm=1.0
        )
        _upstream_reference_train(model_ref, _loader(), 2, max_grad_norm=1.0)
        for p_new, p_ref in zip(
            model_new.parameters(), model_ref.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(p_new, p_ref))

    def test_accumulated_halves_match_full_batch(self):
        from pyhealth.trainer import Trainer

        torch.manual_seed(7)
        model_full = _TinyModel()
        torch.manual_seed(7)
        model_acc = _TinyModel()
        Trainer(model_full, enable_logging=False, device="cpu").train(
            _loader(n=8, batch_size=8),
            epochs=1,
            accumulation_steps=1,
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.1},
        )
        Trainer(model_acc, enable_logging=False, device="cpu").train(
            _loader(n=8, batch_size=4),
            epochs=1,
            accumulation_steps=2,
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.1},
        )
        for p_full, p_acc in zip(
            model_full.parameters(), model_acc.parameters(), strict=True
        ):
            self.assertTrue(torch.allclose(p_full, p_acc, atol=1e-6))

    def test_trailing_partial_window_still_steps(self):
        from pyhealth.trainer import Trainer

        counts = []

        class CountingSGD(torch.optim.SGD):
            def step(self, *args, **kwargs):
                counts.append(1)
                return super().step(*args, **kwargs)

        trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
        trainer.train(
            _loader(n=12, batch_size=4),
            epochs=1,
            steps_per_epoch=3,
            accumulation_steps=2,
            optimizer_class=CountingSGD,
            optimizer_params={"lr": 0.1},
        )
        # batches 1-2 form one update, batch 3 is flushed at epoch end
        self.assertEqual(len(counts), 2)

    def test_steps_per_epoch_shorter_than_loader(self):
        from pyhealth.trainer import Trainer

        trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.train(_loader(n=16, batch_size=4), epochs=2, steps_per_epoch=2)
        after = [p.detach().clone() for p in trainer.model.parameters()]
        self.assertTrue(any(not torch.equal(b, a) for b, a in zip(before, after)))

    def test_accumulation_trains_on_cpu(self):
        from pyhealth.trainer import Trainer

        for accumulation_steps in (1, 2):
            trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
            before = [p.detach().clone() for p in trainer.model.parameters()]
            trainer.train(
                _loader(), epochs=1, accumulation_steps=accumulation_steps
            )
            after = [p.detach().clone() for p in trainer.model.parameters()]
            self.assertTrue(
                any(not torch.equal(b, a) for b, a in zip(before, after))
            )

    def test_amp_bf16_trains_on_cpu(self):
        from pyhealth.trainer import Trainer

        trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
        trainer.train(_loader(), epochs=1, use_amp=True, amp_dtype="bf16")

    def test_amp_fp16_trains_on_cpu(self):
        from pyhealth.trainer import Trainer

        trainer = Trainer(_TinyModel(), enable_logging=False, device="cpu")
        trainer.train(_loader(), epochs=1, use_amp=True, amp_dtype="fp16")
