"""Batch padding is recorded and skipped by the unified path.

The collator padded short samples with 0.0 and nothing recorded that padding,
so padded slots looked like real measurements at admission time.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn


class TestP1RnnClamp(unittest.TestCase):
    def test_all_pad_mask_forward_is_finite(self):
        from pyhealth.models.rnn import RNNLayer

        layer = RNNLayer(input_size=4, hidden_size=8, dropout=0.0).eval()
        x = torch.zeros(2, 5, 4)
        mask = torch.zeros(2, 5)
        with torch.no_grad():
            outputs, last = layer(x, mask)
        self.assertEqual(outputs.shape[0], 2)
        self.assertEqual(tuple(last.shape), (2, 8))
        self.assertTrue(torch.isfinite(last).all())


class TestP1BertPadSkip(unittest.TestCase):
    def test_collate_emits_false_pad_mask_on_empty_note_slots(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

        batch = [
            {"notes": (torch.ones(3, 4, dtype=torch.long), torch.ones(3, 4))},
            {"notes": (torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4))},
        ]
        collated = collate_fn_dict_with_padding(batch)
        ids = collated["notes"][1]
        pad = collated[f"notes{PAD_MASK_SUFFIX}"]
        self.assertEqual(tuple(ids.shape), (2, 3, 4))
        self.assertEqual(pad.tolist(), [[True, True, True], [True, False, False]])

    def test_padded_events_sort_last_and_are_zeroed(self):
        from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors.stagenet_processor import StageNetTensorProcessor

        proc = StageNetTensorProcessor()
        proc.fit([{"labs": ([0.0], [[1.0, 2.0]])}], "labs")
        model = UnifiedMultimodalEmbeddingModel(
            processors={"labs": proc},
            embedding_dim=8,
            normalize_content=True,
        )
        model.eval()
        value = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])
        time = torch.tensor([[6.0, 0.0]])
        pad_mask = torch.tensor([[True, False]])
        with torch.no_grad():
            out = model({"labs": {"value": value, "time": time, "pad_mask": pad_mask}})
        self.assertEqual(out["mask"].tolist(), [[1.0, 0.0]])
        self.assertAlmostEqual(out["time"][0, 0].item(), 6.0)
        self.assertTrue(torch.allclose(out["sequence"][0, 1], torch.zeros(8), atol=1e-6))

    def test_text_encoder_skips_padded_note_slots(self):
        from types import SimpleNamespace

        from pyhealth.models.unified_embedding import UnifiedMultimodalEmbeddingModel
        from pyhealth.processors import TupleTimeTextProcessor

        class CountingEnc(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.lin = nn.Linear(4, 8)
                self.config = type("C", (), {"hidden_size": 8})()

            def forward(self, input_ids, attention_mask=None):
                self.calls += 1
                h = self.lin(input_ids.float())
                return type("O", (), {"last_hidden_state": h.unsqueeze(1)})()

        proc = TupleTimeTextProcessor()
        enc = CountingEnc()
        pre = SimpleNamespace(transformer=enc, fc=nn.Identity(), embedding_dim=8)
        model = UnifiedMultimodalEmbeddingModel(
            {"notes": proc}, embedding_dim=8, field_embeddings={"notes": pre}
        )
        model.eval()
        with torch.no_grad():
            out = model(
                {
                    "notes": {
                        "value": torch.ones(1, 3, 4, dtype=torch.long),
                        "time": torch.tensor([[2.0, 0.0, 1.0]]),
                        "pad_mask": torch.tensor([[True, False, True]]),
                    }
                }
            )
        # One batched call over the 2 valid rows; the padded slot sorts last.
        self.assertEqual(enc.calls, 1)
        self.assertEqual(out["sequence"].shape, (1, 3, 8))
        self.assertEqual(out["mask"].tolist(), [[1.0, 1.0, 0.0]])
        self.assertEqual(out["time"].tolist(), [[1.0, 2.0, 0.0]])

    def test_unified_heads_thread_pad_mask(self):
        import inspect

        from pyhealth.models.ehrmamba import EHRMamba
        from pyhealth.models.jamba_ehr import JambaEHR
        from pyhealth.models.rnn import RNN
        from pyhealth.models.transformer import Transformer

        for cls in (RNN, Transformer, EHRMamba, JambaEHR):
            src = inspect.getsource(cls._build_unified_inputs)
            self.assertIn("PAD_MASK_SUFFIX", src, msg=cls.__name__)
            self.assertIn("pad_mask", src, msg=cls.__name__)


class TestP1TextTupleCollate(unittest.TestCase):
    """Ragged text tuples collate positionally like (time, values) pairs.

    Regression tests: tokenized note fields collate as one 5-tuple per
    sample ``(value, mask, token_type_ids, time, tag)`` with a ragged note
    count, which the old 2-tuple-only branch kept as a nested list. Every
    unified backbone then crashed in ``_build_unified_inputs`` indexing
    into per-sample tuples (``'tuple' object has no attribute 'to'``).
    """

    @staticmethod
    def _note(n, l, t0):
        return (
            torch.ones(n, l, dtype=torch.long),
            torch.ones(n, l, dtype=torch.long),
            torch.zeros(n, l, dtype=torch.long),
            torch.tensor([t0 + i for i in range(n)], dtype=torch.float),
            "note",
        )

    @staticmethod
    def _text_processor(tokenized=True):
        from pyhealth.processors import TupleTimeTextProcessor

        # collate() only reads whether a tokenizer is set, so skip the download.
        processor = TupleTimeTextProcessor.__new__(TupleTimeTextProcessor)
        processor.tokenizer = object() if tokenized else None
        return processor

    def test_ragged_text_tuples_stack_through_the_processor_collate(self):
        from pyhealth.datasets.utils import collate_fn_dict_with_padding

        batch = [
            {"notes": self._note(2, 4, 0.0)},
            {"notes": self._note(1, 4, 5.0)},
            {"notes": self._note(0, 1, 0.0)},
        ]
        collated = collate_fn_dict_with_padding(
            batch, collators={"notes": self._text_processor().collate}
        )
        value, mask, type_ids, time, tag = collated["notes"]
        self.assertEqual(tuple(value.shape), (3, 2, 4))
        self.assertEqual(tuple(mask.shape), (3, 2, 4))
        self.assertEqual(tuple(type_ids.shape), (3, 2, 4))
        self.assertEqual(tuple(time.shape), (3, 2))
        self.assertEqual(tag, ["note", "note", "note"])
        # A padded note slot has an all-zero attention mask.
        self.assertEqual(
            mask.any(dim=-1).tolist(),
            [[True, True], [True, False], [False, False]],
        )

    def test_text_tuples_stay_a_list_without_the_processor_collate(self):
        from pyhealth.datasets.utils import collate_fn_dict_with_padding

        batch = [{"notes": self._note(2, 4, 0.0)}, {"notes": self._note(1, 4, 5.0)}]
        # The shared collator stacks pairs only; longer tuples stay a list.
        self.assertIsInstance(collate_fn_dict_with_padding(batch)["notes"], list)
        untokenized = self._text_processor(tokenized=False).collate
        collated = collate_fn_dict_with_padding(batch, collators={"notes": untokenized})
        self.assertIsInstance(collated["notes"], list)

    def test_build_unified_inputs_consumes_stacked_text(self):
        from types import SimpleNamespace

        from pyhealth.datasets.utils import collate_fn_dict_with_padding
        from pyhealth.models.rnn import RNN
        from pyhealth.processors import StageNetTensorProcessor, TupleTimeTextProcessor

        notes_proc = TupleTimeTextProcessor(
            tokenizer_model="emilyalsentzer/Bio_ClinicalBERT",
            type_tag="note",
            max_length=8,
        )
        labs_proc = StageNetTensorProcessor()
        labs_proc.fit([{"labs": ([0.0], [[1.0, 2.0]])}], "labs")
        dataset = SimpleNamespace(
            input_processors={"notes": notes_proc, "labs": labs_proc}
        )
        batch = collate_fn_dict_with_padding(
            [
                {
                    "notes": notes_proc.process(
                        (["note one", "second note"], [0.0, 1.0])
                    ),
                    "labs": labs_proc.process(([0.0], [[1.0, 2.0]])),
                },
                {
                    "notes": notes_proc.process(([], [])),
                    "labs": labs_proc.process(([0.0, 1.0], [[3.0, 4.0], [5.0, 6.0]])),
                },
            ],
            collators={"notes": notes_proc.collate},
        )
        stub = SimpleNamespace(
            feature_keys=["notes", "labs"], dataset=dataset, device="cpu"
        )
        inputs = RNN._build_unified_inputs(stub, batch)
        # Tokenizer pads to the longest note in the batch (4 tokens here).
        self.assertEqual(tuple(inputs["notes"]["value"].shape), (2, 2, 4))
        self.assertEqual(tuple(inputs["notes"]["mask"].shape), (2, 2, 4))
        self.assertEqual(tuple(inputs["notes"]["time"].shape), (2, 2))
        # Text is stacked by its processor, which emits no pad mask: a
        # padded note slot is the one with an all-zero attention mask.
        self.assertNotIn("pad_mask", inputs["notes"])
        self.assertEqual(
            inputs["notes"]["mask"].any(dim=-1).tolist(),
            [[True, True], [False, False]],
        )
        self.assertEqual(tuple(inputs["labs"]["value"].shape), (2, 2, 2))

    def test_two_tuple_contract_unchanged(self):
        from pyhealth.datasets.utils import PAD_MASK_SUFFIX, collate_fn_dict_with_padding

        same = [
            {"x": (torch.tensor([1.0, 2.0]), torch.tensor([[1.0], [2.0]]))},
            {"x": (torch.tensor([3.0, 4.0]), torch.tensor([[3.0], [4.0]]))},
        ]
        collated = collate_fn_dict_with_padding(same)
        self.assertEqual(tuple(collated["x"][0].shape), (2, 2))
        self.assertEqual(tuple(collated["x"][1].shape), (2, 2, 1))
        self.assertNotIn(f"x{PAD_MASK_SUFFIX}", collated)

        ragged = [
            {"x": (torch.tensor([1.0, 2.0]), torch.tensor([[1.0], [2.0]]))},
            {"x": (torch.tensor([3.0]), torch.tensor([[3.0]]))},
        ]
        collated = collate_fn_dict_with_padding(ragged)
        self.assertEqual(tuple(collated["x"][1].shape), (2, 2, 1))
        self.assertEqual(
            collated[f"x{PAD_MASK_SUFFIX}"].tolist(), [[True, True], [True, False]]
        )


if __name__ == "__main__":
    unittest.main()
