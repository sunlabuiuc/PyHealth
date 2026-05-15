"""Fast unit tests for the local seq2seq T5 model (no network)."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models.t5 import T5


def _tiny_seq2seq_samples():
    return [
        {
            "patient_id": "p0",
            "source_text": "hoc: control sentence",
            "target_text": "none",
        },
        {
            "patient_id": "p1",
            "source_text": "hoc: tumor proliferation pathway activation",
            "target_text": "sustaining proliferative signaling",
        },
    ]


class TestT5Forward(unittest.TestCase):
    def setUp(self):
        self.sample_ds = create_sample_dataset(
            samples=_tiny_seq2seq_samples(),
            input_schema={"source_text": "text"},
            output_schema={"target_text": "text"},
            dataset_name="test_hoc",
            task_name="TestSeq2Seq",
            in_memory=True,
        )

    @patch("pyhealth.models.t5.AutoTokenizer.from_pretrained")
    @patch("pyhealth.models.t5.T5ForConditionalGeneration.from_pretrained")
    def test_forward_loss_and_generate(
        self,
        mock_model_from_pretrained,
        mock_tok_from_pretrained,
    ):
        seq_len = 5
        batch = 2

        hf_model = MagicMock()
        hf_model.config = SimpleNamespace(d_model=32)

        def model_forward(**kwargs):
            out = MagicMock()
            out.loss = torch.tensor(0.25, requires_grad=True)
            out.logits = torch.randn(batch, seq_len, 8)
            return out

        hf_model.side_effect = model_forward
        hf_model.generate.return_value = torch.tensor([[1, 2, 3], [4, 5, 0]])
        mock_model_from_pretrained.return_value = hf_model

        tok = MagicMock()
        mock_tok_from_pretrained.return_value = tok

        def encode_texts(*_args, **_kwargs):
            return {
                "input_ids": torch.ones(batch, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
            }

        tok.side_effect = encode_texts
        tok.pad_token_id = 0
        tok.batch_decode.return_value = ["none", "sustaining proliferative signaling"]

        model = T5(
            dataset=self.sample_ds,
            pretrained_model_name="t5-small",
            max_source_length=32,
            max_target_length=16,
        )

        b0 = self.sample_ds[0]
        b1 = self.sample_ds[1]
        batch_dict = {
            "source_text": [b0["source_text"], b1["source_text"]],
            "target_text": [b0["target_text"], b1["target_text"]],
        }

        out = model(**batch_dict)
        self.assertIn("loss", out)
        generated = model.generate_text(batch_dict["source_text"])
        self.assertEqual(
            generated,
            ["none", "sustaining proliferative signaling"],
        )


if __name__ == "__main__":
    unittest.main()
