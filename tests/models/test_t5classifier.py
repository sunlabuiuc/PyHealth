"""Fast unit tests for T5Classifier (Hugging Face weights mocked; no network)."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models.t5classifier import T5Classifier


def _tiny_multilabel_samples():
    return [
        {
            "patient_id": "p0",
            "text": "control sentence",
            "labels": ["none"],
        },
        {
            "patient_id": "p1",
            "text": "tumor proliferation pathway activation",
            "labels": ["none", "sustaining proliferative signaling"],
        },
    ]


class TestT5ClassifierForward(unittest.TestCase):
    def setUp(self):
        self.sample_ds = create_sample_dataset(
            samples=_tiny_multilabel_samples(),
            input_schema={"text": "text"},
            output_schema={"labels": "multilabel"},
            dataset_name="test_hoc",
            task_name="TestMultilabel",
            in_memory=True,
        )

    @patch("pyhealth.models.t5classifier.AutoTokenizer.from_pretrained")
    @patch("pyhealth.models.t5classifier.T5EncoderModel.from_pretrained")
    def test_forward_logits_and_loss(self, mock_enc_from_pretrained, mock_tok_from_pretrained):
        d_model = 32
        seq_len = 5
        batch = 2

        enc = MagicMock()
        enc.config = SimpleNamespace(d_model=d_model)

        def enc_forward(**kwargs):
            out = MagicMock()
            out.last_hidden_state = torch.randn(batch, seq_len, d_model)
            return out

        enc.side_effect = enc_forward
        mock_enc_from_pretrained.return_value = enc

        tok = MagicMock()
        mock_tok_from_pretrained.return_value = tok

        def encode_texts(*_args, **_kwargs):
            return {
                "input_ids": torch.ones(batch, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
            }

        tok.side_effect = encode_texts

        model = T5Classifier(
            dataset=self.sample_ds,
            pretrained_model_name="t5-small",
            max_length=32,
        )

        b0 = self.sample_ds[0]
        b1 = self.sample_ds[1]
        batch_dict = {
            "text": [b0["text"], b1["text"]],
            "labels": torch.stack([b0["labels"], b1["labels"]]),
        }

        out = model(**batch_dict)
        self.assertIn("logit", out)
        self.assertIn("loss", out)
        self.assertEqual(out["logit"].shape[0], batch)
        self.assertEqual(out["logit"].shape[1], model.get_output_size())


if __name__ == "__main__":
    unittest.main()
