import unittest
import torch

from pyhealth.models import KEEP


class DummyProcessor:
    """Minimal processor with code_vocab_size."""

    def __init__(self, vocab_size: int):
        self.code_vocab_size = vocab_size


class DummyLabelProcessor:
    def get_output_size(self):
        return 1


class DummyInputProcessor:
    def __init__(self, vocab_size):
        self.code_vocab_size = vocab_size


class DummyDataset:
    def __init__(self, vocab_size=20):
        self.vocab_size = vocab_size

        # Required by BaseModel
        self.input_schema = {"conditions": "sequence"}
        self.output_schema = {"label": "binary"}

        self.input_processors = {
            "conditions": DummyInputProcessor(vocab_size)
        }

        self.label_processors = {
            "label": DummyLabelProcessor()
        }


class TestKEEP(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dataset = DummyDataset(vocab_size=20)

    def test_instantiation(self):
        model = KEEP(
            dataset=self.dataset,
            embedding_dim=8,
            lambda_base=0.1,
        )
        self.assertIsInstance(model, KEEP)

    def test_forward_pass(self):
        model = KEEP(
            dataset=self.dataset,
            embedding_dim=8,
            lambda_base=0.1,
        )

        # batch_size=2, seq_len=5
        conditions = torch.randint(0, 20, (2, 5))

        output = model(conditions=conditions)

        self.assertIn("y_prob", output)
        self.assertEqual(output["y_prob"].shape, (2,))

    def test_gradient_computation(self):
        model = KEEP(
            dataset=self.dataset,
            embedding_dim=8,
            lambda_base=0.1,
        )

        conditions = torch.randint(0, 20, (2, 5))
        labels = torch.tensor([1.0, 0.0])

        output = model(
            conditions=conditions,
            label=labels,
        )

        loss = output["loss"]
        loss.backward()

        # Check at least one parameter received gradient
        grads = [
            param.grad
            for param in model.parameters()
            if param.requires_grad and param.grad is not None
        ]

        self.assertTrue(len(grads) > 0)