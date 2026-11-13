import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RETAIN, MultimodalRETAIN, RETAINLayer


class TestRETAIN(unittest.TestCase):
    """Test cases for the RETAIN model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["A", "B"], ["C", "D", "E"]],
                "procedures": [["P1"], ["P2", "P3"]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["F"], ["G", "H"]],
                "procedures": [["P4", "P5"], ["P6"]],
                "label": 0,
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = RETAIN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the RETAIN model initializes correctly."""
        self.assertIsInstance(self.model, RETAIN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_keys[0], "label")

    def test_forward_input_format(self):
        """Test that the dataloader provides tensor inputs."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertIsInstance(data_batch["conditions"], torch.Tensor)
        self.assertIsInstance(data_batch["procedures"], torch.Tensor)
        self.assertIsInstance(data_batch["label"], torch.Tensor)

    def test_model_forward(self):
        """Test that the RETAIN model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the RETAIN model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_loss_is_finite(self):
        """Test that the loss is finite."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        batch_size = 2
        num_labels = 1  # binary classification

        self.assertEqual(ret["y_prob"].shape, (batch_size, num_labels))
        self.assertEqual(ret["y_true"].shape, (batch_size, num_labels))
        self.assertEqual(ret["logit"].shape, (batch_size, num_labels))

    def test_model_with_embedding(self):
        """Test that the RETAIN model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        expected_embed_dim = len(self.model.feature_keys) * self.model.embedding_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)


MODELS = [RETAIN, MultimodalRETAIN]

DEEP_SAMPLES = [
    {"patient_id": "p0", "label": 0,
     "codes": [[["a", "b"], ["c"]], [["a"]], [["d", "e"], ["b"], ["c"]]]},
    {"patient_id": "p1", "label": 1, "codes": [[["b"]]]},
]


class TestDeepNestedCodes(unittest.TestCase):
    """deep_nested_sequence fields embed to (batch, groups, visits, codes, D)."""

    def test_models_pool_each_group_and_ignore_padding(self):
        dataset = create_sample_dataset(
            samples=DEEP_SAMPLES, input_schema={"codes": "deep_nested_sequence"},
            output_schema={"label": "binary"}, dataset_name="deep",
        )
        for cls in MODELS:
            with self.subTest(model=cls.__name__):
                torch.manual_seed(0)
                model = cls(dataset=dataset, embedding_dim=8)
                batch = next(iter(get_dataloader(dataset, batch_size=2)))
                ret = model(**batch)
                ret["loss"].backward()
                self.assertEqual(tuple(ret["y_prob"].shape), (2, 1))
                self.assertTrue(torch.isfinite(ret["loss"]))
                # The short record scores the same alone as when padded next
                # to a longer one.
                model.eval()
                with torch.no_grad():
                    together = model(**batch)["y_prob"][1]
                    single = next(iter(get_dataloader(dataset.subset([1]),
                                                      batch_size=1)))
                    alone = model(**single)["y_prob"][0]
                torch.testing.assert_close(together, alone)


class TestRETAINAttention(unittest.TestCase):
    """RETAINLayer's attention lines up with the visits and ignores padding."""

    def test_attention_lines_up_with_the_visits(self):
        # The GRUs read the visits from last to first, so the output for the
        # last visit has seen only that visit. Changing the first visit must
        # not change it, and must change the first visit's output.
        torch.manual_seed(0)
        layer = RETAINLayer(4, dropout=0.0).eval()
        x = torch.randn(1, 3, 4)
        lengths = torch.tensor([3])
        changed = x.clone()
        changed[0, 0] += 1.0
        with torch.no_grad():
            beta = layer.compute_beta(layer.reverse_x(x, lengths), lengths, 3)
            beta2 = layer.compute_beta(layer.reverse_x(changed, lengths), lengths, 3)
        torch.testing.assert_close(beta[0, -1], beta2[0, -1])
        self.assertFalse(torch.allclose(beta[0, 0], beta2[0, 0]))

    def test_padded_steps_get_no_attention(self):
        torch.manual_seed(0)
        layer = RETAINLayer(4, dropout=0.0).eval()
        x = torch.randn(2, 3, 4)
        lengths = torch.tensor([3, 1])
        with torch.no_grad():
            alpha = layer.compute_alpha(layer.reverse_x(x, lengths), lengths, 3)
        self.assertEqual(float(alpha[1, 1:].sum()), 0.0)
        torch.testing.assert_close(alpha.sum(dim=1).squeeze(-1), torch.ones(2))

    def test_score_does_not_depend_on_the_batch(self):
        # A short record scores the same alone as next to a longer one.
        for key, codes in [("sequence", (["a", "b", "c", "d"], ["b"])),
                           ("nested_sequence", ([["a", "b"], ["c"], ["d"]], [["b"]]))]:
            samples = [{"patient_id": "p0", "label": 0, "codes": codes[0]},
                       {"patient_id": "p1", "label": 1, "codes": codes[1]}]
            dataset = create_sample_dataset(
                samples=samples, input_schema={"codes": key},
                output_schema={"label": "binary"}, dataset_name=key,
            )
            torch.manual_seed(0)
            model = RETAIN(dataset=dataset, embedding_dim=8).eval()
            with torch.no_grad(), self.subTest(input=key):
                batch = next(iter(get_dataloader(dataset, batch_size=2)))
                single = next(iter(get_dataloader(dataset.subset([1]), batch_size=1)))
                torch.testing.assert_close(model(**batch)["y_prob"][1],
                                           model(**single)["y_prob"][0])


if __name__ == "__main__":
    unittest.main()
