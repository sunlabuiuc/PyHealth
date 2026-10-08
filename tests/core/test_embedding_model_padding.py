import unittest

import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import EmbeddingModel


class TestEmbeddingModelPadding(unittest.TestCase):
    def setUp(self):
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["cond-1", "cond-2"], ["cond-3"]],
                "deep_codes": [[["deep-1"], ["deep-2", "deep-3"]]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["cond-4"]],
                "deep_codes": [[["deep-4"]]],
                "label": 0,
            },
        ]
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions": "nested_sequence",
                "deep_codes": "deep_nested_sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="embedding-padding-test",
        )

    def test_nested_sequence_embeddings_use_zero_padding(self):
        model = EmbeddingModel(self.dataset, embedding_dim=8)

        for field in ["conditions", "deep_codes"]:
            embedding = model.embedding_layers[field]
            self.assertEqual(embedding.padding_idx, 0)
            self.assertTrue(torch.equal(embedding.weight[0], torch.zeros(8)))

    def test_nested_sequence_padding_row_does_not_receive_gradients(self):
        model = EmbeddingModel(self.dataset, embedding_dim=8)
        embedding = model.embedding_layers["conditions"]
        token_index = self.dataset.input_processors["conditions"].code_vocab["cond-1"]

        output = embedding(torch.tensor([[[0, token_index]]]))
        output.sum().backward()

        self.assertTrue(torch.equal(embedding.weight.grad[0], torch.zeros(8)))
        self.assertGreater(embedding.weight.grad[token_index].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
