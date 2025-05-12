# Name(s): Abhi Chebiyam, Raj Joshi
# NetID(s): abhijit3, rajj3
# Paper Title: N/A
# Paper Link: N/A
# Description: Implements a GRU-based sequence classification model for binary mortality prediction.
# Includes a dummy dataset, task setup, model class, test case, and main function in a single self-contained script.

import torch
import torch.nn as nn
from typing import Dict, List

# ----- Dummy Dataset -----
class DummyMortalityDataset:
    def __init__(self, split="train"):
        self.output_size = 2  # binary classification
        self.split = split
        self.data = self._generate_dummy_data()

    def _generate_dummy_data(self):
        return [
            {"lab": torch.randn(10, 128), "labels": torch.tensor(0)},
            {"lab": torch.randn(10, 128), "labels": torch.tensor(1)},
        ]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# ----- Task -----
class MortalityPredictionTask:
    def __init__(self):
        self.task_name = "mortality_prediction"
        self.input_keys = ["lab"]
        self.target_keys = ["labels"]
        self.task_type = "binary"

    def generate_labels(self, patient_data):
        return {"mortality": int(patient_data["status"] == "deceased")}


# ----- Model -----
class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        dataset,
        input_size: int = 128,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super(GRUSequenceClassifier, self).__init__()
        self.output_size = dataset.output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_size, self.output_size)

    def forward(self, lab: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(lab)
        final_hidden = out[:, -1, :]
        logits = self.classifier(final_hidden)
        return logits


# ----- Test Case -----
def test_gru_sequence_classifier():
    dataset = DummyMortalityDataset()
    model = GRUSequenceClassifier(dataset)

    dummy_input = {
        "lab": torch.randn(4, 10, 128)
    }

    output = model(**dummy_input)

    assert output.shape == (4, 2), f"Expected shape (4, 2), got {output.shape}"
    assert isinstance(output, torch.Tensor), "Output must be a tensor"
    print("Test passed!")


# ----- Main Example -----
def main():
    dataset = DummyMortalityDataset()
    model = GRUSequenceClassifier(dataset)
    sample = dataset[0]
    input_tensor = sample["lab"].unsqueeze(0)
    output = model(lab=input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    print(f"Predicted class: {prediction} (logits: {output.detach().numpy()})")


if __name__ == "__main__":
    test_gru_sequence_classifier()
    main()