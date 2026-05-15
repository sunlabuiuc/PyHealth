import torch
import torch.nn as nn
import torchvision.models as models

from pyhealth.tasks import MelanomaArtifactClassification


class FakePatient:
    def __init__(self, patient_id, image, label):
        self.patient_id = patient_id
        self.image = image
        self.label = label


def run_example(mode):
    print(f"\nRunning mode: {mode}")

    task = MelanomaArtifactClassification(mode=mode)

    patients = [
        FakePatient("p1", torch.randn(3, 224, 224), 1),
        FakePatient("p2", torch.randn(3, 224, 224), 0),
    ]

    samples = []
    for p in patients:
        samples.extend(task(p))

    X = torch.stack([s["image"] for s in samples])
    y = torch.tensor([s["label"] for s in samples])

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    outputs = model(X).squeeze()
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(outputs, y.float())

    print("Output shape:", outputs.shape)
    print("Loss:", loss.item())


if __name__ == "__main__":
    run_example("whole")
    run_example("background")