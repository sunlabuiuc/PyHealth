from pyhealth.datasets import SampleDataset

samples = [
    {
        "id": "sample1",
        "text": "Lungs are clear.",
        "view": "PA",
        "image_tokens": [12, 57, 91, 33, 44]
    },
    {
        "id": "sample2",
        "text": "Mild cardiomegaly.",
        "view": "AP",
        "image_tokens": [11, 52, 19, 6, 71]
    }
]

dataset = SampleDataset(samples=samples)

print("Loaded dataset length:", len(dataset))
print("Example sample:", dataset[0])
