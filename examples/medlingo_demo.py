from pyhealth.datasets.medlingo import MedLingoDataset

samples = [
    {
        "abbr": "SOB",
        "context": "Patient presents with SOB.",
        "label": "shortness of breath",
        "source": "synthetic_demo",
    }
]

dataset = MedLingoDataset(samples=samples)
records = dataset.process()

print("Dataset loaded")
print(records)