from pyhealth.datasets.synthetic_hirid_dataset import SyntheticHiRIDDataset
from torch.utils.data import DataLoader

dataset = SyntheticHiRIDDataset(data_path="data/synthetic_hirid")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

for x, y in loader:
    print("Batch x shape:", x.shape)  # [B, C, T]
    print("Batch y:", y)
    break

