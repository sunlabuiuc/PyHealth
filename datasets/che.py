from pyhealth.datasets import BaseImageDataset
import os
import pandas as pd

class CheXphotoDataset(BaseImageDataset):
    def __init__(self, root: str, csv_file: str = "labels.csv", transform=None):
        super().__init__(dataset_name="CheXphoto", root=root)
        self.transform = transform
        self.csv_file = csv_file
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_csv(os.path.join(self.root, self.csv_file))
        samples = []
        for _, row in df.iterrows():
            image_path = os.path.join(self.root, row["image_path"])
            label = [row[p] for p in self.pathologies]  # pathologies = list of 14 labels
            samples.append({"image": image_path, "label": label})
        return samples
