"""
Author(s): Nandan Tadi (ntadi2), Ayan Deka (ayand2)
Title: Reproducing “Appropriate Evaluation of Diagnostic Utility of Machine Learning Algorithm-Generated Images”
Source: https://proceedings.mlr.press/v136/kwon20a.html
Description: Dataset wrapper for GAN-generated synthetic X-ray images.
"""

from pyhealth.datasets import BaseDataset
from typing import List
import os

class XRayGANDataset(BaseDataset):
    def __init__(self, root: str):
        super().__init__()
        self.image_dir = root

    def load_data(self) -> List[str]:
        """Loads image paths from the dataset directory.

        Args:
            None

        Returns:
            List[str]: List of image paths.
        """
        # Example: load from pre-generated images
        image_paths = []
        for label in ["positive", "negative"]:
            folder = os.path.join(self.image_dir, label)
            for file in os.listdir(folder):
                if file.endswith(".png"):
                    image_paths.append((os.path.join(folder, file), label))
        return image_paths
