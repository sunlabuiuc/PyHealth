from typing import Dict, Any
from pyhealth.tasks import BaseTask


class CXRMultimodalTask(BaseTask):
    """
    Multimodal task for chest X-ray data combining image and text features.

    This task assumes that the dataset returns a dictionary for each sample with:
    - 'txt': tokenized text report (Tensor)
    - 'img1', 'img2', ..., 'imgN': image tokens (Tensor)
    - 'modes': list of input modes like ['txt', 'img1', ...]
    """

    def __init__(self, dataset):
        """
        Initializes the multimodal task with the dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset instance (e.g., UnifiedCXRDataset).
        """
        super().__init__(dataset=dataset)

    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a raw item from the dataset into model-ready inputs.

        Args:
            raw_data (Dict[str, Any]): Raw output from the dataset __getitem__.

        Returns:
            Dict[str, Any]: A processed dictionary containing:
                - 'text': tokenized report (Tensor)
                - 'images': list of image token Tensors
        """
        text = raw_data["txt"]
        images = [raw_data[key] for key in raw_data if key.startswith("img")]
        return {
            "text": text,
            "images": images,
        }
