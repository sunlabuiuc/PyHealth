from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .base_task import BaseTask


class MelanomaArtifactClassification(BaseTask):
    """Task for binary melanoma classification with simple image perturbations.

    Supports different modes to simulate the paper's idea of testing model
    robustness under different image conditions.

    Modes:
        - "whole": original image
        - "background": zero out center (simulate removing lesion)
        - "low_freq": apply blur (simulate low-frequency emphasis)
    """

    task_name: str = "MelanomaArtifactClassification"
    input_schema: Dict[str, str] = {
        "image": "image",
        "mode": "string",
    }
    output_schema: Dict[str, str] = {
        "label": "binary",
    }

    def __init__(self, mode: str = "whole"):
        self.mode = mode

    def _apply_mode(self, image: torch.Tensor) -> torch.Tensor:
        """Apply simple transformation based on mode."""

        if self.mode == "whole":
            return image

        if self.mode == "background":
            """Zero out center region (simulate removing lesion)"""
            img = image.clone()
            _, H, W = img.shape
            h_start, h_end = H // 4, 3 * H // 4
            w_start, w_end = W // 4, 3 * W // 4
            img[:, h_start:h_end, w_start:w_end] = 0
            return img

        if self.mode == "low_freq":
            """Apply simple blur using average pooling"""
            return F.avg_pool2d(image.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

        return image

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        image = getattr(patient, "image", None)
        label = getattr(patient, "label", None)
        patient_id = getattr(patient, "patient_id", None)

        if image is None or label is None or patient_id is None:
            return []

        """Apply transformation"""
        if isinstance(image, torch.Tensor):
            image = self._apply_mode(image)

        return [
            {
                "patient_id": patient_id,
                "image": image,
                "mode": self.mode,
                "label": int(label),
            }
        ]