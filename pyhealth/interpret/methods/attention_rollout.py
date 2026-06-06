from typing import Dict, Optional

import torch

from pyhealth.models.base_model import BaseModel
from .base_interpreter import BaseInterpreter

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------

class AttentionRollout(BaseInterpreter):
    def __init__(self, model: BaseModel, head_fusion="mean"):
        super().__init__()

        required_methods = [
            "set_attention_hooks",
            "get_attention_layers",
            "get_relevance_tensor",
        ]
        missing_methods = [
            method for method in required_methods if not hasattr(model, method)
        ]

        if missing_methods:
            raise TypeError(
                "AttentionRollout requires a model that exposes the attention "
                "interpretability methods: "
                f"{', '.join(required_methods)}. "
                f"Missing: {', '.join(missing_methods)}."
            )

        self.model = model
        self.head_fusion = head_fusion

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        pass