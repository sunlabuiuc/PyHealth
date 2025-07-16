import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models.base_model import BaseModel

# Import the HALO transformer implementation
from pyhealth.models.generators.halo_resources.halo_model import HALOModel
from pyhealth.models.generators.halo_resources.halo_model import HALOConfig

class HALO(BaseModel):
    """
    HALO model wrapper for PyHealth, leveraging the BaseModel interface.

    Args:
        dataset: SampleEHRDataset for EHR sequences (visits of codes).
        config: Transformer configuration object for HALOModel.
        feature_key: key in dataset samples for code-based visits (e.g., "list_codes").
        mode: one of "binary", "multiclass", or "multilabel" for loss function.
        label_key: key in dataset samples for target visits; defaults to feature_key for next-visit prediction.
        pos_loss_weight: weight applied to positive labels in BCE loss.
    """
    def __init__(
        self,
        dataset: SampleEHRDataset,
        config: HALOConfig,
        feature_key: str = "list_codes",
        mode: str = "multilabel",
        label_key: Optional[str] = None,
        pos_loss_weight: float = 1.0,
    ):
        super(HALO, self).__init__(dataset)
        self.feature_key = feature_key
        self.label_key = label_key or feature_key
        self.pos_loss_weight = pos_loss_weight

        # Set mode for loss and evaluation
        self.mode = mode

        # Tokenizer for the code-based input
        self.tokenizer = dataset.input_processors[feature_key]
        self.vocab_size = self.tokenizer.size()

        # Instantiate the underlying HALO transformer model
        self.halo = HALOModel(config)

    def _prepare_input_visits(self, codes: List[List[Any]]) -> torch.Tensor:
        """
        Convert list of visits of codes into multi-hot tensor.

        Args:
            codes: nested list of shape (batch, num_visits, codes_in_visit)

        Returns:
            Tensor of shape (batch, num_visits, vocab_size) with 0/1 entries.
        """
        # batch_encode_3d returns List[List[List[int]]]
        token_ids = self.tokenizer.batch_encode_3d(codes)
        batch_size = len(token_ids)
        max_visits = len(token_ids[0])

        visits = torch.zeros(
            batch_size, max_visits, self.vocab_size, device=self.device
        )
        for i in range(batch_size):
            for t, visit_ids in enumerate(token_ids[i]):
                for cid in visit_ids:
                    if cid is None:
                        continue
                    visits[i, t, cid] = 1.0
        return visits

    def forward(
        self,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Forward propagation for HALO within PyHealth.

        Expects kwargs to contain:
          - feature_key: raw visit code lists.
          - optional label_key: same format for next-visit targets.
          - optional masks: tensor or nested lists for masking visits.

        Returns a dict with keys:
          - loss: training loss (if labels provided).
          - y_prob: predicted probabilities of codes per visit.
          - y_true: ground-truth multi-hot labels (shifted by one visit).
          - logits: raw logits from the HALO transformer.
        """
        # Prepare input tensor
        raw_codes = kwargs[self.feature_key]
        input_visits = self._prepare_input_visits(raw_codes)

        # Gather optional training labels and masks
        ehr_labels = None
        ehr_masks = None
        if self.label_key in kwargs:
            # similarly convert label visits to multi-hot
            ehr_labels = self._prepare_input_visits(kwargs[self.label_key])
        if "masks" in kwargs:
            ehr_masks = torch.tensor(kwargs["masks"], device=self.device, dtype=torch.float)

        # Call HALOModel: returns loss & probabilities if labels, else probabilities
        if ehr_labels is not None:
            loss, code_probs, shift_labels = self.halo(
                input_visits,
                ehr_labels=ehr_labels,
                ehr_masks=ehr_masks,
                pos_loss_weight=self.pos_loss_weight,
            )
            results = {"loss": loss, "y_prob": code_probs, "y_true": shift_labels}
        else:
            code_probs = self.halo(input_visits)
            results = {"y_prob": code_probs}

        # Attach logits if needed
        if hasattr(self.halo, 'last_logits'):
            results['logits'] = self.halo.last_logits

        return results

# Example usage:
if __name__ == "__main__":
    from pyhealth.datasets import SampleEHRDataset
    # define samples with visits of codes as nested lists
    samples = [
        {"patient_id": "p0", "visit_id": "v0", "list_codes": [["A", "B"], ["C"]]},
        {"patient_id": "p1", "visit_id": "v0", "list_codes": [["D"], ["E", "F"]]},
    ]
    dataset = SampleEHRDataset(samples=samples, dataset_name="halo_test")

    # Build transformer config
    config = HALOConfig(
        n_layer=4,
        n_head=8,
        n_embd=128,
        total_vocab_size=dataset.input_processors['list_codes'].size(),
        n_positions=dataset.max_visit_length,
        n_ctx=dataset.max_visit_length,
        layer_norm_epsilon=1e-5,
    )

    model = HALO(
        dataset=dataset,
        config=config,
        feature_key='list_codes',
        mode='multilabel',
    )
    from pyhealth.datasets import get_dataloader
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    output = model(**batch)
    print(output)
