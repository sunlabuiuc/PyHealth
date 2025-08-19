# ------------------------------------------------------------------------------
# Contribution Header
#
# Name:
#   Zilal Eiz Al Din && Payel Chakraborty
#
# NetID:
#   zelalae2 && payelc2
#
# Paper Title:
#   Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings
#
# Paper Link:
#   https://arxiv.org/pdf/2003.11515
#
# This contribution implements FillInTheBlank, a PyHealth-compatible model
# that uses a masked language model (e.g., BERT) to compute the log-probability
# of specific gendered tokens (e.g., "he" and "she") at the [MASK] positions
# in clinical text, and then copute the bias score. It is built to work directly with MIMIC3Dataset.
# ------------------------------------------------------------------------------

import logging
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models import BaseModel


logger = logging.getLogger(__name__)


class FillInTheBlank(BaseModel):
    logger.info("Fill In The Blank Model started")
    """Masked language model to compute log-probabilities at [MASK] positions.

    This model uses a transformer-based masked language model (MLM) to compute
    log-probabilities assigned to gendered terms (e.g., "he" and "she") at
    [MASK] positions in masked clinical notes from the MIMIC-III dataset.

    It also computes a bias score as the difference between logP("he") and
    logP("she").

    Attributes:
        model: Hugging Face masked language model.
        tokenizer: Tokenizer corresponding to the MLM model.
        target_tokens: List of gendered tokens to evaluate (e.g., ["he", "she"]).
        target_ids: Token IDs of the gendered tokens.

    Example:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import HurtfulWordsPreprocessingTask
        >>> from pyhealth.models  import FillInTheBlank

        >>> # Load raw dataset
        >>> mimic3_base = MIMIC3Dataset(
         ...   root=f"{PATH_TO_CSVs}",
         ...   tables=["noteevents"]
        ... )

        >>> # Apply preprocessing task
        >>> task = HurtfulWordsPreprocessingTask()
        >>> sample_dataset = mimic3_base.set_task(task)

        >>> # Initialize model with SampleDataset
        >>> model = FillInTheBlank(sample_dataset)

        >>> # Use the first real masked_text from the dataset
        >>> first_masked_text = sample_dataset.samples[0]["masked_text"]
        >>> print("Using first masked text:", first_masked_text)

        >>> # Run model on this text
        >>> batch = {"masked_text": [first_masked_text]}
        >>> log_probs = model(batch)
      

        >>> # Display results
        >>> he_logp, she_logp = log_probs[0].tolist()
        >>> print("LogP(he), LogP(she):", round(he_logp, 4), round(she_logp, 4))
        >>> print("Bias Score (he - she):", round(bias_score, 4))
    """

    def __init__(self, dataset: MIMIC3Dataset,
                 model_name: str = "bert-base-uncased"):
        """Initializes the MaskedLogProbModel.

        Args:
            dataset (MIMIC3Dataset): The task-applied dataset containing
                'masked_text' as input field.
            model_name (str): Pretrained HuggingFace model name for MLM.
        """
        super().__init__(dataset)
        logger.info(f"Initializing masked language model from {model_name}")
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.target_tokens = ["he", "she"]
        self.target_ids = [
            self.tokenizer.convert_tokens_to_ids(token)
            for token in self.target_tokens
        ]

    def forward(self, batch: dict) -> torch.Tensor:
        """Computes logP(he), logP(she), and bias score for each [MASK].

        Args:
            batch (dict): A dictionary with "masked_text" key.

        Returns:
            dict: A dictionary with keys:
                - "log_probs": tensor of shape (N, 2) with logP(he), logP(she)
                - "bias_scores": tensor of shape (N,) with (logP(he) - logP(she))
        """
        texts = batch["masked_text"]

        # Tokenize batch of text inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Run model in no-grad mode
        with torch.no_grad():
            logits = self.model(**inputs).logits  # Shape: (B, T, V)

        # Find indices of [MASK] tokens
        mask_indices = inputs.input_ids == self.tokenizer.mask_token_id
        mask_logits = logits[mask_indices]  # Shape: (num_masks, V)

        # Convert to log-probabilities
        log_probs = torch.nn.functional.log_softmax(mask_logits, dim=-1)

        # Extract log-probs for target tokens
        stacked_log_probs = torch.stack(
            [log_probs[:, tid] for tid in self.target_ids],
            dim=1  # Shape: (num_masks, 2)
        )
        # Compute bias score: logP("he") - logP("she")
        bias_scores = stacked_log_probs[:, 0] - stacked_log_probs[:, 1]

        # Stack log-probs for each target token
        result = torch.stack(
            [log_probs[:, token_id] for token_id in self.target_ids],
            dim=1  # Shape: (num_masks, 2)
        )

        return result
