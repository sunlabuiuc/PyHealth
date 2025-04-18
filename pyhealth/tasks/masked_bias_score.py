# ------------------------------------------------------------------------------
# Contribution Header
#
# Name: 
#   Zilal Eiz Al Din
#
# NetID: 
#   zelalae2@illinois.edu
#
# Paper Title: 
#   Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings
#
# Paper Link:
#   https://arxiv.org/pdf/2003.11515
#
# This contribution implements the MaskedBiasScoreTask, a task implements masked token prediction evaluation to quantify gender bias
# in masked language models applied to clinical notes.

# Given clinical notes with gendered words masked (e.g., "he", "she", "male", "female" replaced with [MASK]), 
# this task computes the model's log-probability of predicting "he" versus "she" at each masked position.

# For each input sample (a masked clinical note), the task outputs:
# - The log-probability assigned to "he" (male_log_prob)
# - The log-probability assigned to "she" (female_log_prob)
# - The bias score, computed as (male_log_prob - female_log_prob)

# Inputs:
# - `masked_text` (text): Clinical note with gendered words masked with [MASK].

# Outputs:
# - `male_log_prob` (regression): Log probability assigned to "he" at the masked position.
# - `female_log_prob` (regression): Log probability assigned to "she" at the masked position.
# - `bias_score` (regression): Difference between male and female log-probabilities.

# This task supports masked language model evaluation for reproducibility of experiments 
# described in "Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings" (2020).

# Example Usage:
#     >>> task = MaskedBiasScoreTask()
#     >>> sample_dataset = dataset.set_task(task)
#     >>> print(sample_dataset.samples[0])  # print one sample
#------------------------------------------------------------------------------
from dataclasses import dataclass, field
from pyhealth.data.data import Patient
from typing import Dict, List

from pyhealth.tasks import BaseTask
from pyhealth.data import Patient
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskedBiasScoreTask(BaseTask):
    logger.info("Task Started!")
    """Task to compute male/female log-probability difference from masked notes.

    Args:
        task_name (str): Name of the task.
        input_schema (Dict): Expected input fields from the patient data.
        output_schema (Dict): Output fields after processing.

    Example:
        >>> from pyhealth.tasks import MaskedBiasScoreTask
        >>> task = MaskedBiasScoreTask()
        >>> print(task)
    """

    task_name: str = "masked_bias_score"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"masked_text": "text"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {
        "male_log_prob": "regression",
        "female_log_prob": "regression",
        "bias_score": "regression",
    })

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes one patient to calculate male/female log-probability difference.

        Args:
            patient (Patient): A PyHealth Patient object containing masked text.

        Returns:
            List[Dict]: A list of one or more dictionaries containing:
                - patient_id
                - visit_id
                - male_log_prob
                - female_log_prob
                - bias_score
        """
        samples = []

        # Load HuggingFace model once (can optimize if needed)
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()

        male_token = "he"
        female_token = "she"
        

        for event in patient.get_events(event_type="noteevents"):
            visit_id = event.attr_dict.get("visit_id")
            text = event.attr_dict.get("masked_text")
            if not text or "[MASK]" not in text:
                continue

            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Find [MASK] token position
            mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_token_index) == 0:
                continue

            mask_logits = logits[0, mask_token_index, :]
            log_probs = torch.nn.functional.log_softmax(mask_logits, dim=-1)

            male_id = tokenizer.convert_tokens_to_ids(male_token)
            female_id = tokenizer.convert_tokens_to_ids(female_token)

            male_log_prob = log_probs[0, male_id].item()
            female_log_prob = log_probs[0, female_id].item()
            bias_score = male_log_prob - female_log_prob

            sample = {
                "patient_id": patient.patient_id,
                "timestamp": event.timestamp,
                "male_log_prob": male_log_prob,
                "female_log_prob": female_log_prob,
                "bias_score": bias_score,
                "masked_text": text
            }
            samples.append(sample)

        return samples


if __name__ == "__main__":
    task = MaskedBiasScoreTask()
    print(task)
    print(type(task))
