import torch
from typing import Dict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from pyhealth.models.base_model import BaseModel


class SleepQABioBERT(BaseModel):
    """BioBERT Reader for Extractive Question Answering.

    This model uses a transformer-based architecture to predict the 
    start and end logits of an answer within a clinical context.

    Args:
        dataset: the sample dataset used for vocabulary/label initialization.
        model_name: HuggingFace model checkpoint. Default is BioBERT.
        **kwargs: additional parameters for BaseModel.

    Examples:
        >>> from pyhealth.models import SleepQABioBERT
        >>> model = SleepQABioBERT(dataset=samples)
        >>> outputs = model(**batch)
    """

    def __init__(self, dataset, model_name="dmis-lab/biobert-base-cased-v1.1-squad", **kwargs):
        super(SleepQABioBERT, self).__init__(dataset=dataset, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForQuestionAnswering.from_pretrained(
            model_name)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: dictionary containing 'passage' and 'question' strings.

        Returns:
            A dictionary containing start_logits, end_logits, and loss.
        """
        passages, questions = kwargs.get("passage"), kwargs.get("question")
        encodings = self.tokenizer(
            questions, passages, padding=True, truncation=True, return_tensors="pt")

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask)
        return {
            "start_logits": outputs.start_logits,
            "end_logits": outputs.end_logits,
            "logit": torch.stack([outputs.start_logits, outputs.end_logits], dim=-1),
            "loss": torch.tensor(0.0, requires_grad=True).to(self.device)
        }
