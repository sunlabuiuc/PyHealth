"""
PyHealth model for transformer-based clinical text de-identification.

Performs token-level NER to detect PHI (protected health information)
in clinical notes using a pre-trained transformer with a classification
head.

Paper: Johnson, Alistair E.W., et al. "Deidentification of free-text
    medical records using pre-trained bidirectional transformers."
    Proceedings of the ACM Conference on Health, Inference, and
    Learning (CHIL), 2020.

Paper link:
    https://doi.org/10.1145/3368555.3384455

Model structure (dropout + linear head) follows PyHealth's
TransformersModel (pyhealth/models/transformers_model.py), adapted
for token-level classification instead of sequence-level.

Subword alignment follows the standard HuggingFace token
classification pattern (see BertForTokenClassification).

Author:
    Matt McKenna (mtm16@illinois.edu)
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ..datasets import SampleDataset
from .base_model import BaseModel

logger = logging.getLogger(__name__)

# 7 PHI categories with BIO prefix, plus O for non-PHI.
LABEL_VOCAB = {
    "O": 0,
    "B-AGE": 1, "I-AGE": 2,
    "B-CONTACT": 3, "I-CONTACT": 4,
    "B-DATE": 5, "I-DATE": 6,
    "B-ID": 7, "I-ID": 8,
    "B-LOCATION": 9, "I-LOCATION": 10,
    "B-NAME": 11, "I-NAME": 12,
    "B-PROFESSION": 13, "I-PROFESSION": 14,
}

# Cross-entropy ignores positions with this index (PyTorch convention).
IGNORE_INDEX = -100


def align_labels(
    word_ids: List[int | None],
    word_labels: List[int],
) -> List[int]:
    """Align word-level labels to subword tokens.

    BERT/RoBERTa tokenizers split words into subwords. For example,
    "Smith" might become ["Sm", "##ith"]. This function assigns the
    word's label to the first subtoken and IGNORE_INDEX to the rest,
    so the loss function skips non-first subtokens. Special tokens
    ([CLS], [SEP], padding) have word_id=None and also get
    IGNORE_INDEX.

    Args:
        word_ids: Output of tokenizer.word_ids(). None for special
            tokens, integer word index for real tokens.
        word_labels: Label index for each word in the original text.

    Returns:
        List of label indices, one per subtoken. Non-first subtokens
        and special tokens are set to IGNORE_INDEX (-100).
    """
    aligned = []
    prev_word_id = None
    for word_id in word_ids:
        if word_id is None:
            # Special token ([CLS], [SEP], padding).
            aligned.append(IGNORE_INDEX)
        elif word_id != prev_word_id:
            # First subtoken of a word: use the word's label.
            aligned.append(word_labels[word_id])
        else:
            # Non-first subtoken: ignore during loss computation.
            aligned.append(IGNORE_INDEX)
        prev_word_id = word_id
    return aligned


class TransformerDeID(BaseModel):
    """Transformer-based token classifier for clinical text de-identification.

    Uses a pre-trained transformer encoder with a linear classification
    head to predict BIO-tagged PHI labels for each token.

    Args:
        dataset: A SampleDataset from set_task().
        model_name: HuggingFace model name. Default "bert-base-uncased".
        max_length: Maximum token sequence length. Default 512.
        dropout: Dropout rate for the classification head. Default 0.1.

    Examples:
        >>> from pyhealth.datasets import PhysioNetDeIDDataset
        >>> from pyhealth.tasks import DeIDNERTask
        >>> from pyhealth.models import TransformerDeID
        >>> dataset = PhysioNetDeIDDataset(root="/path/to/data")
        >>> samples = dataset.set_task(DeIDNERTask())
        >>> model = TransformerDeID(dataset=samples)  # BERT
        >>> model = TransformerDeID(dataset=samples, model_name="roberta-base")
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super(TransformerDeID, self).__init__(dataset=dataset)

        assert len(self.feature_keys) == 1, (
            "TransformerDeID expects exactly one input feature (text)."
        )
        assert len(self.label_keys) == 1, (
            "TransformerDeID expects exactly one label key."
        )
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]

        self.model_name = model_name
        self.max_length = max_length
        self.label_vocab = LABEL_VOCAB
        self.num_labels = len(LABEL_VOCAB)

        # add_prefix_space=True is required for RoBERTa when using
        # is_split_into_words=True in the forward pass.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: Must contain self.feature_key (list of
                space-joined token strings) and self.label_key
                (list of space-joined BIO label strings).

        Returns:
            Dict with keys: loss, logit, y_prob, y_true.
        """
        texts: List[str] = kwargs[self.feature_key]
        label_strings: List[str] = kwargs[self.label_key]

        # Tokenize with is_split_into_words=True so the tokenizer
        # knows word boundaries and word_ids() works correctly.
        words_batch = [t.split(" ") for t in texts]
        encoding = self.tokenizer(
            words_batch,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Convert word-level label strings to indices, then align
        # to subword tokens. Positions that should be ignored during
        # loss (special tokens, non-first subtokens, padding) get
        # IGNORE_INDEX (-100), which cross-entropy skips.
        aligned_labels = []
        for i, label_str in enumerate(label_strings):
            word_labels = [
                self.label_vocab[lbl] for lbl in label_str.split(" ")
            ]
            word_ids = encoding.word_ids(batch_index=i)
            aligned_labels.append(align_labels(word_ids, word_labels))

        labels = torch.tensor(aligned_labels, dtype=torch.long)

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        labels = labels.to(self.device)

        # Encoder -> dropout -> classifier (per-token logits)
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        logits = self.classifier(self.dropout(hidden_states))

        # Token-level cross-entropy, ignoring padded/special positions.
        # We can't use BaseModel.get_loss_function() because it assumes
        # one label per sample. Instead we call cross_entropy directly
        # with ignore_index to skip special tokens and non-first subtokens.
        # Flatten + ignore_index pattern from HuggingFace's
        # BertForTokenClassification.forward().
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.num_labels),
            labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

        # Per-token probabilities via softmax.
        y_prob = torch.softmax(logits, dim=-1)

        return {
            "loss": loss,
            "logit": logits,
            "y_prob": y_prob,
            "y_true": labels,
        }

    def deidentify(self, text: str, redact: str = "[REDACTED]") -> str:
        """Replace PHI in a clinical note with a redaction marker.

        Args:
            text: Raw clinical note as a string.
            redact: Replacement string for PHI tokens.

        Returns:
            The note with PHI tokens replaced.

        Example::
            >>> model.deidentify("Patient John Smith was seen")
            'Patient [REDACTED] [REDACTED] was seen'
        """
        words = text.split()
        # Forward pass with dummy labels (all O) since we only
        # need predictions, not loss.
        dummy_labels = " ".join(["O"] * len(words))
        self.eval()
        with torch.no_grad():
            result = self(text=[text], labels=[dummy_labels])

        preds = result["logit"][0].argmax(dim=-1)
        y_true = result["y_true"][0]

        # Map predictions back to words using the non-ignored positions.
        word_idx = 0
        output = []
        for j in range(len(preds)):
            if y_true[j].item() == IGNORE_INDEX:
                continue
            if preds[j].item() != 0:  # non-O = PHI
                output.append(redact)
            else:
                output.append(words[word_idx])
            word_idx += 1

        return " ".join(output)
