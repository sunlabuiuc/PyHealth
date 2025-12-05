"""
Authors: Anish Sao, Kristopher Iotov
NetIDs: sao3, kiotov2

Paper: LabTOP: A Unified Model for Lab Test Outcome Prediction
Paper Link: https://arxiv.org/abs/2502.14259

Description:
    This PyHealth example demonstrates the core ideas of the LabTOP architecture
    proposed in the above paper. It uses a lightweight GPT-2 model (via
    HuggingFace Transformers) with a small configuration (2 layers, 128 dim)
    to illustrate the key innovations of LabTOP:

    1. Digit-wise tokenization:
        Continuous-valued lab measurements (creatinine = 1.23 mg/dL)
        are encoded as sequences of digit characters ['1', '.', '2', '3'],
        enabling precise numerical modeling with an extremely small vocabulary.

    2. Lab-aware vocabulary:
        Special tokens are built for demographic fields and lab item
        identifiers (<|lab_50912|> for creatinine), mirroring the
        structure of the original LabTOP model.

    3. Autoregressive forecasting:
        The GPT-2 decoder predicts the next lab value digit-by-digit from a
        patient's tokenized history. Evaluation uses Mean Absolute Error (MAE)
        by decoding the generated digit sequence back into a float.

    4. Synthetic ICU dataset:
        To keep this example reproducible without requiring MIMIC-IV access,
        we generate realistic synthetic creatinine trajectories with
        demographic context.

Note:
    This example is not a full reproduction of the LabTOP training pipeline.
    Instead, it provides a simplified educational version suitable for PyHealth
    and CS598 DL4H Option 4 requirements.

Requirements:
    pip install torch transformers pyhealth pandas numpy

Usage:
    python labtop_demo.py
"""


from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from pyhealth.datasets import SampleDataset, get_dataloader, split_by_patient
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LAB_ITEMS = {
    50912: "creatinine",
    50931: "glucose",
    50971: "potassium",
}

RANDOM_SEED = 42
NUM_SYNTHETIC_PATIENTS = 100
MAX_SEQ_LENGTH = 128
DIGIT_PRECISION = 2


@dataclass
class ExperimentConfig:
    """Configuration for the LabTOP training experiment."""
    n_patients: int = NUM_SYNTHETIC_PATIENTS
    max_seq_len: int = MAX_SEQ_LENGTH
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-4
    embedding_dim: int = 128
    n_layers: int = 2
    n_heads: int = 2
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# DigitWise Tokenization
# --------------------------------------------------------------------------- #


class DigitWiseTokenizer:
    """Tokenizer that converts numeric lab values to digit sequences."""

    def __init__(self, precision: int = DIGIT_PRECISION) -> None:
        self.precision = precision
        self.special_tokens = {
            "PAD": "<|pad|>", "EOS": "<|endoftext|>", "SEP": "<|sep|>",
            "LAB": "<|lab|>", "AGE": "<|age|>", "GENDER_M": "<|male|>", 
            "GENDER_F": "<|female|>",
        }
        self.digit_tokens = [str(i) for i in range(10)] + [".", "-"]
        
        self.vocab = {}
        self.id_to_token = {}
        
        idx = 0
        for tok in self.special_tokens.values():
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok
            idx += 1
        for tok in self.digit_tokens:
            self.vocab[tok] = idx
            self.id_to_token[idx] = tok
            idx += 1

    def number_to_tokens(self, value: float) -> List[str]:
        """Convert a numeric value to a list of digit tokens.

        Called during encoding of lab values.

        Args:
            value: The numeric lab value to tokenize.

        Returns:
            List of single-character strings representing digits.
        """
        value = round(float(value), self.precision)
        formatted = f"{value:.{self.precision}f}"
        return list(formatted)

    def tokens_to_number(self, tokens: List[str]) -> Optional[float]:
        """Convert digit tokens back to a numeric value.

        Called during decoding of model predictions.

        Args:
            tokens: List of single-character digit strings.

        Returns:
            The reconstructed float, or None if parsing fails.
        """
        try:
            return float("".join(tokens))
        except ValueError:
            return None

    def __len__(self) -> int:
        return len(self.vocab)


class LabTOPVocabulary:
    """Extended vocabulary including lab-specific item tokens."""

    def __init__(self, lab_items: List[int], tokenizer: DigitWiseTokenizer):
        self.tokenizer = tokenizer
        self.vocab = dict(tokenizer.vocab)
        self.id_to_token = dict(tokenizer.id_to_token)

        idx = len(self.vocab)
        for lab_id in sorted(set(lab_items)):
            tok = f"<|lab_{lab_id}|>"
            if tok not in self.vocab:
                self.vocab[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1
        
        self.pad_id = self.vocab[tokenizer.special_tokens["PAD"]]
        self.eos_id = self.vocab[tokenizer.special_tokens["EOS"]]

    def encode_demographics(self, age: int, gender: str) -> List[int]:
        """Encode patient demographics into token IDs.

        Called in process_data() to build input sequences.

        Args:
            age: Patient age in years.
            gender: Patient gender, "M" or "F".

        Returns:
            List of token IDs representing the demographics.
        """
        tokens = [self.tokenizer.special_tokens["AGE"]] + \
                 self.tokenizer.number_to_tokens(age)
        if gender == "M":
            tokens.append(self.tokenizer.special_tokens["GENDER_M"])
        else:
            tokens.append(self.tokenizer.special_tokens["GENDER_F"])
        return [self.vocab[t] for t in tokens]

    def encode_lab_event(self, code: int, value: float) -> List[int]:
        """Encode a single lab event into token IDs.

        Called in process_data() to build input sequences.

        Args:
            code: Lab item ID (MIMIC-IV itemid).
            value: The numeric lab measurement.

        Returns:
            List of token IDs representing the lab event.
        """
        tokens = [
            self.tokenizer.special_tokens["LAB"],
            f"<|lab_{code}|>"
        ] + self.tokenizer.number_to_tokens(value) + \
        [self.tokenizer.special_tokens["SEP"]]
        return [self.vocab[t] for t in tokens]

    def decode_ids_to_number(self, token_ids: List[int]) -> Optional[float]:
        """Extract a numeric value from generated token IDs.

        Called in evaluate_mae() to decode model predictions.

        Args:
            token_ids: List of token IDs from model generation.

        Returns:
            The decoded float value, or None if invalid.
        """
        tokens = [self.id_to_token.get(i, "") for i in token_ids]
        digits = []
        for t in tokens:
            if t in self.tokenizer.digit_tokens:
                digits.append(t)
            elif t == self.tokenizer.special_tokens["SEP"]:
                break  # Stop at separator
            elif t in self.tokenizer.special_tokens.values():
                continue  # Skip special tokens
        
        return self.tokenizer.tokens_to_number(digits)

    def __len__(self) -> int:
        return len(self.vocab)


# --------------------------------------------------------------------------- #
# LabTOP Model
# --------------------------------------------------------------------------- #


class LabTOP(BaseModel):
    """LabTOP: GPT-2 based lab value prediction."""

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "regression",  # Used by Trainer logic, though it's LM
        embedding_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 2,
        max_seq_length: int = MAX_SEQ_LENGTH,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.max_seq_length = max_seq_length
        self.tokenizer = DigitWiseTokenizer(precision=DIGIT_PRECISION)
        self.vocab: Optional[LabTOPVocabulary] = None
        self.gpt2 = None

        # Placeholder config
        self.config = GPT2Config(
            vocab_size=100,
            n_positions=max_seq_length,
            n_embd=embedding_dim,
            n_layer=n_layers,
            n_head=n_heads,
        )

    def build_vocabulary(self, lab_items: List[int]) -> None:
        """Initialize vocabulary and GPT-2 model with lab item tokens.

        Must be called before forward() or generate().

        Args:
            lab_items: List of lab item IDs to include in vocabulary.
        """
        self.vocab = LabTOPVocabulary(lab_items, self.tokenizer)
        self.config.vocab_size = len(self.vocab)
        self.config.pad_token_id = self.vocab.pad_id
        self.config.eos_token_id = self.vocab.eos_id
        self.gpt2 = GPT2LMHeadModel(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the GPT-2 language model.

        Called by PyHealth Trainer during training.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len].
            attention_mask: Attention mask of shape [batch_size, seq_len].
            labels: Target token IDs for loss computation.

        Returns:
            Dict with "logits" and optionally "loss".
        """
        if self.gpt2 is None:
            raise RuntimeError("Model not initialized. Call build_vocabulary().")

        outputs = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # GPT2 calculates LM loss automatically if labels provided
            return_dict=True,
        )

        result = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        
        return result

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate tokens autoregressively.

        Called in evaluate_mae() to predict lab values.

        Args:
            input_ids: Prompt token IDs of shape [batch_size, seq_len].
            **kwargs: Additional arguments passed to GPT-2 generate.

        Returns:
            Generated token IDs including the prompt.
        """
        return self.gpt2.generate(
            input_ids=input_ids,
            pad_token_id=self.vocab.pad_id,
            eos_token_id=self.vocab.eos_id,
            **kwargs
        )


# --------------------------------------------------------------------------- #
# Synthetic Data & Helpers
# --------------------------------------------------------------------------- #


def generate_synthetic_data(n_patients: int = 100) -> pd.DataFrame:
    """Generate synthetic patient data with creatinine trajectories.

    Creates fake ICU patient records for demonstration without MIMIC access.

    Args:
        n_patients: Number of synthetic patients to generate.

    Returns:
        DataFrame with columns: patient_id, age, gender, lab_history, target_value.
    """
    random.seed(RANDOM_SEED)
    records = []
    for pid in range(n_patients):
        age = random.randint(50, 80)
        gender = random.choice(["M", "F"])
        
        # Simple rising trend (worsening kidney function)
        base_val = random.uniform(0.8, 1.2)
        trajectory = []
        for t in range(5):
            val = base_val + (t * 0.1) + random.uniform(-0.05, 0.05)
            trajectory.append((t, round(max(0.1, val), 2)))

        # History is first 4 points, Target is 5th point
        history = [
            {"code": 50912, "value": v, "timestamp": t} 
            for t, v in trajectory[:-1]
        ]
        target_value = trajectory[-1][1]

        records.append({
            "patient_id": f"P{pid}",
            "age": age, "gender": gender,
            "lab_history": history,
            "target_value": target_value
        })
    return pd.DataFrame(records)


def process_data(df: pd.DataFrame, vocab: LabTOPVocabulary) -> SampleDataset:
    """Convert patient DataFrame into PyHealth SampleDataset.

    Tokenizes demographics and lab history for model input.

    Args:
        df: DataFrame from generate_synthetic_data().
        vocab: Initialized LabTOPVocabulary instance.

    Returns:
        SampleDataset ready for PyHealth dataloaders.
    """
    samples = []
    for _, row in df.iterrows():
        # Context tokens
        tokens = vocab.encode_demographics(row["age"], row["gender"])
        for event in row["lab_history"]:
            tokens.extend(vocab.encode_lab_event(event["code"], event["value"]))
        
        # Labels (Shifted context + Target)
        # We append the target marker <|lab|> <|lab_code|> to prompt generation
        target_prompt = [vocab.tokenizer.special_tokens["LAB"], "<|lab_50912|>"]
        target_prompt_ids = [vocab.vocab[t] for t in target_prompt]
        
        # Full input for training (Context + Prompt + Target Value)
        target_val_ids = vocab.tokenizer.number_to_tokens(row["target_value"]) + \
                         [vocab.tokenizer.special_tokens["SEP"]]
        target_val_ids = [vocab.vocab.get(t, vocab.pad_id) for t in target_val_ids]

        input_ids = tokens + target_prompt_ids + target_val_ids
        labels = [-100] * len(tokens) + [-100] * len(target_prompt_ids) + target_val_ids

        # Truncate/Pad
        max_len = MAX_SEQ_LENGTH
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        mask = [1] * len(input_ids)

        while len(input_ids) < max_len:
            input_ids.append(vocab.pad_id)
            labels.append(-100)
            mask.append(0)

        samples.append({
            "patient_id": row["patient_id"],
            "visit_id": f"{row['patient_id']}_v0",
            "input_ids": input_ids,
            "attention_mask": mask,
            "label": labels,
            "target_value_ground_truth": row["target_value"],  # Stored for eval
            "prompt_length": len(tokens) + len(target_prompt_ids)
        })

    return SampleDataset(samples, dataset_name="synthetic", task_name="lab_pred")


# --------------------------------------------------------------------------- #
# Custom Evaluation (MAE)
# --------------------------------------------------------------------------- #


def evaluate_mae(model: LabTOP, dataset: SampleDataset, device: str = "cpu") -> float:
    """Evaluate model using Mean Absolute Error on generated predictions.

    Runs autoregressive generation and decodes digit tokens to floats.

    Args:
        model: Trained LabTOP model instance.
        dataset: Test SampleDataset with ground truth values.
        device: Device for inference ("cpu" or "cuda").

    Returns:
        Mean Absolute Error between predictions and ground truth.
    """
    model.gpt2.to(device)
    model.gpt2.eval()
    
    absolute_errors = []
    print(f"Generating predictions for {len(dataset)} test samples...")

    for i, sample in enumerate(dataset):
        # Prepare Input (Truncate to just the prompt part)
        prompt_len = sample["prompt_length"]
        input_ids = torch.tensor([sample["input_ids"][:prompt_len]]).to(device)
        ground_truth = sample["target_value_ground_truth"]

        # Generate
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=8)
        
        # Decode only the new tokens
        generated_ids = output_ids[0, prompt_len:].tolist()
        pred_val = model.vocab.decode_ids_to_number(generated_ids)

        if pred_val is not None:
            err = abs(pred_val - ground_truth)
            absolute_errors.append(err)
        
        if i < 3:  # Debug print first few
            print(f"  Sample {i}: True={ground_truth}, Pred={pred_val}")

    mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
    return mae


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    print("=== LabTOP Synthetic Demo ===")
    
    # 1. Generate Data
    df = generate_synthetic_data(NUM_SYNTHETIC_PATIENTS)
    
    # 2. Build Model & Vocab
    lab_ids = list(LAB_ITEMS.keys())
    model = LabTOP(None, ["input_ids"], "label")  # Dataset placeholder
    model.build_vocabulary(lab_ids)
    
    # 3. Create Dataset
    full_ds = process_data(df, model.vocab)
    train_ds, _, test_ds = split_by_patient(full_ds, [0.8, 0.1, 0.1])
    
    # 4. Train (Loss Only)
    # Note: We set metrics=None because roc_auc fails on LM logits
    trainer = Trainer(model=model, metrics=None) 
    trainer.train(
        train_dataloader=get_dataloader(train_ds, batch_size=8, shuffle=True),
        val_dataloader=get_dataloader(test_ds, batch_size=8, shuffle=False),
        epochs=3,
        monitor="loss"
    )

    # 5. Custom Evaluation
    print("\n=== Evaluating Mean Absolute Error (MAE) ===")
    mae = evaluate_mae(model, test_ds)
    print(f"\n✅ Final Test MAE: {mae:.4f}")
