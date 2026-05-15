"""Merged Hurtful Words Experiments - MIMIC-III (BERT Version).

This script evaluates how various text-based ablations (Redaction, Limited 
Context) affect the fairness of mortality predictions using a 
Bio_ClinicalBERT model trained on the MIMIC-III dataset.

Ablations:
    - Baseline: Training/Testing on raw clinical notes.
    - Redaction: Masking demographic tokens (gender/race).
    - Limited: Truncating notes to the first 100 words.
"""

import re
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import HurtfulWordsMortalityTask
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def compute_fairness_metrics(samples):
    """Computes the demographic parity gap using mean predicted probabilities.

    Args:
        samples: A list of dictionaries containing 'intersectional_group' 
            and 'mortality_prob'.

    Returns:
        A dictionary containing the 'demographic_parity_gap' and raw 
        'group_means'.
    """
    group_stats = defaultdict(list)
    for s in samples:
        group = s.get("intersectional_group", "UNKNOWN")
        # Extract the probability score assigned by the BERT model
        group_stats[group].append(s.get("mortality_prob", 0.0))
    
    group_means = {g: np.mean(p) for g, p in group_stats.items()}
    
    if len(group_means) > 1:
        # Gap is calculated as the difference between max and min group risk
        gap = max(group_means.values()) - min(group_means.values())
    else:
        gap = 0.0
        
    return {"demographic_parity_gap": gap, "group_means": group_means}


class CounterfactualAugmentor:
    """Helper for swapping demographic tokens to create counterfactual text."""
    
    TOKEN_SWAPS = {
        "he": "she", "she": "he", "male": "female", "female": "male", 
        "black": "white", "white": "black"
    }

    @staticmethod
    def swap_text(text: str) -> str:
        """Swaps gender and race tokens in a string using regex."""
        for original, replacement in CounterfactualAugmentor.TOKEN_SWAPS.items():
            text = re.sub(rf"\b{original}\b", replacement, text, flags=re.IGNORECASE)
        return text


class TextRedactor:
    """Helper for masking explicit demographic identifiers in text."""
    
    PATTERNS = [
        r"\b(he|she|male|female|man|woman)\b", 
        r"\b(black|white|asian|hispanic)\b"
    ]

    @staticmethod
    def redact_text(text: str) -> str:
        """Replaces demographic tokens with a generic [REDACTED] tag."""
        for pattern in TextRedactor.PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text
    

class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the CrossEntropyLoss."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Overrides loss computation to handle class imbalance."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Weighted for ~5.5% mortality rate: [Survival Weight, Mortality Weight]
        weights = torch.tensor([1.0, 18.0]).to(logits.device)
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def run_bert_pipeline(train_samples, test_samples, 
                      model_name="emilyalsentzer/Bio_ClinicalBERT"):
    """Trains ClinicalBERT and returns samples updated with risk probabilities.

    Args:
        train_samples: List of dicts for model fine-tuning.
        test_samples: List of dicts for evaluation.
        model_name: The HuggingFace model checkpoint string.

    Returns:
        The test_samples list with 'prediction' and 'mortality_prob' keys added.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, 
                         max_length=512)

    train_ds = Dataset.from_dict({
        "text": [s.get("clinical_notes", "") for s in train_samples],
        "label": [int(s.get("mortality", 0)) for s in train_samples]
    }).map(tokenize, batched=True)
    
    groups = [s.get("intersectional_group") for s in test_samples]
    print(f"Unique groups in test set: {set(groups)}")
    print(f"Group counts: {pd.Series(groups).value_counts()}")

    test_ds = Dataset.from_dict({
        "text": [s.get("clinical_notes", "") for s in test_samples]
    }).map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="./results", 
        per_device_train_batch_size=8, 
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
        report_to="none"
    )

    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    # Inference logic
    raw_preds = trainer.predict(test_ds)
    logits = torch.tensor(raw_preds.predictions)
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].numpy()
    
    # Use top 15% risk threshold for hard predictions to ensure non-zero classes
    threshold = np.percentile(probs, 85) if len(probs) > 0 else 0.5
    predictions = (probs >= threshold).astype(int)
    
    for i, sample in enumerate(test_samples):
        sample["prediction"] = int(predictions[i])
        sample["mortality_prob"] = float(probs[i])
    
    return test_samples


def main():
    """Executes the BERT training and fairness ablation study."""
    print("Loading Dataset...")
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["NOTEEVENTS", "ADMISSIONS", "PATIENTS"],
        dev=True,
    )
    task = HurtfulWordsMortalityTask()
    samples = list(base_dataset.set_task(task))
    
    # Simple 80/20 train-test split
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    results = []

    # Baseline scenario: raw text
    print("\nRunning BERT Baseline...")
    evaluated_baseline = run_bert_pipeline(train_samples, test_samples)
    
    dist = pd.Series([s['prediction'] for s in evaluated_baseline]).value_counts()
    print(f"Prediction distribution:\n{dist}")
    results.append({"variation": "baseline", 
                    **compute_fairness_metrics(evaluated_baseline)})

    # Redaction ablation: mask demographic markers
    print("\nRunning Redaction Ablation...")
    redacted_test = []
    for s in test_samples:
        s_copy = s.copy()
        s_copy["clinical_notes"] = TextRedactor.redact_text(s_copy["clinical_notes"] or "")
        redacted_test.append(s_copy)
    
    evaluated_redacted = run_bert_pipeline(train_samples, redacted_test)
    results.append({"variation": "redacted", 
                    **compute_fairness_metrics(evaluated_redacted)})

    # Limited Context ablation: first 100 words only
    print("\nRunning Limited Features Ablation (100 words)...")
    limited_test = []
    for s in test_samples:
        s_copy = s.copy()
        text = s_copy.get("clinical_notes", "") or ""
        s_copy["clinical_notes"] = " ".join(text.split()[:100])
        limited_test.append(s_copy)
    
    evaluated_limited = run_bert_pipeline(train_samples, limited_test)
    results.append({"variation": "limited", 
                    **compute_fairness_metrics(evaluated_limited)})

    print("\n" + "="*30)
    print("FINAL FAIRNESS RESULTS")
    print("="*30)
    df = pd.DataFrame(results)
    print(df[["variation", "demographic_parity_gap"]])

"""
Experiment Results with these settings:

Table: Demographic Parity Analysis of Mortality Predictions
Experimental Variation	Parity Gap (ΔP)	Relative Bias	Core Analysis
Baseline	0.0011	1.0x	Reference State: Full context allows the model to prioritize physiological data over demographics, resulting in the highest level of fairness.
Redacted	0.0013	1.18x	Proxy Effect: Masking demographic labels fails to reduce bias. The model likely utilizes "proxy" clinical terms to reconstruct hidden identity markers.
Limited Features	0.0081	7.36x	Contextual Starvation: Truncating text to 100 words forces the model to rely on the demographic-heavy "Admission Header," causing bias to spike by over 700%.    
"""

if __name__ == "__main__":
    main()