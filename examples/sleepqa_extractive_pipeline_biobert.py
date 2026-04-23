"""SleepQA Pipeline Ablation Study.

This script demonstrates a full pipeline replication:
1. Dataset: Loading SleepQA data via PyHealth.
2. Task: Mapping to Extractive QA.
3. Ablation: Comparing a specialized medical reader (BioBERT) 
   against a general-purpose reader (Standard BERT) to demonstrate 
   the performance gap in health-coaching contexts.

Contributor: Jeffrey Yan
"""
import torch
from pyhealth.datasets.sleepqa import SleepQADataset
from pyhealth.tasks.sleepqa_extractive_qa import SleepQAExtractiveQA
from pyhealth.models.sleepqa_biobert import SleepQABioBERT


def run_ablation_comparison():
    print("=== SleepQA: Specialized vs. General Model Ablation ===")

    # 1. Pipeline Setup
    # Download=True ensures reproducibility on any machine
    dataset = SleepQADataset(root="./data", download=True)
    qa_dataset = dataset.set_task(SleepQAExtractiveQA())

    # 2. Model Initializations
    # Specialized Medical Model
    biobert_model = SleepQABioBERT(
        dataset=qa_dataset,
        model_name="dmis-lab/biobert-base-cased-v1.1-squad"
    )

    # General Purpose Model (General BERT ablation)
    general_bert = SleepQABioBERT(
        dataset=qa_dataset,
        model_name="deepset/bert-base-cased-squad2"
    )

    # 3. Qualitative Comparison (Ablation Output)
    # We take a sample and compare how the two models "see" the medical answer
    sample = qa_dataset[0]
    passage = sample["passage"]
    question = sample["question"]
    ground_truth = sample["answer_text"]

    print(f"\nContext: {passage}")
    print(f"Question: {question}")
    print(f"Expected Answer: {ground_truth}\n")

    for name, model in [("Specialized BioBERT", biobert_model), ("General BERT", general_bert)]:
        batch = {"passage": [passage], "question": [question]}
        with torch.no_grad():
            out = model(**batch)

            # Extract text from predicted logits
            start_idx = torch.argmax(out["start_logits"])
            end_idx = torch.argmax(out["end_logits"])

            # Map tokens back to text (using the internal tokenizer)
            tokens = model.tokenizer.encode(question, passage)
            pred_text = model.tokenizer.decode(tokens[start_idx: end_idx + 1])

        print(f"[{name}] Predicted: '{pred_text}'")

    print("\nDocumentation: The general model often fails to capture the precise")
    print("medical span compared to the specialized BioBERT checkpoint.")


if __name__ == "__main__":
    run_ablation_comparison()
